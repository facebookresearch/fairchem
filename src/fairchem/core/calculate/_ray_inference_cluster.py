"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Ray cluster utilities for SLURM and local environments.

This module provides context managers for starting and managing Ray clusters:
- get_slurm_ray_cluster: Start a Ray cluster on SLURM with automatic cleanup
- get_local_ray_cluster: Start a local Ray cluster for testing/development
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import yaml

from fairchem.core.launchers.cluster.ray_cluster import RayCluster
from fairchem.core.units.mlip_unit.batch_server import (
    setup_batch_predict_server,
    setup_multiplexed_batch_predict_server,
    wait_for_serve_ready,
)

logger = logging.getLogger(__name__)


def _find_free_localhost_port() -> int:
    """Find an available port bound to localhost only.

    Binding to 127.0.0.1 avoids exposing ephemeral probe sockets on all
    network interfaces.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def recursive_dict_merge(*dicts: dict) -> dict:
    """
    Recursively merge dictionaries, later values override earlier ones.
    """
    result = {}
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = recursive_dict_merge(result[key], value)
            else:
                result[key] = value
    return result


def _resolve_serve_configs(
    cluster_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extract ``deployment_config`` and ``batch_config`` from the cluster
    config, resolving workflow conventions before they reach Ray Serve:

    - ``autoscaling_config.max_replicas`` / ``min_replicas`` as floats are
      interpreted as multipliers of the cluster's replica capacity
      (``num_workers * gpus_per_node // num_gpus_per_replica`` when the
      deployment requests GPUs, else ``num_workers``). Integers pass
      through unchanged. ``min_replicas`` as a float ``<= 1.0`` is a
      fraction of the resolved ``max_replicas``.
    - ``serve_log_level`` (top-level) is folded into
      ``deployment_config.logging_config`` unless the caller already set
      ``logging_config`` explicitly.
    """
    from ray import serve as _serve

    deployment_config = dict(cluster_config.get("deployment_config") or {})
    batch_config = dict(cluster_config.get("batch_config") or {})

    # Replica capacity ceiling: when the deployment binds to GPUs, the
    # cluster can host at most (total cluster GPUs / GPUs per replica)
    # replicas. Otherwise fall back to num_workers as the unit of scale.
    num_workers = int(cluster_config.get("num_workers", 1) or 1)
    gpus_per_node = int(cluster_config.get("gpus_per_node", 0) or 0)
    actor_opts = (deployment_config.get("ray_actor_options") or {})
    num_gpus_per_replica = actor_opts.get("num_gpus", 0) or 0
    if gpus_per_node > 0 and num_gpus_per_replica > 0:
        capacity = max(
            1,
            (num_workers * gpus_per_node) // max(1, int(num_gpus_per_replica)),
        )
    else:
        capacity = max(1, num_workers)

    autoscaling = dict(deployment_config.get("autoscaling_config") or {})
    if autoscaling:
        max_r = autoscaling.get("max_replicas")
        if isinstance(max_r, float):
            autoscaling["max_replicas"] = max(1, int(round(max_r * capacity)))
        min_r = autoscaling.get("min_replicas")
        if isinstance(min_r, float):
            base = autoscaling.get("max_replicas", capacity)
            if min_r <= 1.0:
                autoscaling["min_replicas"] = max(1, int(round(min_r * base)))
            else:
                autoscaling["min_replicas"] = max(1, int(round(min_r)))
        deployment_config["autoscaling_config"] = autoscaling

    serve_log_level = cluster_config.get("serve_log_level")
    if serve_log_level and "logging_config" not in deployment_config:
        deployment_config["logging_config"] = _serve.schema.LoggingConfig(
            log_level=serve_log_level
        )

    return deployment_config, batch_config


def load_update_config(
    config: str | Path,
    head_file: str | Path | None = None,
    cluster_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load and merge Ray cluster configuration.

    Loads defaults from YAML, generates a unique cluster ID, and sets up the
    head_file path where connection info will be written.

    Args:
        config: Path to YAML config file.
        head_file: Path to head.json file for connecting to existing cluster
            or where to write connection info. If None, generates path based
            on cluster UUID.
        cluster_config_overrides: Additional config overrides to merge.

    Returns:
        Merged configuration with keys:
        - All settings from YAML (partition, time_minutes, cpus_per_node, etc.)
        - cluster_id: Unique identifier for this cluster (if cluster_id generated)
        - head_file: Path to head.json with connection info
    """
    with open(config) as f:
        default_config = yaml.safe_load(f)

    auto_overrides = {}

    # Resolve rdv_dir up-front (overrides > yaml > ~/.fairray) so it can
    # drive the default head_file location.
    rdv_dir = (cluster_config_overrides or {}).get("rdv_dir") or default_config.get(
        "rdv_dir"
    ) or (Path.home() / ".fairray")
    rdv_dir = Path(rdv_dir)
    auto_overrides["rdv_dir"] = str(rdv_dir)

    if head_file is None:
        cluster_id = str(uuid.uuid4())
        logger.info(f"Specifying a Ray cluster with uuid {cluster_id}")
        auto_overrides["cluster_id"] = cluster_id

        head_file = rdv_dir / cluster_id / "head.json"
    auto_overrides["head_file"] = str(head_file)

    return recursive_dict_merge(
        default_config, auto_overrides, cluster_config_overrides
    )


def _build_cluster_config(
    config: str | Path,
    head_file: str | Path | None = None,
    num_workers: int | None = None,
    partition: str | None = None,
    gpus_per_node: int | None = None,
    cpus_per_node: int | None = None,
    time_minutes: int | None = None,
    mem_gb: int | None = None,
    env_vars: dict[str, str] | None = None,
    exclude_nodes: list[str] | None = None,
    slurm_constraint: str | None = None,
    slurm_additional_parameters: dict[str, Any] | None = None,
    start_inference_server: bool = False,
    serve_log_level: str | None = None,
    cluster_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build complete cluster configuration by merging defaults, explicit params,
    and overrides.

    Args:
        config: Path to YAML config file for defaults.
        head_file: Path to head.json file.
        num_workers: Total number of SLURM jobs to submit.
        partition: SLURM partition.
        gpus_per_node: GPUs per node (0 for CPU-only).
        cpus_per_node: CPUs per node.
        time_minutes: SLURM time limit in minutes.
        mem_gb: Memory per node in GB.
        env_vars: Environment variables to set on workers.
        exclude_nodes: SLURM nodes to exclude.
        slurm_constraint: SLURM constraint (e.g., "volta32gb").
        slurm_additional_parameters: Additional SLURM parameters for submitit.
        start_inference_server: If True, start FAIRChem inference server.
        serve_log_level: Log level for Ray Serve inference server.
        cluster_config_overrides: Additional config overrides (merged last).

    Returns:
        Complete merged configuration.
    """
    explicit_overrides = {}
    if num_workers is not None:
        explicit_overrides["num_workers"] = num_workers
    if partition is not None:
        explicit_overrides["partition"] = partition
    if gpus_per_node is not None:
        explicit_overrides["gpus_per_node"] = gpus_per_node
    if cpus_per_node is not None:
        explicit_overrides["cpus_per_node"] = cpus_per_node
    if time_minutes is not None:
        explicit_overrides["time_minutes"] = time_minutes
    if mem_gb is not None:
        explicit_overrides["mem_gb"] = mem_gb
    if env_vars is not None:
        explicit_overrides["env_vars"] = env_vars
    if exclude_nodes is not None:
        explicit_overrides["exclude_nodes"] = exclude_nodes
    if slurm_constraint is not None:
        explicit_overrides["slurm_constraint"] = slurm_constraint
    if slurm_additional_parameters is not None:
        explicit_overrides["slurm_additional_parameters"] = slurm_additional_parameters
    if start_inference_server:
        explicit_overrides["start_inference_server"] = start_inference_server
    if serve_log_level is not None:
        explicit_overrides["serve_log_level"] = serve_log_level

    merged_overrides = recursive_dict_merge(
        explicit_overrides, cluster_config_overrides
    )

    return load_update_config(
        config=config,
        head_file=head_file,
        cluster_config_overrides=merged_overrides,
    )


def _build_slurm_requirements(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build SLURM requirements dict from config for submitit.

    Args:
        config: Cluster configuration.

    Returns:
        Requirements dict for RayCluster.
    """
    cpus_per_node = int(config["cpus_per_node"])

    # If the deployment is configured (start_inference_server=True),
    # make sure each SLURM worker has enough CPUs to host its share of
    # replicas: gpus_per_node * num_cpus_per_replica. Otherwise replicas
    # would queue forever despite having free GPUs.
    if config.get("start_inference_server", False):
        actor_opts = (
            (config.get("deployment_config") or {}).get("ray_actor_options") or {}
        )
        num_cpus_per_replica = int(actor_opts.get("num_cpus", 0) or 0)
        gpus_per_node = int(config.get("gpus_per_node", 0) or 0)
        if gpus_per_node > 0 and num_cpus_per_replica > 0:
            required = gpus_per_node * num_cpus_per_replica
            if cpus_per_node < required:
                logger.info(
                    f"Bumping cpus_per_node from {cpus_per_node} to "
                    f"{required} so each worker can host "
                    f"{gpus_per_node} replica(s) "
                    f"(num_cpus={num_cpus_per_replica} each)."
                )
                cpus_per_node = required

    requirements = {
        "slurm_partition": config["partition"],
        "cpus_per_task": cpus_per_node,
        "slurm_time": config["time_minutes"],
        "mem_gb": config["mem_gb"],
    }

    if config.get("gpus_per_node", 0) > 0:
        requirements["gpus_per_node"] = config["gpus_per_node"]

    if config.get("exclude_nodes"):
        requirements["slurm_exclude"] = ",".join(config["exclude_nodes"])

    if config.get("slurm_constraint"):
        requirements["slurm_constraint"] = config["slurm_constraint"]

    if config.get("slurm_additional_parameters"):
        requirements.update(config["slurm_additional_parameters"])

    return requirements


# TODO move this and other setup somewhere else
def start_ray_cluster(
    config: dict[str, Any],
    return_cluster: bool = False,
) -> str | tuple[str, Any]:
    """
    Start a Ray cluster with the given configuration.

    Helper that handles cluster creation and waiting for head node.

    Args:
        config: Complete cluster configuration from _build_cluster_config.
        return_cluster: If True, return (head_file, cluster) tuple for
            lifecycle management. If False, return just head_file string.

    Returns:
        Path to head.json file, or (head_file, RayCluster) if
        return_cluster=True.
    """
    log_dir = Path(
        os.environ.get("RAY_PREFECT_LOG_DIR", Path.home() / "ray_prefect_logs")
    )

    requirements = _build_slurm_requirements(config)

    rdv_dir_cfg = config.get("rdv_dir")
    cluster = RayCluster(
        log_dir=log_dir,
        rdv_dir=Path(rdv_dir_cfg) if rdv_dir_cfg else None,
        cluster_id=config.get("cluster_id"),
        worker_wait_timeout_seconds=config.get("worker_wait_timeout_seconds", 300),
        temp_dir_template=config.get("temp_dir_template"),
    )

    cluster.start_head(
        requirements=requirements,
        name="ray_cluster",
        enable_client_server=True,
    )

    if config["num_workers"] > 1:
        cluster.start_workers(
            num_workers=config["num_workers"],
            requirements=requirements,
            name="ray_cluster",
        )

    head_file_path = cluster.state._head_json
    logger.info(f"Waiting for Ray cluster (head file: {head_file_path})...")
    while not cluster.state.is_head_ready():
        time.sleep(10)

    head_info = cluster.state.head_info()
    logger.info(f"Ray cluster ready at {head_info.hostname}:{head_info.port}")
    logger.info(f"Head file: {head_file_path}")

    if return_cluster:
        return str(head_file_path), cluster
    return str(head_file_path)


@contextmanager
def get_slurm_ray_cluster(
    config: str | Path | None = None,
    num_workers: int = 1,
    partition: str | None = None,
    gpus_per_node: int | None = None,
    cpus_per_node: int | None = None,
    time_minutes: int | None = None,
    mem_gb: int | None = None,
    env_vars: dict[str, str] | None = None,
    exclude_nodes: list[str] | None = None,
    slurm_constraint: str | None = None,
    slurm_additional_parameters: dict[str, Any] | None = None,
    start_inference_server: bool = False,
    predict_unit: Any = None,
    serve_log_level: str | None = None,
    deployment_name: str | None = None,
    cluster_config_overrides: dict[str, Any] | None = None,
):
    """
    Context manager that starts a Ray cluster on SLURM and shuts it down on
    exit.

    Starts a shared cluster that multiple jobs can connect to, and ensures
    clean shutdown when the context exits.

    If RAY_HEAD_FILE environment variable is set and the file exists,
    connects to that cluster instead of starting a new one (and does not
    shut it down on exit since we didn't create it).

    Usage::

        with get_slurm_ray_cluster(num_workers=32, gpus_per_node=1) as head_file:
            import ray
            with open(head_file) as f:
                head_info = json.load(f)
            ray.init(f"ray://{head_info['hostname']}:{head_info['client_port']}")
            # ... do work ...
        # Cluster is automatically shut down when exiting the context

    Args:
        config: Path to YAML config file for defaults.
        num_workers: Total number of SLURM jobs to submit.
        partition: SLURM partition.
        gpus_per_node: GPUs per node (0 for CPU-only).
        cpus_per_node: CPUs per node.
        time_minutes: SLURM time limit in minutes.
        mem_gb: Memory per node in GB.
        env_vars: Environment variables to set on workers.
        exclude_nodes: SLURM nodes to exclude.
        slurm_constraint: SLURM constraint for node selection.
        slurm_additional_parameters: Additional SLURM parameters for submitit.
        start_inference_server: If True, start FAIRChem inference server on
            cluster startup. Requires ``predict_unit`` to be provided.
        predict_unit: Predict unit to serve. Required when
            ``start_inference_server=True``.
        serve_log_level: Log level for Ray Serve inference server.
        deployment_name: Ray Serve application name.
        cluster_config_overrides: Additional config overrides.

    Yields:
        Path to head.json file for connecting to the cluster.
    """
    cluster = None
    manage_cluster = True
    ray_client_owned = False

    try:
        env_head_file = os.environ.get("RAY_HEAD_FILE")
        if env_head_file and Path(env_head_file).exists():
            logger.info(f"Using existing Ray cluster from RAY_HEAD_FILE: {env_head_file}")
            manage_cluster = False
            head_file = env_head_file
        else:
            cluster_config = _build_cluster_config(
                config=config,
                num_workers=num_workers,
                partition=partition,
                gpus_per_node=gpus_per_node,
                cpus_per_node=cpus_per_node,
                time_minutes=time_minutes,
                mem_gb=mem_gb,
                env_vars=env_vars,
                exclude_nodes=exclude_nodes,
                slurm_constraint=slurm_constraint,
                slurm_additional_parameters=slurm_additional_parameters,
                start_inference_server=start_inference_server,
                serve_log_level=serve_log_level,
                cluster_config_overrides=cluster_config_overrides,
            )

            head_file, cluster = start_ray_cluster(cluster_config, return_cluster=True)

            with open(head_file) as f:
                head_info = json.load(f)
            namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")
            logger.info(f"Ray cluster started with namespace: {namespace_serve_fairchem}")

            if cluster_config.get("start_inference_server", False):
                import ray

                client_address = (
                    f"ray://{head_info['hostname']}:" f"{head_info['client_port']}"
                )
                if ray.is_initialized():
                    # Caller already owns a Ray connection (e.g. a
                    # SlurmRayTaskRunner connected to a shared cluster with
                    # its own Serve app). Don't redeploy on top of it; the
                    # outer owner is responsible for the Serve app.
                    logger.info(
                        "Ray client already initialized; skipping inference "
                        "server deployment (caller owns the connection)."
                    )
                    _skip_serve_setup = True
                else:
                    logger.info(
                        f"Connecting to Ray cluster at {client_address} "
                        "to start inference server..."
                    )
                    ray.init(client_address, namespace=namespace_serve_fairchem)
                    # Remember that we own this connection so the finally
                    # block can release it; otherwise a subsequent
                    # get_slurm_ray_cluster() call in the same process would
                    # fail with "client has already connected" when the dead
                    # connection from the prior cluster lingers.
                    ray_client_owned = True
                    _skip_serve_setup = False

                deployment_config, batch_config = _resolve_serve_configs(cluster_config)

                # Single-model and multiplexed deployments use distinct app
                # names so a consumer can target the right one and the
                # readiness wait is unambiguous.
                resolved_deployment_name = deployment_name
                if resolved_deployment_name is None:
                    resolved_deployment_name = (
                        "multiplexed-predict-server"
                        if predict_unit is None
                        else "predict-server"
                    )

                if _skip_serve_setup:
                    pass
                elif predict_unit is None:
                    @ray.remote
                    def _setup_multiplexed_serve_remote(dep_name, dep_cfg, batch_cfg):
                        setup_multiplexed_batch_predict_server(
                            deployment_config=dep_cfg,
                            batch_config=batch_cfg,
                            deployment_name=dep_name,
                        )
                        return True

                    logger.info(
                        "Initializing multiplexed FAIRChem inference server "
                        "deployment (no predict_unit provided)..."
                    )
                    ray.get(
                        _setup_multiplexed_serve_remote.remote(
                            resolved_deployment_name,
                            deployment_config,
                            batch_config,
                        )
                    )
                else:
                    @ray.remote
                    def _setup_serve_remote(predict_unit_ref, dep_name, dep_cfg, batch_cfg):
                        pu = ray.get(predict_unit_ref)
                        setup_batch_predict_server(
                            pu,
                            deployment_config=dep_cfg,
                            batch_config=batch_cfg,
                            deployment_name=dep_name,
                        )
                        return True

                    predict_unit_ref = ray.put(predict_unit)
                    logger.info("Initializing FAIRChem inference server deployment...")
                    ray.get(
                        _setup_serve_remote.remote(
                            predict_unit_ref,
                            resolved_deployment_name,
                            deployment_config,
                            batch_config,
                        )
                    )

                @ray.remote
                def _wait_for_serve_ready_remote(dep_name):
                    return wait_for_serve_ready(app_name=dep_name)

                if not _skip_serve_setup:
                    logger.info(
                        "Inference server deployment complete, " "verifying readiness..."
                    )
                    ray.get(_wait_for_serve_ready_remote.remote(resolved_deployment_name))
                    logger.info("Inference server ready and accepting requests")

        yield head_file
    finally:
        if ray_client_owned:
            import ray
            try:
                ray.shutdown()
                logger.info("Released Ray client connection.")
            except Exception as e:
                logger.warning(f"Error releasing Ray client connection: {e}")
        if cluster is not None and manage_cluster:
            logger.info("Shutting down Ray cluster...")
            try:
                cluster.shutdown()
                logger.info("Ray cluster shut down successfully")
            except Exception as e:
                logger.warning(f"Error during Ray cluster shutdown: {e}")


@contextmanager
def get_slurm_inference_cluster(*args, **kwargs):
    """Descriptive alias for get_slurm_ray_cluster.

    This helper makes intent explicit: it is primarily used to provision
    inference-serving Ray clusters. Kept as a wrapper for backwards
    compatibility while preserving the implementation in one place.
    """
    with get_slurm_ray_cluster(*args, **kwargs) as head_file:
        yield head_file


@contextmanager
def get_local_ray_cluster(
    head_file: str | Path | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    start_inference_server: bool = True,
    predict_unit: Any = None,
    log_level: str = "WARNING",
    deployment_name: str | None = None,
    deployment_config: dict[str, Any] | None = None,
    batch_config: dict[str, Any] | None = None,
):
    """
    Context manager that starts a local Ray cluster with optional inference
    server.

    Similar to get_slurm_ray_cluster but for local/testing use. Automatically:
    - Detects available GPUs if num_gpus is None
    - Starts Ray Serve
    - Deploys FAIRChem inference server
    - Writes head file for code that expects RAY_HEAD_FILE
    - Cleans up on exit

    Usage::

        with get_local_ray_cluster() as head_file:
            # Run code that uses Ray Serve inference
            ...

        # Or for testing, without inference server:
        with get_local_ray_cluster(start_inference_server=False):
            # Run CPU-only tests
            ...

    Args:
        head_file: Path where head.json will be written. If None, creates
            a temp file.
        num_cpus: Number of CPUs for Ray. Defaults to 8.
        num_gpus: Number of GPUs for Ray. If None, auto-detects via
            torch.cuda.
        start_inference_server: If True (default), start FAIRChem Ray Serve
            inference server. Requires ``predict_unit`` to be provided.
        predict_unit: Predict unit to serve. Required when
            ``start_inference_server=True``.
        log_level: Ray logging level.
        deployment_name: Ray Serve application name.

    Yields:
        Path to head.json file.
    """
    import ray
    from ray import serve

    # Set defaults
    if num_cpus is None:
        num_cpus = 8

    if num_gpus is None:
        try:
            import torch

            num_gpus = torch.cuda.device_count()
        except ImportError:
            num_gpus = 0

    if head_file is None:
        cluster_id = str(uuid.uuid4())
        head_file_path = Path.home() / ".fairray" / cluster_id / "head.json"
    else:
        head_file_path = Path(head_file).expanduser()

    namespace_serve_fairchem = "fairchem_inference" if start_inference_server else None

    # Find free ports for this cluster instance
    dashboard_port = _find_free_localhost_port()

    try:
        if not ray.is_initialized():
            init_kwargs = {
                "num_cpus": num_cpus,
                "ignore_reinit_error": True,
                "log_to_driver": True,
                "logging_config": ray.LoggingConfig(log_level=log_level),
                "dashboard_port": dashboard_port,
                "namespace": namespace_serve_fairchem,
            }

            if num_gpus > 0:
                init_kwargs["num_gpus"] = num_gpus
                logger.info(
                    f"Starting local Ray cluster with {num_cpus} CPUs "
                    f"and {num_gpus} GPUs (namespace: "
                    f"{namespace_serve_fairchem}, dashboard port: "
                    f"{dashboard_port})"
                )
            else:
                logger.info(
                    f"Starting local Ray cluster with {num_cpus} CPUs "
                    f"(no GPUs, namespace: {namespace_serve_fairchem}, "
                    f"dashboard port: {dashboard_port})"
                )

            ray.init(**init_kwargs)
            logger.info("Ray initialized")
            hostname = socket.gethostname()
            logger.info(f"Ray dashboard URL: http://{hostname}:{dashboard_port}")

        if start_inference_server:
            # Multiplexed vs single-model deployments get distinct app
            # names so consumers and readiness waits target the right one.
            resolved_deployment_name = deployment_name
            if resolved_deployment_name is None:
                resolved_deployment_name = (
                    "multiplexed-predict-server"
                    if predict_unit is None
                    else "predict-server"
                )

            if predict_unit is None:
                logger.info(
                    "Initializing multiplexed FAIRChem inference server "
                    "deployment (no predict_unit provided)..."
                )
                setup_multiplexed_batch_predict_server(
                    deployment_config=deployment_config,
                    batch_config=batch_config,
                    deployment_name=resolved_deployment_name,
                )
            else:
                logger.info("Initializing FAIRChem inference server deployment...")
                setup_batch_predict_server(
                    predict_unit,
                    deployment_config=deployment_config,
                    batch_config=batch_config,
                    deployment_name=resolved_deployment_name,
                )

            logger.info(
                "Inference server deployment complete, " "verifying readiness..."
            )
            wait_for_serve_ready(app_name=resolved_deployment_name)
            logger.info("Inference server ready and accepting requests")

        head_file_path.parent.mkdir(parents=True, exist_ok=True)
        head_file_path.write_text(
            json.dumps(
                {
                    "hostname": "localhost",
                    "dashboard_port": dashboard_port,
                    "local": True,
                    "num_cpus": num_cpus,
                    "num_gpus": num_gpus,
                    "namespace_serve_fairchem": namespace_serve_fairchem,
                }
            )
        )

        yield str(head_file_path)

    finally:
        logger.info("Shutting down local Ray cluster...")
        if start_inference_server:
            with suppress(Exception):
                serve.shutdown()
        ray.shutdown()

        if head_file_path.exists():
            head_file_path.unlink()
            with suppress(OSError):
                head_file_path.parent.rmdir()


@contextmanager
def get_local_inference_cluster(*args, **kwargs):
    """Descriptive alias for get_local_ray_cluster."""
    with get_local_ray_cluster(*args, **kwargs) as head_file:
        yield head_file
