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
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def recursive_dict_merge(*dicts: dict) -> dict:
    """Recursively merge dictionaries, later values override earlier ones."""
    result = {}
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_dict_merge(result[key], value)
            else:
                result[key] = value
    return result


def load_update_config(
    config: str | Path | None = None,
    head_file: str | Path | None = None,
    cluster_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load and merge Ray cluster configuration.

    Loads defaults from YAML, generates a unique cluster ID, and sets up the
    head_file path where connection info will be written.

    Parameters
    ----------
    config : str | Path, optional
        Path to YAML config file. Defaults to ray_cluster.yaml in this directory.
    head_file : str | Path, optional
        Path to head.json file for connecting to existing cluster or where to
        write connection info. If None, generates path based on cluster UUID.
    cluster_config_overrides : dict, optional
        Additional config overrides to merge.

    Returns
    -------
    dict
        Merged configuration with keys:
        - All settings from YAML (partition, time_minutes, cpus_per_node, etc.)
        - cluster_id: Unique identifier for this cluster (if cluster_id generated)
        - head_file: Path to head.json with connection info
    """
    # Use default config in local directory if not present
    if config is None:
        config = Path(__file__).parent / "ray_cluster.yaml"

    # Load the default config
    with open(config) as f:
        default_config = yaml.safe_load(f)

    auto_overrides = {}

    # Set up head_file path
    if head_file is None:
        # Generate cluster ID
        cluster_id = str(uuid.uuid4())
        logger.info(f"Specifying a Ray cluster with uuid {cluster_id}")
        auto_overrides["cluster_id"] = cluster_id

        head_file = Path.home() / ".fairray" / cluster_id / "head.json"
    auto_overrides["head_file"] = str(head_file)

    return recursive_dict_merge(default_config, auto_overrides, cluster_config_overrides)


def _build_cluster_config(
    config: str | Path | None = None,
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
    Build complete cluster configuration by merging defaults, explicit params, and overrides.

    Parameters
    ----------
    config : str | Path, optional
        Path to YAML config file for defaults
    head_file : str | Path, optional
        Path to head.json file (if connecting to existing cluster or explicit path)
    num_workers : int, optional
        Total number of SLURM jobs to submit (1 head + remaining workers)
    partition : str, optional
        SLURM partition
    gpus_per_node : int, optional
        GPUs per node (0 for CPU-only)
    cpus_per_node : int, optional
        CPUs per node
    time_minutes : int, optional
        SLURM time limit in minutes
    mem_gb : int, optional
        Memory per node in GB
    env_vars : dict, optional
        Environment variables to set on workers
    exclude_nodes : list, optional
        SLURM nodes to exclude
    slurm_constraint : str, optional
        SLURM constraint (e.g., "volta32gb" for GPU type)
    slurm_additional_parameters : dict, optional
        Additional SLURM parameters passed to submitit
    start_inference_server : bool, optional
        If True, start FAIRChem inference server on cluster
    serve_log_level : str, optional
        Log level for Ray Serve inference server (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    cluster_config_overrides : dict, optional
        Additional config overrides (merged last)

    Returns
    -------
    dict
        Complete merged configuration
    """
    # Build overrides from explicit parameters
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

    # Merge explicit overrides with cluster_config_overrides
    merged_overrides = recursive_dict_merge(explicit_overrides, cluster_config_overrides)

    # Load and merge all config
    return load_update_config(
        config=config,
        head_file=head_file,
        cluster_config_overrides=merged_overrides,
    )


def _build_slurm_requirements(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build SLURM requirements dict from config for submitit.

    Parameters
    ----------
    config : dict
        Cluster configuration

    Returns
    -------
    dict
        Requirements dict for RayCluster
    """
    requirements = {
        "slurm_partition": config["partition"],
        "cpus_per_task": config["cpus_per_node"],
        "slurm_time": config["time_minutes"],
        "mem_gb": config["mem_gb"],
    }

    if config.get("gpus_per_node", 0) > 0:
        requirements["gpus_per_node"] = config["gpus_per_node"]

    if config.get("exclude_nodes"):
        requirements["slurm_exclude"] = ",".join(config["exclude_nodes"])

    # Add GPU/hardware constraint (e.g., volta32gb)
    if config.get("slurm_constraint"):
        requirements["slurm_constraint"] = config["slurm_constraint"]

    # Merge any additional SLURM parameters
    if config.get("slurm_additional_parameters"):
        requirements.update(config["slurm_additional_parameters"])

    return requirements


def _start_ray_cluster_internal(
    config: dict[str, Any],
    return_cluster: bool = False,
) -> str | tuple[str, Any]:
    """
    Start a Ray cluster with the given configuration.

    Internal helper that handles cluster creation and waiting for head node.

    Parameters
    ----------
    config : dict
        Complete cluster configuration from _build_cluster_config
    return_cluster : bool
        If True, return (head_file, cluster) tuple for lifecycle management.
        If False, return just head_file string.

    Returns
    -------
    str or tuple
        Path to head.json file, or (head_file, RayCluster) if return_cluster=True
    """
    from fairchem.core.launchers.cluster.ray_cluster import RayCluster

    log_dir = Path(os.environ.get("RAY_PREFECT_LOG_DIR", Path.home() / "ray_prefect_logs"))

    requirements = _build_slurm_requirements(config)

    # Start cluster with the same cluster_id from config for consistency
    cluster = RayCluster(
        log_dir=log_dir,
        cluster_id=config.get("cluster_id"),
        worker_wait_timeout_seconds=config.get("worker_wait_timeout_seconds", 300),
    )

    # Start head node - uses native _ray_head_script which writes head.json
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

    # Wait for head.json to be written by _ray_head_script
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
    serve_log_level: str | None = None,
    cluster_config_overrides: dict[str, Any] | None = None,
):
    """
    Context manager that starts a Ray cluster on SLURM and shuts it down on exit.

    Starts a shared cluster that multiple jobs can connect to, and ensures
    clean shutdown when the context exits.

    If RAY_HEAD_FILE environment variable is set and the file exists,
    connects to that cluster instead of starting a new one (and does not
    shut it down on exit since we didn't create it).

    Usage:
        with get_slurm_ray_cluster(num_workers=32, gpus_per_node=1) as head_file:
            # Use head_file to connect to the cluster
            import ray
            with open(head_file) as f:
                head_info = json.load(f)
            ray.init(f"ray://{head_info['hostname']}:{head_info['client_port']}")
            # ... do work ...
        # Cluster is automatically shut down when exiting the context

    Parameters
    ----------
    config : str | Path, optional
        Path to YAML config file for defaults
    num_workers : int
        Total number of SLURM jobs to submit (1 head + remaining workers)
    partition : str, optional
        SLURM partition
    gpus_per_node : int, optional
        GPUs per node (0 for CPU-only)
    cpus_per_node : int, optional
        CPUs per node
    time_minutes : int, optional
        SLURM time limit in minutes
    mem_gb : int, optional
        Memory per node in GB
    env_vars : dict, optional
        Environment variables to set on workers
    exclude_nodes : list, optional
        SLURM nodes to exclude
    slurm_constraint : str, optional
        SLURM constraint for node selection (e.g., "volta32gb" for GPU type)
    slurm_additional_parameters : dict, optional
        Additional SLURM parameters passed to submitit (for future flexibility)
    start_inference_server : bool
        If True, start FAIRChem inference server on cluster startup
    serve_log_level : str, optional
        Log level for Ray Serve inference server (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    cluster_config_overrides : dict, optional
        Additional config overrides

    Yields
    ------
    str
        Path to head.json file for connecting to the cluster (namespace is stored in the file)
    """
    cluster = None
    manage_cluster = True

    # Check for existing cluster via environment variable
    env_head_file = os.environ.get("RAY_HEAD_FILE")
    if env_head_file and Path(env_head_file).exists():
        logger.info(f"Using existing Ray cluster from RAY_HEAD_FILE: {env_head_file}")
        manage_cluster = False
        head_file = env_head_file
    else:
        # Build complete configuration using shared helper
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

        # Start cluster using shared helper with cluster reference for shutdown
        head_file, cluster = _start_ray_cluster_internal(cluster_config, return_cluster=True)

        # Read namespace from head.json for inference server startup
        with open(head_file) as f:
            head_info = json.load(f)
        namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")
        logger.info(f"Ray cluster started with namespace: {namespace_serve_fairchem}")

        if cluster_config.get("start_inference_server", False):
            import ray

            client_address = f"ray://{head_info['hostname']}:{head_info['client_port']}"
            logger.info(f"Connecting to Ray cluster at {client_address} to start inference server...")
            ray.init(client_address, namespace=namespace_serve_fairchem)

            serve_log_level_val = cluster_config.get("serve_log_level", "WARNING")
            batch_config = cluster_config.get("batch_config")
            deployment_config = cluster_config.get("deployment_config")

            @ray.remote
            def _start_serve_remote(num_workers: int, log_level: str, batch_config: dict | None, deployment_config: dict | None):
                from fairchem.core.units.mlip_unit.batch import start_serve
                start_serve(num_workers=num_workers, log_level=log_level, batch_config=batch_config, deployment_config=deployment_config)
                return True

            @ray.remote
            def _wait_for_serve_ready_remote():
                from fairchem.core.units.mlip_unit.batch import wait_for_serve_ready
                return wait_for_serve_ready()

            num_workers_config = cluster_config.get("num_workers", 1)
            logger.info("Initializing FAIRChem inference server deployment...")
            logger.info(f"Deploying inference server to Ray Serve with {num_workers_config} max replicas (log_level={serve_log_level_val})...")
            ray.get(_start_serve_remote.remote(num_workers_config, serve_log_level_val, batch_config, deployment_config))
            logger.info("Inference server deployment complete, verifying readiness...")
            ray.get(_wait_for_serve_ready_remote.remote())
            logger.info("Inference server ready and accepting requests")

    try:
        yield head_file
    finally:
        # Only shut down the cluster if we created it
        if cluster is not None and manage_cluster:
            logger.info("Shutting down Ray cluster...")
            try:
                cluster.shutdown()
                logger.info("Ray cluster shut down successfully")
            except Exception as e:
                logger.warning(f"Error during Ray cluster shutdown: {e}")


@contextmanager
def get_local_ray_cluster(
    head_file: str | Path | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    start_inference_server: bool = True,
    log_level: str = "WARNING",
    batch_config: dict | None = None,
    deployment_config: dict | None = None,
):
    """
    Context manager that starts a local Ray cluster with optional inference server.

    Similar to get_slurm_ray_cluster but for local/testing use. Automatically:
    - Detects available GPUs if num_gpus is None
    - Starts Ray Serve
    - Deploys FAIRChem inference server
    - Writes head file for code that expects RAY_HEAD_FILE
    - Cleans up on exit

    Usage:
        with get_local_ray_cluster() as head_file:
            # Run code that uses Ray Serve inference
            ...

        # Or for testing, without inference server:
        with get_local_ray_cluster(start_inference_server=False):
            # Run CPU-only tests
            ...

    Parameters
    ----------
    head_file : str | Path, optional
        Path where head.json will be written. If None, creates a temp file.
    num_cpus : int, optional
        Number of CPUs for Ray. Defaults to 8.
    num_gpus : int, optional
        Number of GPUs for Ray. If None, auto-detects via torch.cuda.
    start_inference_server : bool
        If True (default), start FAIRChem Ray Serve inference server.
    log_level : str
        Ray logging level. Default: "WARNING"
    batch_config : dict, optional
        Batch configuration for inference server. See start_serve().
    deployment_config : dict, optional
        Deployment configuration for inference server. See start_serve().

    Yields
    ------
    str
        Path to head.json file (namespace is stored in the file)
    """
    import ray
    from ray import serve

    def find_free_port():
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    # Set defaults
    if num_cpus is None:
        num_cpus = 8

    # Auto-detect GPUs if not specified
    if num_gpus is None:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            num_gpus = 0

    # Generate head_file path and namespace if not provided
    if head_file is None:
        cluster_id = str(uuid.uuid4())
        head_file_path = Path.home() / ".fairray" / cluster_id / "head.json"
        namespace_serve_fairchem = cluster_id
    else:
        head_file_path = Path(head_file).expanduser()
        # Extract namespace from path (parent dir is cluster_id)
        namespace_serve_fairchem = head_file_path.parent.name

    # Find free ports for this cluster instance
    dashboard_port = find_free_port()

    try:
        # Initialize Ray locally with unique ports
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
                logger.info(f"Starting local Ray cluster with {num_cpus} CPUs and {num_gpus} GPUs (namespace: {namespace_serve_fairchem}, dashboard port: {dashboard_port})")
            else:
                logger.info(f"Starting local Ray cluster with {num_cpus} CPUs (no GPUs, namespace: {namespace_serve_fairchem}, dashboard port: {dashboard_port})")

            ray.init(**init_kwargs)
            logger.info("Ray initialized")
            hostname = socket.gethostname()
            logger.info(f"Ray dashboard URL: http://{hostname}:{dashboard_port}")

        # Start Ray Serve if inference server requested
        if start_inference_server:
            try:
                serve.status()
                logger.info("Ray Serve already running")
            except Exception:
                logger.info("Starting Ray Serve...")
                serve.start(detached=True)
                logger.info("Ray Serve started")

            # Start the FAIRChem inference server deployment
            logger.info("Initializing FAIRChem inference server deployment...")
            from fairchem.core.units.mlip_unit.batch import start_serve as start_fairchem_serve
            # For local clusters, use num_gpus as num_workers (1 replica per GPU)
            local_num_workers = num_gpus if num_gpus > 0 else 1
            logger.info(f"Deploying inference server to Ray Serve with {local_num_workers} max replicas (log_level={log_level})...")
            start_fairchem_serve(num_workers=local_num_workers, log_level=log_level, batch_config=batch_config, deployment_config=deployment_config)

            # Wait for the server to be fully ready
            logger.info("Inference server deployment complete, verifying readiness...")
            from fairchem.core.units.mlip_unit.batch import wait_for_serve_ready
            wait_for_serve_ready()
            logger.info("Inference server ready and accepting requests")

        # Write head file for compatibility with code expecting RAY_HEAD_FILE
        # Note: For local clusters, there's no Ray client port - use "local" flag instead
        head_file_path.parent.mkdir(parents=True, exist_ok=True)
        head_file_path.write_text(json.dumps({
            "hostname": "localhost",
            "dashboard_port": dashboard_port,
            "local": True,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "namespace_serve_fairchem": namespace_serve_fairchem,
        }))

        yield str(head_file_path)

    finally:
        logger.info("Shutting down local Ray cluster...")
        if start_inference_server:
            with suppress(Exception):
                serve.shutdown()
        ray.shutdown()

        # Clean up head file
        if head_file_path.exists():
            head_file_path.unlink()
            # Try to remove parent dir if empty
            with suppress(OSError):
                head_file_path.parent.rmdir()
