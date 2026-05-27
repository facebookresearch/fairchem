"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import ray
import torch
from ray import serve
from ray.serve.schema import ApplicationStatus

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.predict import move_tensors_to_cpu

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


# Centralized batching defaults. Kept here (not on the server class
# __init__s) so the two setup helpers can't drift apart silently.
DEFAULT_MAX_BATCH_SIZE = 512
DEFAULT_BATCH_WAIT_TIMEOUT_S = 0.1


@dataclass
class DeploymentConfig:
    """Typed mirror of the most common ``@serve.deployment`` / ``.options()`` kwargs.

    Fields default to ``None`` and are dropped before being forwarded to
    Ray Serve, so unspecified options inherit Ray Serve's own defaults
    rather than this class re-asserting them.
    """

    num_replicas: int | None = None
    ray_actor_options: dict | None = None
    # ``autoscaling_config`` accepts a dict or a ``ray.serve.schema.AutoscalingConfig``;
    # ``logging_config`` accepts a dict or ``ray.serve.schema.LoggingConfig``.
    autoscaling_config: Any = None
    max_ongoing_requests: int | None = None
    max_queued_requests: int | None = None
    max_replicas_per_node: int | None = None
    graceful_shutdown_timeout_s: float | None = None
    graceful_shutdown_wait_loop_s: float | None = None
    health_check_period_s: float | None = None
    health_check_timeout_s: float | None = None
    logging_config: Any = None
    user_config: dict | None = None
    placement_group_bundles: list | None = None
    placement_group_strategy: str | None = None
    request_router_config: Any = None

    def to_options_kwargs(self) -> dict[str, Any]:
        """Return ``{name: value}`` for fields that were explicitly set (non-None)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class BatchConfig:
    """Typed mirror of kwargs accepted by ``BatchPredictServer.__init__`` and
    ``MultiplexedBatchPredictServer.__init__``.
    """

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    batch_wait_timeout_s: float = DEFAULT_BATCH_WAIT_TIMEOUT_S
    split_oom_batch: bool = True
    # None defers to Ray Serve's ``@serve.batch`` default.
    max_concurrent_batches: int | None = None

    def to_init_kwargs(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("max_concurrent_batches") is None:
            d.pop("max_concurrent_batches")
        return d


class BatchPredictServerMixin:
    """
    Shared batched-inference logic mixed into Ray Serve deployment classes.

    This mixin is **not** decorated with ``@serve.deployment`` so that it
    can be used as a regular base class.  The concrete subclasses
    ``BatchPredictServer`` and ``MultiplexedBatchPredictServer`` apply the
    decorator themselves.
    """

    def configure_batching(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        max_concurrent_batches: int | None = None,
    ):
        # ``max_concurrent_batches`` is a decoration-time-only arg in
        # Ray Serve's ``@serve.batch`` (no runtime setter is exposed),
        # so we (re)apply the decorator here against the undecorated
        # ``_predict_impl`` bound method. ``max_batch_size`` and
        # ``batch_wait_timeout_s`` still have runtime setters, but we
        # pass them as decoration args too so the initial values are
        # correct from the first request.
        batch_kwargs: dict[str, Any] = {
            "batch_size_fn": lambda batch: sum(
                sample.natoms.sum() for sample in batch
            ).item(),
            "max_batch_size": max_batch_size,
            "batch_wait_timeout_s": batch_wait_timeout_s,
        }
        if max_concurrent_batches is not None:
            batch_kwargs["max_concurrent_batches"] = max_concurrent_batches
        self.predict = serve.batch(**batch_kwargs)(self._predict_impl)

    def get_predict_unit_attribute(self, attribute_name: str, **kwargs) -> Any:
        # Move the returned value to CPU so that callers running on
        # CPU-only Ray workers can deserialize it without requiring CUDA
        # (e.g. ``atom_refs`` typically contains tensors stored on the
        # server's device).
        return move_tensors_to_cpu(getattr(self.predict_unit, attribute_name))

    def validate_atoms_data(self, atoms_info: dict, task_name: str) -> dict:
        """
        Run the predict unit's validation and return the (possibly mutated) atoms.info.

        Validation may set defaults (e.g. charge, spin) on ``atoms.info``.
        Because the caller needs those mutations applied locally, this method
        accepts and returns the ``atoms.info`` dict rather than a full
        ``Atoms`` object.

        Args:
            atoms_info: Copy of ``atoms.info`` from the caller's Atoms object.
            task_name: Task name passed through to the predict unit.

        Returns:
            The mutated ``atoms.info`` dict with any defaults applied.
        """
        from ase import Atoms

        # Build a minimal Atoms stub just for validation — only atoms.info
        # is read/mutated by validate_atoms_data implementations.
        stub = Atoms()
        stub.info = atoms_info
        self.predict_unit.validate_atoms_data(stub, task_name)
        return stub.info

    def update_predict_unit(self, predict_unit) -> None:
        """
        Update the predict unit with a new checkpoint.

        Args:
            predict_unit: New MLIPPredictUnit instance (Ray resolves any ObjectRef
                before invoking this method, so the argument is always the actual object)
        """
        self.predict_unit = predict_unit
        logging.info("predict_unit updated")

    def _run_batched_inference(
        self,
        items: list[AtomicData],
        predict_unit: MLIPPredictUnit,
        undo_element_references: bool,
    ) -> list[dict]:
        """
        Run batched inference with OOM splitting.
        """
        data_deque: deque[list[AtomicData]] = deque([items])
        results: list[dict] = []
        while data_deque:
            oom = False
            current = data_deque.popleft()
            batch = atomicdata_list_to_batch(current)
            try:
                preds = predict_unit.predict(
                    batch, undo_element_references=undo_element_references
                )
                results.extend(self._split_predictions(preds, batch))
            except torch.OutOfMemoryError as err:
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Reduce max_batch_size or set split_oom_batch=True "
                        "to automatically split OOM batches."
                    ) from err
                if len(current) == 1:
                    raise torch.OutOfMemoryError(
                        "Out of memory for a single system left in batch."
                    ) from err
                logging.warning(
                    "Caught out of memory error. Splitting batch and retrying."
                )
                oom = True
                torch.cuda.empty_cache()
            if oom:
                mid = len(current) // 2
                data_deque.appendleft(current[mid:])
                data_deque.appendleft(current[:mid])
        return results

    async def __call__(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict:
        """
        Main entry point for inference requests.

        Args:
            data: Single AtomicData object
            undo_element_references: Whether to undo element references in predictions

        Returns:
            Prediction dictionary for this system
        """
        predictions = await self.predict(data, undo_element_references)
        return predictions

    def _split_predictions(
        self,
        predictions: dict,
        batch: AtomicData,
    ) -> list[dict]:
        """
        Split batched predictions back into individual system predictions.

        Args:
            predictions: Dictionary of batched prediction tensors
            batch: The batched AtomicData used for inference

        Returns:
            List of prediction dictionaries, one per system
        """
        split_preds = []
        for i in range(len(batch)):
            system_predictions = {}

            for key, pred in predictions.items():
                if pred.shape[0] == len(batch):
                    # Per-system prediction
                    system_predictions[key] = pred[i : i + 1]
                elif pred.shape[0] == len(batch.batch):
                    # Per-atom prediction
                    mask = batch.batch == i
                    system_predictions[key] = pred[mask]
                else:
                    raise ValueError(
                        f"Cannot split prediction for key '{key}': "
                        f"unexpected shape {pred.shape} for batch size {len(batch)} "
                        f"and num_atoms {batch.num_atoms}"
                    )

                # Move to CPU before returning so the caller (which may be a
                # CPU-only Ray worker) can deserialize the result without
                # requiring CUDA.
                if hasattr(system_predictions[key], "detach"):
                    system_predictions[key] = system_predictions[key].detach().cpu()

            split_preds.append(system_predictions)

        return split_preds


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class BatchPredictServer(BatchPredictServerMixin):
    """
    Ray Serve deployment that batches incoming inference requests
    for a single pre-loaded model.
    """

    def __init__(
        self,
        predict_unit_ref,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = True,
        max_concurrent_batches: int | None = None,
    ):
        """
        Initialize with a Ray object reference to a PredictUnit.

        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
            max_batch_size: Maximum number of atoms in a batch.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction
            split_oom_batch: If true will split batch if an OOM error is raised
            max_concurrent_batches: Max concurrent batches for the @serve.batch
                decorator. If None, uses Ray Serve's default.
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.split_oom_batch = split_oom_batch
        self.configure_batching(
            max_batch_size, batch_wait_timeout_s, max_concurrent_batches
        )

        logging.info(
            "BatchPredictServer initialized with predict_unit from object store"
        )

    async def _predict_impl(
        self, data_list: list[AtomicData], undo_element_references: bool = True
    ) -> list[dict]:
        return self._run_batched_inference(
            data_list, self.predict_unit, undo_element_references
        )

    async def is_multiplexed(self) -> bool:
        return False


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class MultiplexedBatchPredictServer(BatchPredictServerMixin):
    """
    Ray Serve deployment that supports multiplexed model loading with batching.

    Unlike ``BatchPredictServer`` which serves a single pre-loaded model,
    this deployment loads models on demand using ``@serve.multiplexed``.
    Different clients can request different models by passing a ``model_id``
    and an LRU cache keeps up to ``max_num_models_per_replica`` models
    resident on each replica.

    **Batching with per-model routing.**  ``@serve.batch`` collects requests
    from concurrent ``__call__`` invocations.  Because
    ``serve.get_multiplexed_model_id()`` is only reliable in per-request
    context (``__call__``), each request captures its ``model_id`` there and
    passes it explicitly as a second positional argument to ``predict()``.
    Inside the batch function, requests are grouped by ``model_id``, each
    group is processed with the correct cached ``predict_unit`` via
    ``await self.get_model(model_id)`` (LRU cache hit), and results are
    reassembled in original request order.
    """

    def __init__(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = True,
        max_concurrent_batches: int | None = None,
    ):
        """
        Initialize the multiplexed predict server.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction.
            split_oom_batch: If true will split batch if an OOM error is raised.
            max_concurrent_batches: Max concurrent batches for the @serve.batch
                decorator. If None, uses Ray Serve's default.
        """
        self.split_oom_batch = split_oom_batch
        self.configure_batching(
            max_batch_size, batch_wait_timeout_s, max_concurrent_batches
        )
        logging.info("MultiplexedBatchPredictServer initialized")

    async def is_multiplexed(self) -> bool:
        return True

    async def _predict_impl(
        self,
        data_list: list[AtomicData],
        model_id_list: list[str],
        undo_element_references: bool = True,
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects, grouped by model_id.

        Requests for different models accumulate in the same ``@serve.batch``
        window and are then dispatched in sequential per-model groups.  The
        ``model_id`` for each request is passed explicitly from ``__call__``
        (where ``serve.get_multiplexed_model_id()`` is still valid) rather than
        being read inside this function (where only one request context is
        active).

        Args:
            data_list: List of AtomicData objects (automatically batched by
                Ray Serve).
            model_id_list: Corresponding model IDs, one per request.
            undo_element_references: Whether to undo element references.
                Ray Serve batches this into a list; the first value is used.

        Returns:
            List of prediction dictionaries, one per input, in original order.
        """
        # @serve.batch batches all positional args; take first value for scalar
        undo_refs = (
            undo_element_references[0]
            if isinstance(undo_element_references, list)
            else undo_element_references
        )

        # Group (original_index, data) pairs by model_id
        groups: dict[str, list[tuple[int, AtomicData]]] = defaultdict(list)
        for i, (data, model_id) in enumerate(zip(data_list, model_id_list)):
            groups[model_id].append((i, data))

        results: list[dict | None] = [None] * len(data_list)

        for model_id, indexed_items in groups.items():
            predict_unit = await self.get_model(model_id)  # LRU cache hit

            indices, group_data = zip(*indexed_items)
            group_results = self._run_batched_inference(
                list(group_data), predict_unit, undo_refs
            )
            for orig_idx, pred in zip(indices, group_results):
                results[orig_idx] = pred

        return results

    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_id: str):
        """
        Load (or retrieve from cache) a predict unit by model key.

        The ``@serve.multiplexed`` decorator caches the *return value* of
        this method in an LRU cache (keyed by ``model_id``).  Returning the
        ``predict_unit`` directly means the cached object is the unit itself,
        so subsequent calls for the same ``model_id`` skip the loading code
        and return the cached unit without touching any instance state.

        Args:
            model_id: Key in the format ``"checkpoint_name_or_path:settings"``
                where ``settings`` is one of the recognized inference setting
                names (e.g. ``"default"``, ``"turbo"``) or an empty string for
                the default settings.

        Returns:
            The loaded ``MLIPPredictUnit`` for this model_id.
        """
        parts = model_id.split(":", 1)
        checkpoint = parts[0]
        settings_name = parts[1] if len(parts) > 1 and parts[1] else "default"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.isfile(checkpoint):
            from fairchem.core.units.mlip_unit import load_predict_unit

            predict_unit = load_predict_unit(
                checkpoint,
                inference_settings=settings_name,
                device=device,
            )
        else:
            from fairchem.core.calculate import pretrained_mlip

            predict_unit = pretrained_mlip.get_predict_unit(
                checkpoint,
                inference_settings=settings_name,
                device=device,
            )

        logging.info(f"MultiplexedBatchPredictServer loaded model_id={model_id!r}")
        return predict_unit

    async def get_predict_unit_attribute(
        self, attribute_name: str, model_id: str | None = None
    ) -> Any:
        """
        Get an attribute from a loaded predict unit.

        Uses the ``multiplexed_model_id`` set on the request by the caller
        to resolve the correct model first.
        """
        model_id = model_id or serve.get_multiplexed_model_id()
        predict_unit = await self.get_model(model_id)
        attr = getattr(predict_unit, attribute_name)
        # Move any CUDA tensors to CPU before returning so callers (which
        # may be CPU-only Ray workers) can deserialize the result without
        # requiring CUDA.
        return move_tensors_to_cpu(attr)

    async def validate_atoms_data(self, atoms_info: dict, task_name: str) -> dict:
        """
        Run model-specific validation after loading the correct model.
        """
        from ase import Atoms

        model_id = serve.get_multiplexed_model_id()
        predict_unit = await self.get_model(model_id)
        stub = Atoms()
        stub.info = atoms_info
        predict_unit.validate_atoms_data(stub, task_name)
        return stub.info

    async def __call__(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict:
        """
        Main entry point for multiplexed inference requests.

        ``serve.get_multiplexed_model_id()`` is called here (per-request
        context) and forwarded explicitly to ``predict()``.  Inside the
        ``@serve.batch`` function only one request context is active, so
        the model ID cannot be reliably read there.

        Args:
            data: Single AtomicData object.
            undo_element_references: Whether to undo element references.

        Returns:
            Prediction dictionary for this system.
        """
        model_id = serve.get_multiplexed_model_id()
        predictions = await self.predict(data, model_id, undo_element_references)
        return predictions


def _init_ray_and_serve(
    ray_actor_options: dict,
    num_replicas: int,
) -> None:
    """
    Ensure Ray and Ray Serve are initialised.
    """
    cpus_per_actor = ray_actor_options.get("num_cpus", min(cpu_count(), 8))
    ray_actor_options["num_cpus"] = cpus_per_actor

    requested_cpus = cpus_per_actor * num_replicas

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=requested_cpus,
        )
        logging.info("Ray initialized")

    # If the deployment's CPU request plus that overhead consumes the whole
    # cluster, downstream Ray tasks (and even Serve's own actors) can
    # starve. Warn (don't raise) because multi-node Ray clusters can
    # auto-grow as workers join, and autoscaling deployments only need
    # capacity for ``min_replicas`` at startup.
    serve_overhead_cpus = 2  # Serve controller + HTTP proxy
    cluster_cpus = ray.cluster_resources().get("CPU", 0)
    if requested_cpus + serve_overhead_cpus > cluster_cpus:
        logging.warning(
            f"Ray Serve deployment requests {cpus_per_actor} CPU(s) x "
            f"{num_replicas} replica(s) = {requested_cpus} CPU(s), plus "
            f"~{serve_overhead_cpus} CPU(s) for the Serve controller/proxy, "
            f"but the Ray cluster currently reports {cluster_cpus:g} CPU(s). "
            "Replicas will queue until workers join or autoscaling adds "
            "capacity. If the cluster is fixed-size and small, reduce "
            "ray_actor_options['num_cpus'] / num_replicas."
        )

    try:
        serve.status()
        logging.info("Ray Serve already running")
    except Exception:
        serve.start(
            logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
        )
        logging.info("Ray Serve started")


def _effective_replicas(deployment_config: dict) -> int:
    """
    Best-effort estimate of the number of replicas that need to be
    schedulable at startup, for CPU-sizing warnings.

    For autoscaling configs, returns ``min_replicas`` (autoscaling will
    grow to ``max_replicas`` later if capacity becomes available).
    Otherwise returns ``num_replicas`` if it is an int, else ``1``.
    """
    ac = deployment_config.get("autoscaling_config") or {}
    if ac:
        for key in ("min_replicas", "max_replicas"):
            if key in ac:
                try:
                    return max(1, int(ac[key]))
                except (TypeError, ValueError):
                    pass
    nr = deployment_config.get("num_replicas")
    if isinstance(nr, int):
        return max(1, nr)
    return 1


def _prepare_deployment_config(
    deployment_config: DeploymentConfig | dict | None,
    default_num_gpus_when_cuda: bool,
) -> DeploymentConfig:
    """
    Normalize ``deployment_config`` to a :class:`DeploymentConfig` and ensure
    ``ray_actor_options`` is a dict, defaulting ``num_gpus=1`` when CUDA is
    wanted and the caller did not pin a value.
    """
    if not isinstance(deployment_config, DeploymentConfig):
        deployment_config = DeploymentConfig(**(deployment_config or {}))
    actor_opts = dict(deployment_config.ray_actor_options or {})
    if default_num_gpus_when_cuda and "num_gpus" not in actor_opts:
        actor_opts["num_gpus"] = 1
    deployment_config.ray_actor_options = actor_opts
    return deployment_config


def setup_batch_predict_server(
    predict_unit: MLIPPredictUnit,
    deployment_config: DeploymentConfig | dict | None = None,
    batch_config: BatchConfig | dict | None = None,
    deployment_name: str = "predict-server",
    route_prefix: str = "/predict",
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``BatchPredictServer`` that serves a single pre-loaded model.

    Args:
        predict_unit: An MLIPPredictUnit instance to use for inference.
        deployment_config: :class:`DeploymentConfig` (or equivalent dict) of
            kwargs forwarded to ``BatchPredictServer.options(...)``. Any field
            on :class:`DeploymentConfig` is valid (e.g. ``num_replicas``,
            ``autoscaling_config``, ``ray_actor_options``,
            ``max_ongoing_requests``, ``graceful_shutdown_timeout_s``,
            ``logging_config``).
        batch_config: :class:`BatchConfig` (or equivalent dict) of kwargs
            forwarded into ``BatchPredictServer.__init__`` via
            ``.bind(**batch_config)``. Accepts ``max_batch_size``,
            ``batch_wait_timeout_s``, ``split_oom_batch``,
            ``max_concurrent_batches``.
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.

    Returns:
        Ray Serve deployment handle.
    """
    dc = _prepare_deployment_config(
        deployment_config,
        default_num_gpus_when_cuda="cuda" in predict_unit.device,
    )
    if not isinstance(batch_config, BatchConfig):
        batch_config = BatchConfig(**(batch_config or {}))

    dc_kwargs = dc.to_options_kwargs()
    _init_ray_and_serve(dc_kwargs["ray_actor_options"], _effective_replicas(dc_kwargs))

    predict_unit_ref = ray.put(predict_unit)
    logging.info("Predict unit stored in Ray object store")

    deployment = BatchPredictServer.options(**dc_kwargs).bind(
        predict_unit_ref, **batch_config.to_init_kwargs()
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)
    logging.info(f"BatchPredictServer deployed: name={deployment_name}")
    return handle


def setup_multiplexed_batch_predict_server(
    deployment_config: DeploymentConfig | dict | None = None,
    batch_config: BatchConfig | dict | None = None,
    deployment_name: str = "multiplexed-predict-server",
    route_prefix: str = "/multiplex-predict",
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``MultiplexedBatchPredictServer`` that loads models on demand.

    Models are loaded lazily when a request arrives with a
    ``multiplexed_model_id`` set on the handle.

    Args:
        deployment_config: :class:`DeploymentConfig` (or equivalent dict)
            forwarded to ``MultiplexedBatchPredictServer.options(...)``.
        batch_config: :class:`BatchConfig` (or equivalent dict) forwarded
            into ``MultiplexedBatchPredictServer.__init__`` via
            ``.bind(**batch_config)``.
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.

    Returns:
        Ray Serve deployment handle.
    """
    dc = _prepare_deployment_config(
        deployment_config,
        default_num_gpus_when_cuda=torch.cuda.is_available(),
    )
    if not isinstance(batch_config, BatchConfig):
        batch_config = BatchConfig(**(batch_config or {}))

    dc_kwargs = dc.to_options_kwargs()
    _init_ray_and_serve(dc_kwargs["ray_actor_options"], _effective_replicas(dc_kwargs))

    deployment = MultiplexedBatchPredictServer.options(**dc_kwargs).bind(
        **batch_config.to_init_kwargs()
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)
    logging.info(f"MultiplexedBatchPredictServer deployed: name={deployment_name}")
    return handle


def wait_for_serve_ready(
    app_name: str,
    poll_interval_seconds: float = 2,
    timeout_seconds: float = 600,
) -> bool:
    """
    Wait for Ray Serve to be fully ready to accept requests.

    Blocks until the Ray Serve controller is running and the specified
    application reaches RUNNING status.

    Args:
        app_name: Name of the Ray Serve application to wait for.
        poll_interval_seconds: How often to check status.
        timeout_seconds: Maximum total time to wait before raising
            ``TimeoutError``. Prevents indefinite hangs when a deployment
            cannot be scheduled (e.g. no free GPU).

    Returns:
        True if server is ready.

    Raises:
        RuntimeError: If server fails to deploy.
        TimeoutError: If the application does not reach RUNNING within
            ``timeout_seconds``.
    """
    deadline = time.monotonic() + timeout_seconds

    def _check_deadline(phase: str) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for Ray Serve "
                f"({phase}) for app {app_name!r}."
            )

    # Phase 1: Wait for Ray Serve controller
    logging.info("Waiting for Ray Serve controller to start...")
    while True:
        try:
            status = serve.status()
            logging.info("Ray Serve controller is running")
            break
        except Exception as e:
            error_msg = str(e)
            if (
                "SERVE_CONTROLLER_ACTOR" in error_msg
                or "Failed to look up actor" in error_msg
            ):
                logging.debug(f"Ray Serve controller not ready yet: {error_msg}")
                _check_deadline("controller startup")
                time.sleep(poll_interval_seconds)
            else:
                raise

    # Phase 2: Wait for the application to be deployed and running
    logging.info(f"Waiting for application '{app_name}' to be ready...")
    while True:
        _check_deadline("application RUNNING")
        try:
            status = serve.status()

            if app_name not in status.applications:
                logging.debug(f"Application '{app_name}' not found yet, waiting...")
                time.sleep(poll_interval_seconds)
                continue

            app_status = status.applications[app_name]

            if app_status.status == ApplicationStatus.RUNNING:
                logging.info(f"Application '{app_name}' is RUNNING and ready")
                return True
            elif app_status.status == ApplicationStatus.DEPLOYING:
                logging.debug(f"Application '{app_name}' is still deploying...")
                time.sleep(poll_interval_seconds)
            elif app_status.status in (
                ApplicationStatus.DEPLOY_FAILED,
                ApplicationStatus.UNHEALTHY,
            ):
                raise RuntimeError(
                    f"Application '{app_name}' failed to deploy. "
                    f"Status: {app_status.status}, Message: {app_status.message}"
                )
            else:
                logging.debug(f"Application '{app_name}' status: {app_status.status}")
                time.sleep(poll_interval_seconds)

        except RuntimeError:
            raise
        except Exception as e:
            logging.warning(f"Error checking serve status: {e}")
            time.sleep(poll_interval_seconds)


def get_ray_connection_info(head_file: str) -> dict[str, str | None]:
    """
    Read Ray connection info from a head.json file.

    Args:
        head_file: Path to head.json file from a Ray cluster.

    Returns:
        Dictionary with ``ray_address``, ``namespace_serve_fairchem``, and
        ``local`` keys. For local clusters ``ray_address`` is *None*.
    """
    with open(head_file) as f:
        head_info = json.load(f)

    namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")
    is_local = head_info.get("local", False)

    if is_local:
        return {
            "ray_address": None,
            "namespace_serve_fairchem": namespace_serve_fairchem,
            "local": True,
        }

    hostname = head_info.get("hostname")
    client_port = head_info.get("client_port")

    if not hostname or not client_port:
        raise ValueError(
            f"Invalid head.json: missing hostname or client_port in {head_file}"
        )

    return {
        "ray_address": f"ray://{hostname}:{client_port}",
        "namespace_serve_fairchem": namespace_serve_fairchem,
        "local": False,
    }
