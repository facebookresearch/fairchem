"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Literal, Protocol

import ray
from ray.util import ActorPool

from fairchem.core.components.batch_server import (
    AutobatchConfig,
    AutobatchResult,
    probe_optimal_batch_size,
    setup_batch_predict_server,
)
from fairchem.core.units.mlip_unit.predict import (
    BatchServerPredictUnit,
    MLIPPredictUnit,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


class ExecutorProtocol(Protocol):
    def submit(self, fn, *args, **kwargs): ...
    def map(self, fn, *iterables, **kwargs): ...
    def shutdown(self, wait: bool = True): ...


class RayActorPoolExecutor:
    """
    Executor-like wrapper around Ray's ActorPool for concurrent task execution.

    This provides an interface compatible with ExecutorProtocol while using
    Ray actors for distributed execution.

    Note: Tasks submitted to this executor must be picklable and should not
    rely on shared state between workers.
    """

    def __init__(
        self,
        num_workers: int = 4,
        ray_actor_options: dict | None = None,
    ):
        """
        Args:
            num_workers: Number of Ray actor workers in the pool.
            ray_actor_options: Options to pass to Ray actor creation
                (e.g., {"num_cpus": 1, "num_gpus": 0}).
        """

        if ray_actor_options is None:
            ray_actor_options = {}

        if not ray.is_initialized():
            ray.init(
                log_to_driver=False,
                logging_config=ray.LoggingConfig(log_level="WARNING"),
            )

        @ray.remote
        class _TaskWorker:
            """
            Simple worker that executes arbitrary callables.
            """

            def execute(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        self._actors = [
            _TaskWorker.options(**ray_actor_options).remote()
            for _ in range(num_workers)
        ]
        self._pool = ActorPool(self._actors)
        self._current_actor_idx = 0

    def submit(self, fn, *args, **kwargs):
        """
        Submit a callable to be executed by a worker using round-robin selection.

        Args:
            fn: Callable to execute (must be picklable).
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            A Ray ObjectRef. Use ray.get() to retrieve the result.
        """
        actor = self._actors[self._current_actor_idx]
        self._current_actor_idx = (self._current_actor_idx + 1) % len(self._actors)
        return actor.execute.remote(fn, *args, **kwargs)

    def map(self, fn, *iterables, **kwargs):
        """
        Map a callable over iterables using the actor pool.

        Args:
            fn: Callable to execute (must be picklable).
            *iterables: Iterables to map over.
            **kwargs: Additional keyword arguments (timeout is ignored).

        Returns:
            Iterator of results.
        """
        args_list = list(zip(*iterables))

        def _task(actor, args):
            return actor.execute.remote(fn, *args)

        return self._pool.map(_task, args_list)

    def shutdown(self, wait: bool = True):
        """
        Shutdown the actor pool.

        Args:
            wait: If True, wait for pending tasks (not fully supported with Ray actors).
        """
        import ray

        for actor in self._actors:
            ray.kill(actor)
        self._actors = []


def _get_concurrency_backend(
    backend: Literal["threads", "processes", "ray-actors"], options: dict
) -> ExecutorProtocol:
    """
    Get a backend to run ASE calculations concurrently.

    Args:
        backend: The concurrency backend type:
            - "threads": ThreadPoolExecutor for I/O-bound tasks (default).
            - "processes": ProcessPoolExecutor for CPU-bound tasks.
                Note: Tasks must be picklable; not suitable for GPU operations.
            - "ray-actors": Ray actor pool for distributed execution.
                Supports options like num_workers and ray_actor_options.
        options: Backend-specific options dictionary.

    Returns:
        An executor implementing ExecutorProtocol.

    Raises:
        ValueError: If an invalid backend is specified.
    """
    if backend == "threads":
        return ThreadPoolExecutor(**options)
    elif backend == "processes":
        return ProcessPoolExecutor(**options)
    elif backend == "ray-actors":
        return RayActorPoolExecutor(**options)
    raise ValueError(f"Invalid concurrency backend: {backend}")


class InferenceBatcher:
    """
    Batches incoming inference requests.

    This class provides a high-level API for running concurrent simulations
    with batched inference calls to an AI model. It supports multiple
    concurrency backends for different use cases.

    Example:
        >>> predict_unit = MLIPPredictUnit(model_path, device="cuda")
        >>> with InferenceBatcher(predict_unit, max_batch_size=1024) as batcher:
        ...     futures = [batcher.executor.submit(run_sim, atoms) for atoms in systems]

    Example with autobatching:
        >>> predict_unit = MLIPPredictUnit(model_path, device="cuda")
        >>> data = [AtomicData.from_ase(bulk("Cu"), task_name="omat")]
        >>> with InferenceBatcher(predict_unit) as batcher:
        ...     batcher.auto_configure_batching(data)
        ...     futures = [batcher.executor.submit(run_sim, atoms) for atoms in systems]
    """

    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        max_batch_size: int = 512,
        batch_wait_timeout_s: float = 0.1,
        split_oom_batch: bool = False,
        num_replicas: int = 1,
        concurrency_backend: Literal["threads", "processes", "ray-actors"] = "threads",
        concurrency_backend_options: dict | None = None,
        ray_actor_options: dict | None = None,
        deployment_name: str | None = None,
        autoscaling_config: dict | None = None,
    ):
        """
        Args:
            predict_unit: The predict unit to use for inference.
            max_batch_size: Maximum number of atoms in a batch.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: The maximum time to wait for a batch to be ready.
            split_oom_batch: If True, split and retry on OOM errors.
            num_replicas: The number of replicas to use for inference. Ignored if
                autoscaling_config is provided.
            concurrency_backend: The concurrency backend to use for running simulations:
                - "threads": ThreadPoolExecutor (default). Best for I/O-bound tasks.
                    Options: max_workers (int).
                - "processes": ProcessPoolExecutor. Best for CPU-bound tasks.
                    Note: Tasks must be picklable; not suitable for GPU operations.
                    Options: max_workers (int).
                - "ray-actors": Ray actor pool for distributed execution.
                    Options: num_workers (int), ray_actor_options (dict).
            concurrency_backend_options: Options to pass to the concurrency backend.
                See backend descriptions above for available options.
            ray_actor_options: Options to pass to the Ray actor running the batch server.
            deployment_name: Name for the Ray Serve deployment. If None, generates a
                unique name. This allows multiple InferenceBatchers to coexist on the
                same Ray cluster.
            autoscaling_config: Optional autoscaling configuration. If provided, enables
                autoscaling and num_replicas is ignored. Example:
                {
                    "min_replicas": 0,  # Scale to zero when idle
                    "max_replicas": 4,
                    "target_ongoing_requests": 2,
                    "downscale_delay_s": 60,  # Wait 60s before scaling down
                    "upscale_delay_s": 5,  # Scale up quickly
                }
        """
        self.predict_unit = predict_unit
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.num_replicas = num_replicas
        self.autoscaling_config = autoscaling_config

        # Generate unique deployment name if not provided
        if deployment_name is None:
            deployment_name = f"predict-server-{uuid.uuid4().hex[:8]}"
        self.deployment_name = deployment_name

        self.predict_server_handle = setup_batch_predict_server(
            predict_unit=self.predict_unit,
            deployment_config={
                "ray_actor_options": ray_actor_options or {},
                **(
                    {"autoscaling_config": self.autoscaling_config}
                    if self.autoscaling_config is not None
                    else {"num_replicas": self.num_replicas}
                ),
            },
            batch_config={
                "max_batch_size": self.max_batch_size,
                "batch_wait_timeout_s": self.batch_wait_timeout_s,
                "split_oom_batch": split_oom_batch,
            },
            deployment_name=self.deployment_name,
            route_prefix=f"/{self.deployment_name}",
        )

        if concurrency_backend_options is None:
            concurrency_backend_options = {}

        # Set default max_workers for thread and process backends
        if concurrency_backend in ("threads", "processes"):
            if "max_workers" not in concurrency_backend_options:
                concurrency_backend_options["max_workers"] = min(cpu_count(), 16)
        # Set default num_workers for ray-actors backend
        elif (
            concurrency_backend == "ray-actors"
            and "num_workers" not in concurrency_backend_options
        ):
            concurrency_backend_options["num_workers"] = min(cpu_count(), 16)

        self.executor: ExecutorProtocol = _get_concurrency_backend(
            concurrency_backend, concurrency_backend_options
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @cached_property
    def batch_predict_unit(self) -> BatchServerPredictUnit:
        return BatchServerPredictUnit(
            server_handle=self.predict_server_handle,
        )

    def auto_configure_batching(
        self,
        data: list[AtomicData],
        config: AutobatchConfig | None = None,
    ) -> AutobatchResult:
        """
        Probe for optimal batch size and timeout using representative data.

        This method runs inference with increasing batch sizes until OOM,
        then configures the server with optimal parameters.

        Args:
            data: List of AtomicData objects to use for probing. The data
                will be repeated if needed to reach larger batch sizes during probing.
            config: Autobatch configuration. Uses defaults if None.

        Returns:
            AutobatchResult with the determined optimal parameters.
        """
        if config is None:
            config = AutobatchConfig()

        result = probe_optimal_batch_size(
            predict_unit=self.predict_unit,
            probe_data=data,
            config=config,
        )

        # Update the server with the new batching parameters
        self.predict_server_handle.configure_batching.remote(
            result.max_batch_size, result.batch_wait_timeout_s
        )

        return result

    def update_checkpoint(self, new_predict_unit: MLIPPredictUnit) -> None:
        """Update the checkpoint being served without shutting down the deployment.

        Args:
            new_predict_unit: A new MLIPPredictUnit instance with the updated checkpoint
        """
        import ray

        # Put the model in the object store so only a lightweight reference
        # travels through the Serve routing layer; Ray resolves it on the server.
        predict_unit_ref = ray.put(new_predict_unit)
        # Update all replicas with the new predict unit and wait for completion
        self.predict_server_handle.update_predict_unit.remote(predict_unit_ref).result()

    def delete(self) -> None:
        """Delete the Ray Serve deployment without shutting down Ray or the executor.

        This allows the InferenceBatcher to be removed while keeping Ray running
        for other batchers or applications.
        """
        if (
            hasattr(self, "predict_server_handle")
            and self.predict_server_handle is not None
        ):
            import ray
            from ray import serve

            # Check if Ray is still initialized before trying to delete
            if ray.is_initialized():
                with contextlib.suppress(Exception):
                    serve.delete(self.deployment_name)

            self.predict_server_handle = None

    def shutdown(self, wait: bool = True, shutdown_ray: bool = False) -> None:
        """Shutdown the executor, Ray Serve deployment, and optionally Ray itself.

        Args:
            wait: If True, wait for pending tasks to complete before returning.
            shutdown_ray: If True, shutdown Ray Serve and Ray completely. If False,
                only delete this deployment and shutdown the executor.
                DEFAULT: False for safety with concurrent Ray usage.
        """
        # Shutdown the executor
        if hasattr(self, "executor"):
            with contextlib.suppress(Exception):
                self.executor.shutdown(wait=wait)

        # Delete the deployment (safe for concurrent usage)
        self.delete()

        # Optionally shutdown Ray Serve and Ray completely
        # This should only be used when you're SURE no other batchers are running
        if shutdown_ray:
            import ray
            from ray import serve

            with contextlib.suppress(Exception):
                serve.shutdown()

            with contextlib.suppress(Exception):
                if ray.is_initialized():
                    ray.shutdown()

    def __del__(self):
        """Cleanup on deletion."""
        # Only delete deployment, don't shutdown Ray in __del__
        with contextlib.suppress(Exception):
            self.delete()
        with contextlib.suppress(Exception):
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
