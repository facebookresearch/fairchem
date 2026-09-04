"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Literal, Protocol

import ray
from ray import serve

from fairchem.core.components.batch_server import (
    AutobatchConfig,
    AutobatchResult,
    BatchConfig,
    probe_optimal_batch_size,
    setup_batch_predict_server,
    update_batch_config,
    update_served_predict_unit,
)
from fairchem.core.units.mlip_unit.predict import (
    BatchServerPredictUnit,
    MLIPPredictUnit,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


DEFAULT_EXECUTOR_WORKER_CAP = 16


class ExecutorProtocol(Protocol):
    def submit(self, fn, *args, **kwargs): ...
    def map(self, fn, *iterables, **kwargs): ...
    def shutdown(self, wait: bool = True): ...


def _get_concurrency_backend(
    backend: Literal["threads"], options: dict
) -> ExecutorProtocol:
    """Get a backend to run ASE calculations concurrently.

    Args:
        backend: The concurrency backend type. Only ``"threads"`` is supported:
            simulations submitted here hold a Ray ``DeploymentHandle``, which
            is not usable across a plain process boundary.
        options: Backend-specific options dictionary (e.g. ``max_workers``).

    Returns:
        An executor implementing ExecutorProtocol.

    Raises:
        ValueError: If an invalid backend is specified.
    """
    if backend == "threads":
        return ThreadPoolExecutor(**options)
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
        ...     # Run concurrent simulations using batcher.executor
        ...     futures = [batcher.executor.submit(run_sim, atoms) for atoms in systems]

    Example with autobatching:
        >>> predict_unit = MLIPPredictUnit(model_path, device="cuda")
        >>> data = [AtomicData.from_ase(bulk("Cu"), task_name="omat")]
        >>> with InferenceBatcher(predict_unit) as batcher:
        ...     # Probe for optimal batch size using representative data
        ...     batcher.auto_configure_batching(data)
        ...     # Now run simulations with optimal batch size
        ...     futures = [batcher.executor.submit(run_sim, atoms) for atoms in systems]
    """

    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        max_batch_size: int = 512,
        batch_wait_timeout_s: float = 0.1,
        split_oom_batch: bool = False,
        num_replicas: int = 1,
        concurrency_backend: Literal["threads"] = "threads",
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
            concurrency_backend: The concurrency backend to use for running
                simulations. Only "threads" (ThreadPoolExecutor) is supported;
                simulations submitted to the executor hold a Ray
                DeploymentHandle, which cannot cross a process boundary.
                Requests block on the server, so threads are the right fit.
            concurrency_backend_options: Options to pass to the concurrency
                backend, e.g. max_workers (int).
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
        self.split_oom_batch = split_oom_batch
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
                "split_oom_batch": self.split_oom_batch,
            },
            deployment_name=self.deployment_name,
            route_prefix=f"/{self.deployment_name}",
        )

        # Copy rather than mutate: the caller's dict should not gain a
        # max_workers key as a side effect of constructing the batcher.
        concurrency_backend_options = dict(concurrency_backend_options or {})

        if "max_workers" not in concurrency_backend_options:
            concurrency_backend_options["max_workers"] = min(
                cpu_count(), DEFAULT_EXECUTOR_WORKER_CAP
            )

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

        Args:
            data: List of AtomicData objects to use for probing.
            config: Autobatch configuration. Uses defaults if None.

        Returns:
            AutobatchResult with the determined optimal parameters.
        """
        result = probe_optimal_batch_size(
            predict_unit=self.predict_unit,
            probe_data=data,
            config=config,
        )
        # Broadcast through user_config rather than a handle call: a handle
        # call reaches one replica and would leave the rest -- and any replica
        # autoscaling adds later -- on the old batch size. This blocks until
        # the update is applied, so a failure raises instead of being dropped.
        update_batch_config(
            self.deployment_name,
            BatchConfig(
                max_batch_size=result.max_batch_size,
                batch_wait_timeout_s=result.batch_wait_timeout_s,
                split_oom_batch=self.split_oom_batch,
            ),
        )
        # Keep the batcher's advertised settings in step with the server's.
        self.max_batch_size = result.max_batch_size
        self.batch_wait_timeout_s = result.batch_wait_timeout_s
        return result

    def update_checkpoint(self, new_predict_unit: MLIPPredictUnit) -> None:
        """Update the checkpoint being served without shutting down the deployment.

        The new checkpoint is rolled out to every replica. Replicas restart to
        pick it up, so in-flight requests are drained by Ray Serve's rolling
        update rather than being served a mix of old and new weights.

        Args:
            new_predict_unit: A new MLIPPredictUnit instance with the updated checkpoint
        """
        update_served_predict_unit(self.deployment_name, new_predict_unit)
        self.predict_unit = new_predict_unit

    def delete(self) -> None:
        """Delete the Ray Serve deployment without shutting down Ray or the executor.

        This allows the InferenceBatcher to be removed while keeping Ray running
        for other batchers or applications.
        """
        if (
            hasattr(self, "predict_server_handle")
            and self.predict_server_handle is not None
        ):
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
