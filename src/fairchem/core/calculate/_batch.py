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
from typing import Literal, Protocol

from fairchem.core.units.mlip_unit._batch_serve import setup_batch_predict_server
from fairchem.core.units.mlip_unit.predict import (
    BatchServerPredictUnit,
    MLIPPredictUnit,
)


class ExecutorProtocol(Protocol):
    def submit(self, fn, *args, **kwargs): ...
    def map(self, fn, *iterables, **kwargs): ...
    def shutdown(self, wait: bool = True): ...


def _get_concurrency_backend(
    backend: Literal["threads"], options: dict
) -> ExecutorProtocol:
    """Get a backend to run ASE calculations concurrently."""
    if backend == "threads":
        return ThreadPoolExecutor(**options)
    raise ValueError(f"Invalid concurrency backend: {backend}")


class InferenceBatcher:
    """Batches incoming inference requests."""

    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        max_batch_size: int = 512,
        batch_wait_timeout_s: float = 0.1,
        num_replicas: int = 1,
        concurrency_backend: Literal["threads"] = "threads",
        concurrency_backend_options: dict | None = None,
        ray_actor_options: dict | None = None,
        deployment_name: str | None = None,
    ):
        """
        Args:
            predict_unit: The predict unit to use for inference.
            max_batch_size: Maximum number of atoms in a batch.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: The maximum time to wait for a batch to be ready.
            num_replicas: The number of replicas to use for inference.
            concurrency_backend: The concurrency backend to use for inference.
            concurrency_backend_options: Options to pass to the concurrency backend.
            ray_actor_options: Options to pass to the Ray actor running the batch server.
            deployment_name: Name for the Ray Serve deployment. If None, generates a unique name.
                This allows multiple InferenceBatchers to coexist on the same Ray cluster.
        """
        self.predict_unit = predict_unit
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.num_replicas = num_replicas

        # Generate unique deployment name if not provided
        if deployment_name is None:
            deployment_name = f"predict-server-{uuid.uuid4().hex[:8]}"
        self.deployment_name = deployment_name

        self.predict_server_handle = setup_batch_predict_server(
            predict_unit=self.predict_unit,
            max_batch_size=self.max_batch_size,
            batch_wait_timeout_s=self.batch_wait_timeout_s,
            num_replicas=self.num_replicas,
            ray_actor_options=ray_actor_options or {},
            deployment_name=self.deployment_name,
            route_prefix=f"/{self.deployment_name}",
        )

        if concurrency_backend_options is None:
            concurrency_backend_options = {}

        if (
            concurrency_backend == "threads"
            and "max_workers" not in concurrency_backend_options
        ):
            concurrency_backend_options["max_workers"] = min(cpu_count(), 16)

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
            predict_unit=self.predict_unit,
        )

    def update_checkpoint(self, new_predict_unit: MLIPPredictUnit) -> None:
        """Update the checkpoint being served without shutting down the deployment.

        Args:
            new_predict_unit: A new MLIPPredictUnit instance with the updated checkpoint
        """
        import ray
        # Store the new predict unit in Ray's object store
        predict_unit_ref = ray.put(new_predict_unit)
        # Update all replicas with the new predict unit
        self.predict_server_handle.update_predict_unit.remote(predict_unit_ref)

    def delete(self) -> None:
        """Delete the Ray Serve deployment without shutting down Ray or the executor.

        This allows the InferenceBatcher to be removed while keeping Ray running
        for other batchers or applications.
        """
        if hasattr(self, "predict_server_handle") and self.predict_server_handle is not None:
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
