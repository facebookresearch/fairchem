"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

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
    else:
        raise ValueError(f"Invalid concurrency backend: {backend}")


class InferenceBatcher:
    """
    Batches incoming inference requests.
    """

    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        max_batch_size: int = 16,
        batch_wait_timeout_s: float = 0.1,
        num_replicas: int = 1,
        concurrency_backend: Literal["threads"] = "threads",
        concurrency_backend_options: dict | None = None,
        ray_actor_options: dict | None = None,
    ):
        """
        Args:
            predict_unit: The predict unit to use for inference.
            max_batch_size: The maximum batch size to use for inference.
            batch_wait_timeout_s: The maximum time to wait for a batch to be ready.
            num_replicas: The number of replicas to use for inference.
            concurrency_backend: The concurrency backend to use for inference.
            concurrency_backend_options: Options to pass to the concurrency backend.
            ray_actor_options: Options to pass to the Ray actor running the batch server.
        """
        self.predict_unit = predict_unit
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.num_replicas = num_replicas

        self.predict_server_handle = setup_batch_predict_server(
            predict_unit=self.predict_unit,
            max_batch_size=self.max_batch_size,
            batch_wait_timeout_s=self.batch_wait_timeout_s,
            num_replicas=self.num_replicas,
            ray_actor_options=ray_actor_options or {},
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
        # serve.shutdown()
        self.executor.shutdown()

    @cached_property
    def batch_predict_unit(self):
        return BatchServerPredictUnit(
            server_handle=self.predict_server_handle,
        )

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        # serve.shutdown()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=wait)

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown(wait=False)
