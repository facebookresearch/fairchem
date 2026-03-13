"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import ray
import torch
from ray import serve

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


@dataclass
class AutobatchConfig:
    """Configuration for probing-based autobatching.

    Attributes:
        min_batch_size: Minimum batch size (in atoms) to start probing from.
        max_batch_size_cap: Maximum batch size cap to avoid excessive probing.
        probe_steps: Number of probe steps to run at each batch size.
        backoff_factor: Factor to reduce batch size by after OOM (e.g., 0.8 = 80%).
        timeout_floor_s: Minimum batch wait timeout in seconds.
        timeout_ceil_s: Maximum batch wait timeout in seconds.
        timeout_latency_multiplier: Multiplier applied to median latency to compute timeout.
        warmup_steps: Number of warmup inference steps before probing.
    """

    min_batch_size: int = 128
    max_batch_size_cap: int = 16384
    probe_steps: int = 3
    backoff_factor: float = 0.8
    timeout_floor_s: float = 0.01
    timeout_ceil_s: float = 1.0
    timeout_latency_multiplier: float = 2.0
    warmup_steps: int = 2


@dataclass
class AutobatchResult:
    """Result from autobatch probing.

    Attributes:
        max_batch_size: Optimal maximum batch size in atoms.
        batch_wait_timeout_s: Optimal batch wait timeout in seconds.
        median_latency_s: Median inference latency observed during probing.
        probe_timestamp: Unix timestamp when probing was performed.
    """

    max_batch_size: int
    batch_wait_timeout_s: float
    median_latency_s: float
    probe_timestamp: float = field(default_factory=time.time)


def _expand_probe_data(
    data_list: list[AtomicData], target_num_atoms: int
) -> list[AtomicData]:
    """Expand probe data by repeating items to reach target atom count.

    Args:
        data_list: List of AtomicData objects to use as base data.
        target_num_atoms: Target total number of atoms for the batch.

    Returns:
        List of AtomicData objects with total atoms >= target_num_atoms.
    """
    if not data_list:
        raise ValueError("data_list cannot be empty")

    # Calculate total atoms in the provided data
    base_num_atoms = sum(data.natoms.sum().item() for data in data_list)

    if base_num_atoms >= target_num_atoms:
        return data_list

    num_repeats = (target_num_atoms + base_num_atoms - 1) // base_num_atoms

    expanded_list = []
    for _ in range(num_repeats):
        expanded_list.extend(data_list)

    return expanded_list


def probe_optimal_batch_size(
    predict_unit: MLIPPredictUnit,
    probe_data: list[AtomicData],
    config: AutobatchConfig | None = None,
) -> AutobatchResult:
    """Probe for optimal batch size and timeout using runtime GPU memory behavior.

    This function performs a binary search-like probing to find the maximum
    batch size that doesn't cause OOM errors, then derives an appropriate
    batch wait timeout from observed latencies.

    Args:
        predict_unit: The MLIPPredictUnit to probe with.
        probe_data: List of AtomicData objects to use for probing. If the total
            number of atoms is less than the target batch size being probed,
            the data will be repeated to reach the target size.
        config: Autobatch configuration. Uses defaults if None.

    Returns:
        AutobatchResult with optimal parameters.
    """
    if config is None:
        config = AutobatchConfig()

    if not probe_data:
        raise ValueError("probe_data cannot be empty")

    device = predict_unit.device

    # For CPU, use conservative defaults
    if "cuda" not in str(device):
        logging.info("Autobatch probing skipped for CPU device, using defaults")
        return AutobatchResult(
            max_batch_size=config.min_batch_size,
            batch_wait_timeout_s=config.timeout_ceil_s,
            median_latency_s=0.1,
        )

    logging.info("Starting autobatch probing...")

    # Get initial GPU memory state
    free_mem, total_mem = (
        (0, 0) if not torch.cuda.is_available() else torch.cuda.mem_get_info()
    )
    logging.info(
        f"GPU memory: {free_mem / 1e9:.2f}GB free / {total_mem / 1e9:.2f}GB total"
    )

    # Warmup the model using the provided probe data
    logging.info(f"Running {config.warmup_steps} warmup steps...")
    warmup_batch = atomicdata_list_to_batch(probe_data)
    for _ in range(config.warmup_steps):
        try:
            predict_unit.predict(warmup_batch, undo_element_references=False)
        except Exception as e:
            logging.warning(f"Warmup step failed: {e}")
    torch.cuda.empty_cache()

    # Binary search for optimal batch size
    low = config.min_batch_size
    high = config.max_batch_size_cap
    best_batch_size = low
    latencies: list[float] = []

    logging.info(f"Probing batch sizes in range [{low}, {high}]...")

    while low <= high:
        mid = (low + high) // 2
        success = True
        step_latencies = []

        logging.debug(f"Testing batch size: {mid} atoms")

        for step in range(config.probe_steps):
            try:
                # Expand probe data to reach target batch size by repeating items
                expanded_data = _expand_probe_data(probe_data, mid)
                batch = atomicdata_list_to_batch(expanded_data)

                # Time the inference
                torch.cuda.synchronize()
                start = time.perf_counter()
                predict_unit.predict(batch, undo_element_references=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                step_latencies.append(elapsed)
                logging.debug(f"  Step {step + 1}: {elapsed:.4f}s")

            except torch.OutOfMemoryError:
                logging.debug(f"  OOM at batch size {mid}")
                success = False
                torch.cuda.empty_cache()
                break
            except Exception as e:
                logging.warning(f"  Probe failed at batch size {mid}: {e}")
                success = False
                break

        if success:
            best_batch_size = mid
            latencies.extend(step_latencies)
            low = mid + 1
            logging.debug(f"  Success at {mid}, trying larger...")
        else:
            high = mid - 1
            logging.debug(f"  Failed at {mid}, trying smaller...")

    # Apply backoff factor for safety margin
    final_batch_size = int(best_batch_size * config.backoff_factor)
    final_batch_size = max(final_batch_size, config.min_batch_size)

    # Compute timeout from latencies
    if latencies:
        sorted_latencies = sorted(latencies)
        median_latency = sorted_latencies[len(sorted_latencies) // 2]
        timeout = median_latency * config.timeout_latency_multiplier
        timeout = max(config.timeout_floor_s, min(timeout, config.timeout_ceil_s))
    else:
        median_latency = 0.1
        timeout = config.timeout_ceil_s

    result = AutobatchResult(
        max_batch_size=final_batch_size,
        batch_wait_timeout_s=timeout,
        median_latency_s=median_latency,
    )

    logging.info(
        f"Autobatch probing complete: max_batch_size={result.max_batch_size}, "
        f"timeout={result.batch_wait_timeout_s:.4f}s, "
        f"median_latency={result.median_latency_s:.4f}s"
    )

    return result


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class BatchPredictServer:
    """
    Ray Serve deployment that batches incoming inference requests.
    """

    def __init__(
        self,
        predict_unit_ref,
        max_batch_size: int | None,
        batch_wait_timeout_s: float | None,
        split_oom_batch: bool = True,
    ):
        """
        Initialize with a Ray object reference to a PredictUnit.

        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
            max_batch_size: Maximum number of atoms in a batch. If None, batching
                must be configured via configure_batching() or auto_configure_batching()
                before running predictions.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction.
                If None, batching must be configured before running predictions.
            split_oom_batch: If true will split batch if an OOM error is raised
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.split_oom_batch = split_oom_batch
        self._batching_configured = False

        if max_batch_size is not None and batch_wait_timeout_s is not None:
            self.configure_batching(max_batch_size, batch_wait_timeout_s)
        elif max_batch_size is not None or batch_wait_timeout_s is not None:
            raise ValueError(
                "Both max_batch_size and batch_wait_timeout_s must be provided together, "
                "or both must be None for autobatch configuration."
            )
        else:
            logging.info(
                "BatchPredictServer initialized without batching configuration. "
                "Call configure_batching() or use InferenceBatcher.auto_configure_batching() "
                "before running predictions."
            )

        logging.info("BatchedPredictor initialized with predict_unit from object store")

    def configure_batching(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
    ):
        """Configure batching parameters.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Maximum wait time before processing partial batch.

        Raises:
            ValueError: If max_batch_size or batch_wait_timeout_s is invalid.
        """
        if max_batch_size is None or max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be a positive integer, got {max_batch_size}"
            )
        if batch_wait_timeout_s is None or batch_wait_timeout_s <= 0:
            raise ValueError(
                f"batch_wait_timeout_s must be a positive float, got {batch_wait_timeout_s}"
            )

        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)
        self._batching_configured = True
        logging.info(
            f"Batching configured: max_batch_size={max_batch_size}, "
            f"batch_wait_timeout_s={batch_wait_timeout_s}"
        )

    def get_predict_unit_attribute(self, attribute_name: str) -> Any:
        return getattr(self.predict_unit, attribute_name)

    @serve.batch(
        batch_size_fn=lambda batch: sum(sample.natoms.sum() for sample in batch).item()
    )
    async def predict(
        self, data_list: list[AtomicData], undo_element_references: bool = True
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects.

        Args:
            data_list: List of AtomicData objects (automatically batched by Ray Serve)
            undo_element_references: Whether to undo element references in predictions

        Returns:
            List of prediction dictionaries, one per input

        Raises:
            RuntimeError: If batching has not been configured.
        """
        if not self._batching_configured:
            raise RuntimeError(
                "Batching has not been configured. Call configure_batching() with "
                "explicit max_batch_size and batch_wait_timeout_s values, or use "
                "InferenceBatcher.auto_configure_batching() to automatically determine "
                "optimal parameters before running predictions."
            )
        data_deque = deque([data_list])
        prediction_list = []
        while len(data_deque) > 0:
            oom = False
            data_list = data_deque.popleft()
            batch = atomicdata_list_to_batch(data_list)

            try:
                predictions = self.predict_unit.predict(
                    batch, undo_element_references=undo_element_references
                )
                prediction_list.extend(self._split_predictions(predictions, batch))
            except torch.OutOfMemoryError as err:
                print(f"OutOfMemoryError: {err}!!!!!")
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Reduce max_batch_size or set split_oom_batch=True to automatically split OOM batches."
                    ) from err

                if len(data_list) == 1:
                    raise torch.OutOfMemoryError(
                        "Out of memory for a single system left in batch."
                    ) from err

                logging.warning(
                    "Caught out of memory error. Splitting batch and retrying."
                )
                oom = True
                torch.cuda.empty_cache()

            if oom:
                mid = len(data_list) // 2
                data_deque.appendleft(data_list[mid:])
                data_deque.appendleft(data_list[:mid])

        return prediction_list

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

            split_preds.append(system_predictions)

        return split_preds


def setup_batch_predict_server(
    predict_unit: MLIPPredictUnit,
    max_batch_size: int | None = None,
    batch_wait_timeout_s: float | None = None,
    split_oom_batch: bool = True,
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    deployment_name: str = "predict-server",
    route_prefix: str = "/predict",
) -> serve.handle.DeploymentHandle:
    """
    Set up and deploy a BatchPredictServer for batched inference.

    Args:
        predict_unit: An MLIPPredictUnit instance to use for batched inference
        max_batch_size: Maximum number of atoms in a batch. If None, batching must
            be configured later via configure_batching() before running predictions.
        batch_wait_timeout_s: Maximum wait time before processing partial batch.
            If None, batching must be configured later.
        split_oom_batch: Whether to split batches that cause OOM errors.
        num_replicas: Number of deployment replicas for scaling.
        ray_actor_options: Additional Ray actor options (e.g., {"num_gpus": 1, "num_cpus": 4})
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.

    Returns:
        Ray Serve deployment handle that can be used to initialize BatchServerPredictUnit
    """
    if ray_actor_options is None:
        ray_actor_options = {}

    cpus_per_actor = ray_actor_options.get("num_cpus", min(cpu_count(), 8))
    ray_actor_options["num_cpus"] = cpus_per_actor

    if "cuda" in predict_unit.device and "num_gpus" not in ray_actor_options:
        # assign 1 GPU per replica by default if using GPU device
        ray_actor_options["num_gpus"] = 1

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=cpus_per_actor * num_replicas,
        )
        logging.info("Ray initialized by setup_batch_predict_server")

    serve.start(
        logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    )
    logging.info("Ray Serve started by setup_batch_predict_server")

    predict_unit_ref = ray.put(predict_unit)
    logging.info("Predict unit stored in Ray object store")

    deployment = BatchPredictServer.options(
        num_replicas=num_replicas,
        ray_actor_options=ray_actor_options,
    ).bind(
        predict_unit_ref,
        max_batch_size=max_batch_size,
        batch_wait_timeout_s=batch_wait_timeout_s,
        split_oom_batch=split_oom_batch,
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)

    logging.info(
        f"BatchPredictServer deployed with max_batch_size={max_batch_size}, "
        f"batch_wait_timeout_s={batch_wait_timeout_s}, num_replicas={num_replicas}, "
        f"name={deployment_name}"
    )

    return handle
