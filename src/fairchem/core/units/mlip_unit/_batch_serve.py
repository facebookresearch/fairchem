"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import deque
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import ray
import torch
from ray import serve

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


@serve.deployment(
    serve.schema.LoggingConfig(log_level="WARNING"), max_ongoing_requests=100
)
class BatchPredictServer:
    """
    Ray Serve deployment that batches incoming inference requests.
    """

    def __init__(
        self,
        predict_unit_ref,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.05,
        split_oom_batch: bool = True,
    ):
        """
        Initialize with a Ray object reference to a PredictUnit.

        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
            max_batch_size: Maximum number of prediction requests to send to Ray.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction
            split_oom_batch: If true will split batch if an OOM error is raised
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.configure_batching(max_batch_size, batch_wait_timeout_s)
        self.split_oom_batch = split_oom_batch

        logging.info("BatchedPredictor initialized with predict_unit from object store")

    def configure_batching(
        self, max_batch_size: int = 32, batch_wait_timeout_s: float = 0.05
    ):
        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)

    @serve.batch
    async def predict(
        self, data_list: list[AtomicData], undo_element_references: bool = True
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects.

        Args:
            data_list: List of AtomicData objects (automatically batched by Ray Serve)

        Returns:
            List of prediction dictionaries, one per input
        """
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
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Reduce max_batch_size or set oom_split_batch=True to automatically split OOM batches."
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
                mid = len(data_deque) // 2
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
            batch_predictions: Dictionary of batched prediction tensors
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
    max_batch_size: int = 32,
    batch_wait_timeout_s: float = 0.1,
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    deployment_name: str = "predict-server",
    route_prefix: str = "/predict",
) -> serve.handle.DeploymentHandle:
    """
    Set up and deploy a BatchPredictServer for batched inference.

    Args:
        predict_unit: An MLIPPredictUnit instance to use for batched inference
        max_batch_size: Maximum number of systems per batch (default: 32)
        batch_wait_timeout_s: Maximum wait time before processing partial batch (default: 0.05)
        num_replicas: Number of deployment replicas for scaling (default: 1)
        ray_actor_options: Additional Ray actor options (e.g., {"num_gpus": 1, "num_cpus": 4})
        deployment_name: Name for the Ray Serve deployment (default: "predict-server")
        route_prefix: HTTP route prefix for the deployment (default: "/predict")

    Returns:
        Ray Serve deployment handle that can be used to initialize BatchServerPredictUnit
    """
    cpu_per_actor = (
        ray_actor_options.get("num_cpus", 1)
        if ray_actor_options
        else min(cpu_count(), 4)
    )

    if ray_actor_options is None:
        ray_actor_options = {}

    if "cuda" in predict_unit.device and "num_gpus" not in ray_actor_options:
        # assign 1 GPU per replica by default if using GPU device
        ray_actor_options["num_gpus"] = 1

    if "num_cpus" not in ray_actor_options:
        ray_actor_options["num_cpus"] = cpu_per_actor

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=cpu_per_actor * num_replicas,
        )
        logging.info("Ray initialized by setup_batch_predict_server")

    serve.start(
        logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
        log_to_driver=False,
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
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)

    logging.info(
        f"BatchPredictServer deployed with max_batch_size={max_batch_size}, "
        f"batch_wait_timeout_s={batch_wait_timeout_s}, num_replicas={num_replicas}, "
        f"name={deployment_name}"
    )

    return handle
