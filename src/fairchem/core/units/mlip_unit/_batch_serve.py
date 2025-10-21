"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import ray
from ray import serve
from ray.serve.schema import LoggingConfig

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


@serve.deployment(max_ongoing_requests=100, logging_config=LoggingConfig(log_level="WARNING"))
class BatchPredictServer:
    """
    Ray Serve deployment that batches incoming inference requests.
    """

    def __init__(self, predict_unit_ref, max_batch_size: int = 32, batch_wait_timeout_s: float = 0.05):
        """
        Initialize with a Ray object reference to a PredictUnit.
        
        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.configure_batching(max_batch_size, batch_wait_timeout_s)

        logging.info("BatchedPredictor initialized with predict_unit from object store")

    def configure_batching(self, max_batch_size: int = 32, batch_wait_timeout_s: float = 0.05):
        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)

    @serve.batch
    async def predict(self, data_list: list[AtomicData]) -> list[dict]:
        """
        Process a batch of AtomicData objects.
          
        Args:
            data_list: List of AtomicData objects (automatically batched by Ray Serve)
              
        Returns:
            List of prediction dictionaries, one per input
        """
        batch = atomicdata_list_to_batch(data_list)
        predictions = self.predict_unit.predict(batch)        
        prediction_list = self._split_predictions(predictions, batch)
        
        return prediction_list

    async def __call__(self, data: AtomicData) -> dict:
        """
        Main entry point for inference requests.
        
        Args:
            data: Single AtomicData object
            
        Returns:
            Prediction dictionary for this system
        """
        predictions = await self.predict(data)
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
                if pred.dim() == 0:
                    # Scalar prediction (shouldn't happen in batched case)
                    system_predictions[key] = pred
                elif pred.dim() == 1:
                    # System-level prediction (e.g., energy)
                    system_predictions[key] = pred[i:i+1]
                else:
                    # Atom-level prediction (e.g., forces)
                    # Extract predictions for atoms belonging to system i
                    mask = batch.batch == i
                    system_predictions[key] = pred[mask]
            
            split_preds.append(system_predictions)
        
        return split_preds
