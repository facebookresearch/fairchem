"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

RayServeMLIPUnit - A predict unit that sends inference to Ray Serve.

This unit behaves like MLIPPredictUnit but sends inference requests to a
centralized Ray Serve deployment instead of running inference locally.

Usage patterns:
1. Inside Ray cluster (e.g., within Ray remote tasks):
   - Don't specify ray_address, it uses local Ray Serve discovery

2. Outside Ray cluster (connecting to remote cluster):
   - Specify ray_address and namespace_serve_fairchem to connect

This allows FAIRChemCalculator to work seamlessly with both local inference
(using MLIPPredictUnit) and remote inference (using RayServeMLIPUnit).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData

logger = logging.getLogger(__name__)


@dataclass
class RayServeTask:
    """Minimal task representation for RayServeMLIPUnit."""

    name: str
    property: str
    level: str = "system"
    datasets: list = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = []


class RayServeMLIPUnit:
    """
    Predict unit that forwards inference to Ray Serve deployment.

    This class provides the same interface as MLIPPredictUnit, allowing
    FAIRChemCalculator to use it transparently. The actual inference
    happens on the Ray Serve deployment.

    Usage patterns:
    1. Inside Ray cluster (no ray_address needed):
       ```python
       @ray.remote
       def my_task():
           unit = RayServeMLIPUnit(model_id="uma-s-1p1")
           calc = FAIRChemCalculator(predict_unit=unit, task_name="omat")
           return atoms.get_potential_energy()
       ```

    2. Outside Ray cluster (specify ray_address):
       ```python
       conn_info = get_ray_connection_info("/path/to/head.json")
       unit = RayServeMLIPUnit(
           ray_address=conn_info["ray_address"],
           namespace_serve_fairchem=conn_info["namespace_serve_fairchem"],
           model_id="uma-s-1p1",
       )
       calc = FAIRChemCalculator(predict_unit=unit, task_name="omat")
       ```

    Args:
        model_id: Model identifier for the checkpoint (e.g., "uma-s-1p1").
        inference_settings: Inference settings ("default" or "turbo")
        ray_address: Ray client address (e.g., "ray://hostname:10001").
            If None, assumes Ray is already connected (use inside Ray tasks).
            Can also be set via RAY_ADDRESS environment variable.
        namespace_serve_fairchem: Ray namespace for the inference server.
            Only used when ray_address is specified.
            Can also be set via RAY_NAMESPACE_SERVE_FAIRCHEM environment variable.
        device: Not used (Ray Serve manages devices), but kept for API compatibility
    """

    def __init__(
        self,
        model_id: str = "uma-s-1p1",
        inference_settings: str = "default",
        ray_address: str | None = None,
        namespace_serve_fairchem: str | None = None,
        device: str = "cuda",  # Ignored - for API compatibility
    ):
        self.model_id = model_id
        self._inference_settings = inference_settings
        self._device = device
        self._client = None
        self._connected = False

        # Check environment variables for ray_address if not provided
        if ray_address is None:
            ray_address = os.environ.get("RAY_ADDRESS")

        self._ray_address = ray_address

        # Get namespace from parameter or environment (only used when ray_address is set)
        if namespace_serve_fairchem is None:
            namespace_serve_fairchem = os.environ.get("RAY_NAMESPACE_SERVE_FAIRCHEM")
        self._namespace_serve_fairchem = namespace_serve_fairchem

        # Build the full model key (checkpoint:settings)
        self._model_key = f"{model_id}:{inference_settings}"

        # Cached metadata - loaded lazily from server
        self._form_elem_refs = None
        self._atom_refs = None
        self._dataset_to_tasks = None

        if self._ray_address:
            logger.info(
                f"RayServeMLIPUnit initialized with model_key={self._model_key}, "
                f"ray_address={ray_address}"
            )
        else:
            logger.info(
                f"RayServeMLIPUnit initialized with model_key={self._model_key} "
                "(using local Ray discovery)"
            )

    def _ensure_connected(self):
        """Connect to Ray cluster if ray_address was specified and not yet connected."""
        if self._connected:
            return

        import ray

        if self._ray_address:
            # External connection mode - connect to remote cluster
            if ray.is_initialized():
                logger.debug(
                    f"Ray already initialized, assuming connected to {self._ray_address}"
                )
            else:
                logger.info(f"Connecting to Ray cluster at {self._ray_address}...")
                ray.init(self._ray_address, namespace=self._namespace_serve_fairchem)
                logger.info("Connected to Ray cluster")

        # If no ray_address, assume we're already inside Ray (local discovery)
        self._connected = True

    @property
    def device(self) -> str:
        """Device property for API compatibility. Ray Serve manages actual device."""
        return self._device

    @property
    def inference_settings(self):
        """Return inference settings object-like wrapper."""
        # Return a simple namespace that has the attributes FAIRChemCalculator expects
        from types import SimpleNamespace
        return SimpleNamespace(external_graph_gen=None)

    @property
    def form_elem_refs(self) -> dict:
        """
        Formation element references from the model.

        Lazily fetched from the server on first access.
        """
        if self._form_elem_refs is None:
            self._fetch_model_metadata()
        return self._form_elem_refs if self._form_elem_refs else {}

    @property
    def atom_refs(self) -> dict:
        """Atom references from the model."""
        if self._atom_refs is None:
            self._fetch_model_metadata()
        return self._atom_refs if self._atom_refs else {}

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        """
        Mapping from dataset names to tasks.

        Required by FAIRChemCalculator to determine valid tasks and properties.
        For UMA models, this typically includes: omat, omol, oc20, odac, omc
        """
        if self._dataset_to_tasks is None:
            self._fetch_model_metadata()
        return self._dataset_to_tasks

    def _get_client(self):
        """Lazy-load the inference client, connecting to Ray if needed."""
        self._ensure_connected()
        if self._client is None:
            from fairchem.core.units.mlip_unit.batch import get_inference_client

            self._client = get_inference_client()
        return self._client

    def _fetch_model_metadata(self):
        """Fetch model metadata from the server (loads model if needed)."""
        client = self._get_client()
        try:
            metadata = client.get_model_metadata(self._model_key)
            self._form_elem_refs = metadata.get("form_elem_refs") or {}
            self._atom_refs = metadata.get("atom_refs") or {}

            # Convert server's dict-based tasks to RayServeTask objects
            fetched_tasks = metadata.get("dataset_to_tasks") or {}
            self._dataset_to_tasks = {}
            for dataset_name, task_dicts in fetched_tasks.items():
                self._dataset_to_tasks[dataset_name] = [
                    RayServeTask(
                        name=t.get("name", dataset_name),
                        property=t["property"],
                        level=t.get("level", "system"),
                        datasets=[dataset_name],
                    )
                    for t in task_dicts
                ]

            if not self._dataset_to_tasks:
                raise ValueError("Server returned empty dataset_to_tasks")

            logger.info(f"Loaded metadata from server: {list(self._dataset_to_tasks.keys())}")
        except Exception as e:
            logger.error(f"Failed to fetch model metadata from server: {e}")
            raise RuntimeError(
                f"Could not fetch model metadata for {self._model_key}. "
                "Ensure the Ray Serve inference server is running."
            ) from e

    def predict(
        self,
        data: AtomicData,
        undo_element_references: bool = True,
    ) -> dict[str, Any]:
        """
        Run inference via Ray Serve.

        Args:
            data: AtomicData with .dataset attribute set to the task name
            undo_element_references: Whether to undo element references

        Returns:
            Dictionary of predictions (energy, forces, stress, etc.) as tensors
        """
        client = self._get_client()

        # The task_name comes from data.dataset (set by FAIRChemCalculator's a2g)
        result = client.predict_from_atomic_data(
            model_key=self._model_key,
            atomic_data=data,
            undo_element_references=undo_element_references,
        )

        # Convert returned values to tensors (FAIRChemCalculator expects tensors)
        # Energy needs shape (1,), forces shape (N, 3), stress shape (1, 3, 3)
        tensor_result = {}
        for key, value in result.items():
            if value is not None:
                arr = np.array(value)
                if key == "energy":
                    # Ensure energy has shape (1,)
                    tensor_result[key] = torch.tensor(arr.reshape(1))
                elif key == "stress":
                    # Ensure stress has shape (1, 3, 3)
                    arr = arr.reshape(-1)[:9].reshape(1, 3, 3)
                    tensor_result[key] = torch.tensor(arr)
                else:
                    tensor_result[key] = torch.tensor(arr)

        return tensor_result

    def move_to_device(self):
        """No-op for RayServeMLIPUnit - Ray Serve manages devices."""

    def __repr__(self) -> str:
        if self._ray_address:
            return (
                f"RayServeMLIPUnit(model_id='{self.model_id}', "
                f"settings='{self._inference_settings}', ray_address='{self._ray_address}')"
            )
        return f"RayServeMLIPUnit(model_id='{self.model_id}', settings='{self._inference_settings}')"


def get_ray_connection_info(head_file: str) -> dict[str, str]:
    """
    Read Ray connection info from a head.json file.

    Args:
        head_file: Path to head.json file from a Ray cluster.

    Returns:
        Dictionary with 'ray_address' and 'namespace_serve_fairchem' keys.

    Example:
        ```python
        from fairchem.core.units.mlip_unit.batch import (
            RayServeMLIPUnit,
            get_ray_connection_info,
        )

        conn_info = get_ray_connection_info("/path/to/head.json")
        unit = RayServeMLIPUnit(
            ray_address=conn_info["ray_address"],
            namespace_serve_fairchem=conn_info["namespace_serve_fairchem"],
            model_id="uma-s-1p1",
        )
        ```
    """
    with open(head_file) as f:
        head_info = json.load(f)

    hostname = head_info.get("hostname")
    client_port = head_info.get("client_port")
    namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")

    if not hostname or not client_port:
        raise ValueError(f"Invalid head.json: missing hostname or client_port in {head_file}")

    return {
        "ray_address": f"ray://{hostname}:{client_port}",
        "namespace_serve_fairchem": namespace_serve_fairchem,
    }
