"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FAIRChem Inference Client - Client for the Ray Serve inference server.

This client is used by RayServeMLIPUnit to submit inference requests.
It's a singleton that maintains a persistent handle to the Ray Serve deployment.

Model key format: "{checkpoint_name}:{inference_settings}"
Task name comes from AtomicData.dataset at inference time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData

logger = logging.getLogger(__name__)


class FAIRChemInferenceClient:
    """
    Singleton client for FAIRChem inference server.

    Uses Ray Serve deployment handle to submit predictions.
    Thread-safe and async-compatible.
    """

    _instance = None
    _handle = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton - useful for testing."""
        cls._instance = None
        cls._handle = None

    def _get_handle(self):
        """Get or create the deployment handle."""
        if FAIRChemInferenceClient._handle is None:
            from ray import serve

            # Get handle to the inference deployment
            FAIRChemInferenceClient._handle = serve.get_deployment_handle(
                "ConfiguredFAIRChemInferenceServer",
                "fairchem_inference",
            )
            logger.info("Connected to FAIRChem inference server")

        return FAIRChemInferenceClient._handle

    async def predict_from_atomic_data_async(
        self,
        model_key: str,
        atomic_data: AtomicData,
        undo_element_references: bool = True,
    ) -> dict[str, Any]:
        """
        Submit async prediction request with AtomicData.

        Args:
            model_key: Model identifier (checkpoint:settings format)
            atomic_data: AtomicData object (with .dataset set to task_name)
            undo_element_references: Whether to undo element references

        Returns:
            Dictionary with predictions (energy, forces, stress, etc.)
        """
        handle = self._get_handle()
        request = {
            "model_key": model_key,
            "atomic_data": atomic_data,
            "undo_element_references": undo_element_references,
        }

        return await handle.remote(request)

    def predict_from_atomic_data(
        self,
        model_key: str,
        atomic_data: AtomicData,
        undo_element_references: bool = True,
    ) -> dict[str, Any]:
        """
        Submit sync prediction request with AtomicData.

        Convenience wrapper around predict_from_atomic_data_async for non-async code.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    self.predict_from_atomic_data_async(model_key, atomic_data, undo_element_references),
                    loop,
                )
                return future.result()
            else:
                return loop.run_until_complete(
                    self.predict_from_atomic_data_async(model_key, atomic_data, undo_element_references)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.predict_from_atomic_data_async(model_key, atomic_data, undo_element_references)
            )

    async def predict_async(
        self,
        model_key: str,
        atoms: Atoms,
        task_name: str,
        undo_element_references: bool = True,
    ) -> dict[str, Any]:
        """
        Submit async prediction request with ASE Atoms.

        Args:
            model_key: Model identifier (checkpoint:settings format)
            atoms: ASE Atoms object
            task_name: Task name for UMA models (e.g., "omat", "omol")
            undo_element_references: Whether to undo element references

        Returns:
            Dictionary with predictions (energy, forces, stress, etc.)
        """
        from fairchem.core.datasets.atomic_data import AtomicData

        # Convert atoms to AtomicData with task_name
        atomic_data = AtomicData.from_ase(atoms, task_name=task_name)

        return await self.predict_from_atomic_data_async(
            model_key, atomic_data, undo_element_references
        )

    def predict(
        self,
        model_key: str,
        atoms: Atoms,
        task_name: str,
        undo_element_references: bool = True,
    ) -> dict[str, Any]:
        """
        Submit sync prediction request with ASE Atoms.

        Convenience wrapper around predict_async for non-async code.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.predict_async(model_key, atoms, task_name, undo_element_references),
                    loop,
                )
                return future.result()
            else:
                return loop.run_until_complete(
                    self.predict_async(model_key, atoms, task_name, undo_element_references)
                )
        except RuntimeError:
            return asyncio.run(
                self.predict_async(model_key, atoms, task_name, undo_element_references)
            )

    def get_model_metadata(self, model_key: str) -> dict[str, Any]:
        """
        Get metadata for a model (form_elem_refs, atom_refs, dataset_to_tasks).

        This fetches metadata from the server, loading the model if necessary.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._get_model_metadata_async(model_key),
                    loop,
                )
                return future.result(timeout=60)
            else:
                return loop.run_until_complete(self._get_model_metadata_async(model_key))
        except RuntimeError:
            return asyncio.run(self._get_model_metadata_async(model_key))

    async def _get_model_metadata_async(self, model_key: str) -> dict[str, Any]:
        """Async version of get_model_metadata."""
        handle = self._get_handle()
        request = {
            "request_type": "metadata",
            "model_key": model_key,
        }
        return await handle.remote(request)

    async def predict_multiple_async(
        self,
        model_key: str,
        atoms_list: list[Atoms],
        task_name: str,
        undo_element_references: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Submit multiple predictions in parallel.

        More efficient than calling predict() multiple times.
        """
        from fairchem.core.datasets.atomic_data import AtomicData

        handle = self._get_handle()

        # Submit all requests
        futures = []
        for atoms in atoms_list:
            atomic_data = AtomicData.from_ase(atoms, task_name=task_name)
            request = {
                "model_key": model_key,
                "atomic_data": atomic_data,
                "undo_element_references": undo_element_references,
            }
            futures.append(handle.remote(request))

        # Wait for all results
        results = await asyncio.gather(*futures)
        return list(results)

    def predict_multiple(
        self,
        model_key: str,
        atoms_list: list[Atoms],
        task_name: str,
        undo_element_references: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Sync version of predict_multiple_async.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.predict_multiple_async(model_key, atoms_list, task_name, undo_element_references),
                    loop,
                )
                return future.result()
            else:
                return loop.run_until_complete(
                    self.predict_multiple_async(model_key, atoms_list, task_name, undo_element_references)
                )
        except RuntimeError:
            return asyncio.run(
                self.predict_multiple_async(model_key, atoms_list, task_name, undo_element_references)
            )


def get_inference_client() -> FAIRChemInferenceClient:
    """Get the singleton inference client instance."""
    return FAIRChemInferenceClient()
