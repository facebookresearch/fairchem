"""FairChem model wrapper for torch-sim.

Provides a TorchSim-compatible interface to FairChem models for computing
energies, forces, and stresses of atomistic systems.
"""

from __future__ import annotations

import os
import typing
from pathlib import Path

import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface

from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import UMATask
from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.typing import StateDict


class FairChemModel(ModelInterface):
    """FairChem model wrapper for computing atomistic properties.

    Wraps FairChem models to compute energies, forces, and stresses. Can be
    initialized with a model checkpoint path or pretrained model name.

    Uses the fairchem-core-2.2.0+ predictor API for batch inference.

    Attributes:
        predictor: The FairChem predictor for batch inference
        task_name (UMATask): Task type for the model
        _device (torch.device): Device where computation is performed
        _dtype (torch.dtype): Data type used for computation
        _compute_stress (bool): Whether to compute stress tensor
        implemented_properties (list): Model outputs the model can compute

    Examples:
        >>> model = FairChemModel(model="path/to/checkpoint.pt", compute_stress=True)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: str | Path,
        neighbor_list_fn: Callable | None = None,
        *,  # force remaining arguments to be keyword-only
        model_cache_dir: str | Path | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
        task_name: UMATask | str | None = None,
    ) -> None:
        """Initialize the FairChem model.

        Args:
            model (str | Path): Either a pretrained model name or path to model
                checkpoint file. The function will first check if the input matches
                a known pretrained model name, then check if it's a valid file path.
            neighbor_list_fn (Callable | None): Function to compute neighbor lists
                (not currently supported)
            model_cache_dir (str | Path | None): Path where to save the model
            device (torch.device | None): Device to use for computation. If None,
                defaults to CUDA if available, otherwise CPU.
            dtype (torch.dtype | None): Data type to use for computation
            compute_stress (bool): Whether to compute stress tensor
            task_name (UMATask | str | None): Task type for UMA models (optional,
                only needed for UMA models)

        Raises:
            NotImplementedError: If custom neighbor list function is provided
            ValueError: If model is not a known model name or valid file path
        """
        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._memory_scales_with = "n_atoms"

        if neighbor_list_fn is not None:
            raise NotImplementedError(
                "Custom neighbor list is not supported for FairChemModel."
            )

        # Convert Path to string for consistency
        if isinstance(model, Path):
            model = str(model)

        # Convert task_name to UMATask if it's a string (only for UMA models)
        if isinstance(task_name, str):
            task_name = UMATask(task_name)

        # Use the efficient predictor API for optimal performance
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device_str = str(self._device)
        self.task_name = task_name

        # Create efficient batch predictor for fast inference
        if model in pretrained_mlip.available_models:
            if model_cache_dir and model_cache_dir.exists():
                self.predictor = pretrained_mlip.get_predict_unit(
                    model, device=device_str, cache_dir=model_cache_dir
                )
            else:
                self.predictor = pretrained_mlip.get_predict_unit(
                    model, device=device_str
                )
        elif os.path.isfile(model):
            self.predictor = pretrained_mlip.load_predict_unit(model, device=device_str)
        else:
            raise ValueError(
                f"Invalid model name or checkpoint path: {model}. "
                f"Available pretrained models are: {pretrained_mlip.available_models}"
            )

        # Determine implemented properties
        # This is a simplified approach - in practice you might want to
        # inspect the model configuration more carefully
        self.implemented_properties = ["energy", "forces"]
        if compute_stress:
            self.implemented_properties.append("stress")

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type used by the model."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        return self._device

    def forward(self, state: ts.SimState | StateDict) -> dict:
        """Compute energies, forces, and other properties.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3]
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        if sim_state.device != self._device:
            sim_state = sim_state.to(self._device)

        # Ensure system_idx has integer dtype (SimState guarantees presence)
        if sim_state.system_idx.dtype != torch.int64:
            sim_state.system_idx = sim_state.system_idx.to(dtype=torch.int64)

        # Convert SimState to AtomicData objects for efficient batch processing
        from ase import Atoms

        n_atoms = torch.bincount(sim_state.system_idx)
        atomic_data_list = []

        for idx, (n, c) in enumerate(
            zip(n_atoms, torch.cumsum(n_atoms, dim=0), strict=False)
        ):
            # Extract system data
            positions = sim_state.positions[c - n : c].cpu().numpy()
            atomic_nums = sim_state.atomic_numbers[c - n : c].cpu().numpy()
            pbc = sim_state.pbc.cpu().numpy()
            cell = (
                sim_state.row_vector_cell[idx].cpu().numpy()
                if sim_state.row_vector_cell is not None
                else None
            )

            # Create ASE Atoms object first
            atoms = Atoms(
                numbers=atomic_nums,
                positions=positions,
                cell=cell,
                pbc=pbc if cell is not None else False,
            )

            atoms.info["charge"] = sim_state.charge[idx].item()
            atoms.info["spin"] = sim_state.spin[idx].item()

            # Convert ASE Atoms to AtomicData (task_name only applies to UMA models)
            # r_data_keys must be passed for charge/spin to be read from atoms.info
            if self.task_name is None:
                atomic_data = AtomicData.from_ase(atoms, r_data_keys=["charge", "spin"])
            else:
                atomic_data = AtomicData.from_ase(
                    atoms, task_name=self.task_name, r_data_keys=["charge", "spin"]
                )
            atomic_data_list.append(atomic_data)

        # Create batch for efficient inference
        batch = atomicdata_list_to_batch(atomic_data_list)
        batch = batch.to(self._device)

        # Run efficient batch prediction
        predictions = self.predictor.predict(batch)

        # Convert predictions to torch-sim format
        results: dict[str, torch.Tensor] = {}
        results["energy"] = predictions["energy"].to(dtype=self._dtype)
        results["forces"] = predictions["forces"].to(dtype=self._dtype)

        # Handle stress if requested and available
        if self._compute_stress and "stress" in predictions:
            stress = predictions["stress"].to(dtype=self._dtype)
            # Ensure stress has correct shape [batch_size, 3, 3]
            if stress.dim() == 2 and stress.shape[0] == len(atomic_data_list):
                stress = stress.view(-1, 3, 3)
            results["stress"] = stress

        return results
