"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Literal

import torch

from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import UMATask
from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets.atomic_data import AtomicData

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.transforms import pbc_wrap_batched
except ImportError:
    ts = None  # type: ignore[assignment]
    ModelInterface = None  # type: ignore[assignment]
    pbc_wrap_batched = None  # type: ignore[assignment]


if typing.TYPE_CHECKING:
    from torch_sim import SimState

# Use object as fallback base class if ModelInterface is not available
# The __init__ method will raise ImportError if torch-sim is not installed
_TSModelInterface = ModelInterface if ModelInterface is not None else object


def _simstate_to_atomicdata_batch(
    sim_state: ts.SimState,
    task_name: UMATask | str | None,
    target_dtype: torch.dtype = torch.float32,
) -> AtomicData:
    """Convert a batched SimState directly to a batched AtomicData in a single vectorized operation.

    Args:
        sim_state: A SimState containing one or more systems
        task_name: Task name for UMA models
        target_dtype: Target dtype for tensors

    Returns:
        Batched AtomicData object
    """
    positions = sim_state.positions.to(target_dtype)
    atomic_numbers = sim_state.atomic_numbers.long()

    n_atoms_per_system = torch.bincount(sim_state.system_idx)
    n_systems = sim_state.n_systems

    if sim_state.cell is not None:
        cell = sim_state.row_vector_cell.to(target_dtype)  # (n_systems, 3, 3)
    else:
        # Create identity cells for molecules
        cell = (
            torch.eye(3, dtype=target_dtype, device=positions.device)
            .unsqueeze(0)
            .expand(n_systems, -1, -1)
        )

    pbc = sim_state.pbc.bool()  # Already a tensor, shape (3,)
    pbc = pbc.unsqueeze(0).expand(n_systems, 3)  # (n_systems, 3)

    if torch.any(pbc):
        cell_col = cell.transpose(-2, -1)  # (n_systems, 3, 3) column vectors
        pbc_single = pbc[0] if pbc.ndim > 0 else pbc  # (3,) or bool
        positions = pbc_wrap_batched(
            positions, cell_col, sim_state.system_idx, pbc_single
        )

    natoms = n_atoms_per_system  # (n_systems,)

    edge_index = torch.empty((2, 0), dtype=torch.long, device=positions.device)
    cell_offsets = torch.empty((0, 3), dtype=target_dtype, device=positions.device)
    nedges = torch.zeros(n_systems, dtype=torch.long, device=positions.device)

    charge = (
        sim_state.charge.long()
        if sim_state.has_extras("charge")
        else torch.zeros(n_systems, dtype=torch.long, device=positions.device)
    )
    spin = (
        sim_state.spin.long()
        if sim_state.has_extras("spin")
        else torch.zeros(n_systems, dtype=torch.long, device=positions.device)
    )

    fixed = torch.zeros_like(atomic_numbers, dtype=torch.long)
    tags = torch.zeros_like(atomic_numbers, dtype=torch.long)

    batch_indices = sim_state.system_idx.long()

    batched_dict = {
        "pos": positions,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "pbc": pbc,
        "natoms": natoms,
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "nedges": nedges,
        "charge": charge,
        "spin": spin,
        "fixed": fixed,
        "tags": tags,
        "batch": batch_indices,
        "sid": [""] * n_systems,  # Empty string IDs for each system
    }

    if task_name is not None:
        task_name_str = task_name if isinstance(task_name, str) else task_name.value
        batched_dict["dataset"] = [task_name_str] * n_systems

    atomic_data_batch = AtomicData.from_dict(batched_dict)

    natoms_list = n_atoms_per_system.tolist()
    n_atoms_cumsum = torch.cumsum(n_atoms_per_system, dim=0).tolist()

    slices = {}
    cumsum = {}
    cat_dims = {}

    for key, value in batched_dict.items():
        if key == "batch" or not isinstance(value, torch.Tensor):
            continue

        cat_dims[key] = atomic_data_batch.__cat_dim__(key, value) or 0

        is_edge_key = "index" in key or key == "cell_offsets"
        if is_edge_key:
            slices[key] = [0] * (n_systems + 1)
            cumsum[key] = [0] + n_atoms_cumsum
        elif key in ("cell", "pbc", "charge", "spin", "natoms", "nedges"):
            slices[key] = list(range(n_systems + 1))
            cumsum[key] = [0] * (n_systems + 1)
        else:
            slices[key] = [0] + n_atoms_cumsum
            cumsum[key] = [0] * (n_systems + 1)

    atomic_data_batch.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)

    return atomic_data_batch.contiguous()


class FairChemModel(_TSModelInterface):
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
            ImportError: If torch-sim is not installed
            NotImplementedError: If custom neighbor list function is provided
            ValueError: If model is not a known model name or valid file path
        """
        if ts is None or ModelInterface is None:
            raise ImportError(
                "torch-sim is required to use FairChemModel. "
                + "Install it with: pip install fairchem-core[torchsim]"
            )

        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._memory_scales_with = "n_atoms"  # TODO: this does vary with model type

        # Convert Path to string for consistency
        model = str(model) if isinstance(model, Path) else model
        model_cache_dir = (
            str(model_cache_dir)
            if isinstance(model_cache_dir, Path)
            else model_cache_dir
        )

        # Convert task_name to UMATask if it's a string (only for UMA models)
        if isinstance(task_name, str):
            task_name = UMATask(task_name)

        # Use the efficient predictor API for optimal performance
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device_str: Literal["cuda", "cpu"] = (
            self._device.type
        )  # ty:ignore[invalid-assignment]
        self.task_name = task_name

        # Create efficient batch predictor for fast inference
        if model in pretrained_mlip.available_models:
            if model_cache_dir and os.path.exists(model_cache_dir):
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

    def forward(self, state: SimState, **kwargs: Any) -> dict[str, torch.Tensor]:
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
        if state.device != self._device:
            state = state.to(self._device)

        # Convert SimState to batched AtomicData in a single vectorized operation
        batch = _simstate_to_atomicdata_batch(
            sim_state=state,
            task_name=self.task_name,
            target_dtype=self._dtype,
        )
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
            if stress.dim() == 2 and stress.shape[0] == batch.num_graphs:
                stress = stress.view(-1, 3, 3)
            results["stress"] = stress

        return {k: v.detach() for k, v in results.items()}
