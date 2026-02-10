"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def handle_single_atom_prediction(
    data: AtomicData,
    atom_refs: dict[str, dict],
    tasks: dict[str, Task],
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """
    Handle prediction for single-atom systems without PBC.

    Single isolated atoms (natoms==1, no PBC) cannot be processed by the model,
    so we use precomputed DFT reference energies instead.

    This function only handles the case where a single atom is the sole entry
    in a batch. If a batch contains multiple systems and any of them is a single
    atom, this function raises an error.

    Args:
        data: The AtomicData batch
        atom_refs: Dictionary mapping dataset names to their atom reference dictionaries
        tasks: Dictionary mapping task names to Task objects
        device: Device to place output tensors on

    Returns:
        Dictionary mapping task names to prediction tensors if the batch contains
        a single isolated atom, or None if the batch should be processed normally.

    Raises:
        ValueError: If the batch contains multiple systems and at least one is a
            single isolated atom (unsupported case).
    """
    # Identify single-atom systems (natoms==1 and pbc all False)
    is_single_atom = data.natoms == 1
    has_no_pbc = ~data.pbc.any(dim=1)
    single_atom_mask = is_single_atom & has_no_pbc
    num_single_atoms = single_atom_mask.sum().item()
    num_systems = len(data.natoms)

    # No single atoms in batch - proceed with normal prediction
    if num_single_atoms == 0:
        return None

    # Check for unsupported case: multiple systems with at least one single atom
    if num_systems > 1:
        raise ValueError(
            f"Batch contains {num_systems} systems, {num_single_atoms} of which are "
            "single isolated atoms. Single atoms must be batched alone (one per batch). "
            "Please submit single-atom systems in separate batches."
        )

    # Handle single-atom case (exactly one system that is a single atom)
    if not atom_refs:
        raise ValueError(
            "Single atom system encountered but no atom_refs available. "
            "Please call fairchem.core.pretrained_mlip.get_predict_unit() "
            "with an appropriate checkpoint name."
        )

    logging.warning(
        "Single atom system detected; using precomputed DFT references "
        "instead of model predictions. Spin multiplicity is ignored for "
        "monoatomic systems."
    )

    # Get atomic number and charge for the single atom
    atomic_number = int(data.atomic_numbers[0].item())
    charge = int(data.charge[0].item())
    dataset_name = data.dataset[0]

    if dataset_name not in atom_refs:
        raise ValueError(
            f"No atom references available for dataset '{dataset_name}'. "
            "Cannot compute single-atom energy."
        )
    ds_refs = atom_refs[dataset_name]
    try:
        energy = ds_refs.get(atomic_number, {}).get(charge)
    except AttributeError:
        energy = ds_refs[atomic_number]

    if energy is None:
        raise ValueError(
            f"No atom reference for element {atomic_number} with charge {charge} "
            f"in dataset '{dataset_name}'."
        )

    # Build output dict for single atom
    pred_output = {}
    for task_name, task in tasks.items():
        if dataset_name not in task.datasets:
            continue
        if task.property == "energy":
            pred_output[task_name] = torch.tensor(
                [float(energy)], dtype=torch.float32, device=device
            )
        elif task.property == "forces":
            pred_output[task_name] = torch.zeros(
                (1, 3), dtype=torch.float32, device=device
            )
        elif task.property == "stress":
            pred_output[task_name] = torch.zeros(
                (1, 9), dtype=torch.float32, device=device
            )

    return pred_output
