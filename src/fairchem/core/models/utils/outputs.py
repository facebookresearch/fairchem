# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol


def get_numerical_hessian(
    data: AtomicData,
    predict_unit: MLIPPredictUnitProtocol,
    eps: float = 1e-4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Calculate the Hessian matrix for the given atomic data using finite differences.

    This function computes the Hessian matrix by displacing each atom in each
    Cartesian direction and computing the change in forces using finite differences.
    The Hessian H = d²E/dx² = -dF/dx.

    Args:
        data: The atomic data to calculate the Hessian for. Must contain
            positions ('pos') and number of atoms ('natoms').
        predict_unit: An instance implementing MLIPPredictUnitProtocol, which
            provides a predict method that takes an AtomicData object and returns
            a dictionary containing at least 'forces' as a key.
        eps: The finite difference step size. Defaults to 1e-4.
        device: The device to create the output tensor on. Defaults to "cpu".

    Returns:
        The Hessian matrix with shape (n_atoms * 3, n_atoms * 3).

    Example:
        >>> from fairchem.core.models.utils.outputs import get_numerical_hessian
        >>> # Assuming you have a predict_unit instance
        >>> hessian = get_numerical_hessian(
        ...     data=atomic_data,
        ...     predict_unit=predict_unit,
        ...     eps=1e-4,
        ...     device="cuda"
        ... )
    """
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    n_atoms = data.natoms.item() if hasattr(data.natoms, "item") else int(data.natoms)

    # Create displaced data objects in batch
    data_list = []
    for i in range(n_atoms):
        for j in range(3):
            # Create displaced versions
            data_plus = data.clone()
            data_minus = data.clone()

            data_plus.pos[i, j] += eps
            data_minus.pos[i, j] -= eps

            data_list.append(data_plus)
            data_list.append(data_minus)

    # Batch and predict
    batch = atomicdata_list_to_batch(data_list)
    pred = predict_unit.predict(batch)

    # Get the forces
    forces = pred["forces"].reshape(-1, n_atoms, 3)

    # Calculate the Hessian using finite differences
    # Hessian H = d²E/dx² = -dF/dx, so we need -(F+ - F-) / (2*eps)
    hessian = torch.zeros(
        (n_atoms * 3, n_atoms * 3), device=device, requires_grad=False
    )
    for i in range(n_atoms):
        for j in range(3):
            idx = i * 3 + j
            force_plus = forces[2 * idx].flatten()
            force_minus = forces[2 * idx + 1].flatten()
            hessian[:, idx] = -(force_plus - force_minus) / (2 * eps)

    return hessian
