"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.build import molecule
from numpy import testing as npt

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import InferenceSettings

if TYPE_CHECKING:
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


@pytest.mark.gpu()
@pytest.mark.parametrize("vmap", [True, False])
def test_hessian(vmap):
    """Test Hessian calculation using MLIPPredictUnit directly."""
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device="cuda",
        inference_settings=InferenceSettings(
            predict_untrained_hessian={"omol"}, hessian_vmap=vmap
        ),
    )

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    preds = predict_unit.predict(batch)
    hessian = preds["hessian"].detach().cpu().numpy()

    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()


@pytest.mark.xfail(reason="Need to fix the numerical/autograd Hessian calculation")
@pytest.mark.gpu()
def test_hessian_vs_numerical():
    """Test that analytical and numerical Hessians are close."""
    hessian_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device="cuda",
        inference_settings=InferenceSettings(
            predict_untrained_hessian={"omol"}, hessian_vmap=True
        ),
    )
    forces_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device="cuda",
    )

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    preds = hessian_unit.predict(batch)
    hessian_analytical = preds["hessian"].detach().cpu().numpy()

    hessian_numerical = (
        get_numerical_hessian(data, forces_unit, eps=1e-4, device="cuda")
        .detach()
        .cpu()
        .numpy()
    )

    npt.assert_allclose(
        hessian_analytical.diagonal(),
        hessian_numerical.diagonal(),
        rtol=1e-3,  # 0.1% relative tolerance
        atol=1e-3,  # Absolute tolerance for small values
        err_msg="Analytical and numerical Hessians differ significantly",
    )


@pytest.mark.gpu()
def test_hessian_symmetry():
    """Test that the Hessian matrix is symmetric."""
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device="cuda",
        inference_settings=InferenceSettings(
            predict_untrained_hessian={"omol"}, hessian_vmap=True
        ),
    )

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    preds = predict_unit.predict(batch)
    hessian = preds["hessian"]

    npt.assert_allclose(
        hessian.detach().cpu().numpy(),
        hessian.T.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Hessian matrix is not symmetric",
    )
