"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from ase.build import bulk, make_supercell, molecule

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip

# Mark all tests in this module as GPU tests
pytestmark = pytest.mark.gpu

# Supercell configurations: (transformation_matrix, integer_multiplier)
SUPERCELL_CONFIGS = [
    pytest.param(np.diag([2, 2, 2]), 8, id="2x2x2"),
    pytest.param(np.diag([2, 1, 1]), 2, id="2x1x1"),
    pytest.param(np.diag([4, 4, 4]), 64, id="4x4x4"),
    pytest.param(np.diag([4, 1, 1]), 4, id="4x1x1"),
    pytest.param(np.diag([3, 4, 1]), 12, id="3x4x1"),
]


@pytest.fixture(scope="module")
def predict_unit():
    """
    Module-scoped predict unit for uma-s-1p1.
    """
    return pretrained_mlip.get_predict_unit("uma-s-1p1")


@pytest.fixture(scope="module")
def omat_calc(predict_unit):
    """
    FAIRChemCalculator configured for the omat task.
    """
    return FAIRChemCalculator(predict_unit, task_name="omat")


@pytest.fixture(scope="module")
def omol_calc(predict_unit):
    """
    FAIRChemCalculator configured for the omol task.
    """
    return FAIRChemCalculator(predict_unit, task_name="omol")


# --- PBC extensivity tests ---


@pytest.mark.parametrize("supercell_matrix, multiplier", SUPERCELL_CONFIGS)
def test_pbc_extensivity_energy(supercell_matrix, multiplier, omat_calc):
    """
    Verify energy scales linearly with supercell size for PBC systems.

    For a periodic crystal tiled into a supercell, the total energy must
    equal the unit cell energy times the number of replicas.
    """
    atoms_unit = bulk("MgO", "rocksalt", a=4.213)
    atoms_unit.calc = omat_calc
    energy_unit = atoms_unit.get_potential_energy()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    atoms_super.calc = omat_calc
    energy_super = atoms_super.get_potential_energy()

    npt.assert_allclose(
        energy_super / energy_unit,
        multiplier,
        rtol=1e-4,
        err_msg=(
            f"Energy does not scale by {multiplier}x for supercell. "
            f"E_unit={energy_unit:.6f}, E_super={energy_super:.6f}"
        ),
    )


@pytest.mark.parametrize("supercell_matrix, multiplier", SUPERCELL_CONFIGS)
def test_pbc_extensivity_forces(supercell_matrix, multiplier, omat_calc):
    """
    Verify forces in a supercell match the tiled unit cell forces.

    Each atom in the supercell sees the same local environment as the
    corresponding atom in the unit cell, so forces must be identical.
    """
    atoms_unit = bulk("MgO", "rocksalt", a=4.213)
    atoms_unit.calc = omat_calc
    forces_unit = atoms_unit.get_forces()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    atoms_super.calc = omat_calc
    forces_super = atoms_super.get_forces()

    n_unit = len(atoms_unit)
    for i in range(len(atoms_super)):
        unit_idx = i % n_unit
        npt.assert_allclose(
            forces_super[i],
            forces_unit[unit_idx],
            atol=1e-4,
            err_msg=(
                f"Force mismatch: supercell atom {i} " f"vs unit cell atom {unit_idx}"
            ),
        )


# --- OMol (molecular) extensivity tests ---


def test_omol_extensivity_energy(omol_calc):
    """
    Verify energy extensivity for isolated molecules.

    Two identical H2O molecules separated by 50 angstroms (well beyond
    the 6 angstrom model cutoff) should have a combined energy equal to
    twice the single molecule energy.
    """
    mol_single = molecule("H2O")
    mol_single.info["charge"] = 0
    mol_single.info["spin"] = 1
    mol_single.calc = omol_calc
    energy_single = mol_single.get_potential_energy()

    # Create second copy far away (beyond model cutoff)
    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    # Combine into one system
    mol_combined = mol_single.copy() + mol_copy
    mol_combined.info["charge"] = 0
    mol_combined.info["spin"] = 1
    mol_combined.calc = omol_calc
    energy_combined = mol_combined.get_potential_energy()

    npt.assert_allclose(
        energy_combined,
        2.0 * energy_single,
        atol=1e-4,
        err_msg=(
            f"Energy is not extensive for two separated molecules. "
            f"E_single={energy_single:.6f}, E_combined={energy_combined:.6f}"
        ),
    )


def test_omol_extensivity_forces(omol_calc):
    """
    Verify force extensivity for isolated molecules.

    Forces on each molecule in the combined system should match the
    forces on the isolated single molecule.
    """
    mol_single = molecule("H2O")
    mol_single.info["charge"] = 0
    mol_single.info["spin"] = 1
    mol_single.calc = omol_calc
    forces_single = mol_single.get_forces()

    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    mol_combined = mol_single.copy() + mol_copy
    mol_combined.info["charge"] = 0
    mol_combined.info["spin"] = 1
    mol_combined.calc = omol_calc
    forces_combined = mol_combined.get_forces()

    n = len(mol_single)
    npt.assert_allclose(
        forces_combined[:n],
        forces_single,
        atol=1e-4,
        err_msg="Forces on first molecule don't match isolated molecule.",
    )
    npt.assert_allclose(
        forces_combined[n:],
        forces_single,
        atol=1e-4,
        err_msg="Forces on second molecule don't match isolated molecule.",
    )
