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

pytestmark = pytest.mark.gpu

# Extensivity is a property of the architecture, not of any particular head:
# every task head should satisfy E(N replicas) = N * E(1 replica) when the
# replicas don't interact. We therefore parametrize across every head shipped
# with the pretrained model.
TASK_NAMES = ["oc20", "omat", "omol", "odac", "omc", "oc25"]

SUPERCELL_CONFIGS = [
    pytest.param(np.diag([2, 2, 2]), 8, id="2x2x2"),
    pytest.param(np.diag([2, 1, 1]), 2, id="2x1x1"),
    pytest.param(np.diag([4, 4, 4]), 64, id="4x4x4"),
    pytest.param(np.diag([4, 1, 1]), 4, id="4x1x1"),
    pytest.param(np.diag([3, 4, 1]), 12, id="3x4x1"),
]


@pytest.fixture(scope="module")
def predict_unit():
    return pretrained_mlip.get_predict_unit("uma-s-1p1")


@pytest.fixture(scope="module", params=TASK_NAMES)
def calc(predict_unit, request):
    task_name = request.param
    if task_name not in predict_unit.dataset_to_tasks:
        pytest.skip(f"task {task_name!r} not supported by uma-s-1p1")
    return FAIRChemCalculator(predict_unit, task_name=task_name)


def _set_neutral_singlet(atoms):
    # charge=0, spin=1 are accepted defaults for every UMA task head
    # (omol requires spin=1; the materials heads accept any value).
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1


def _assert_multiset_close(a, b, *, atol):
    # Compare two arrays as multisets of row vectors: extensivity says the
    # supercell forces equal the unit-cell forces repeated `multiplier` times,
    # but ASE's make_supercell doesn't guarantee `i % n_unit` ordering.
    order_a = np.lexsort(a.T)
    order_b = np.lexsort(b.T)
    npt.assert_allclose(a[order_a], b[order_b], atol=atol)


# --- PBC extensivity ---


@pytest.mark.parametrize("supercell_matrix, multiplier", SUPERCELL_CONFIGS)
def test_pbc_extensivity_energy(supercell_matrix, multiplier, calc):
    atoms_unit = bulk("MgO", "rocksalt", a=4.213)
    _set_neutral_singlet(atoms_unit)
    atoms_unit.calc = calc
    energy_unit = atoms_unit.get_potential_energy()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    _set_neutral_singlet(atoms_super)
    atoms_super.calc = calc
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
def test_pbc_extensivity_forces(supercell_matrix, multiplier, calc):
    atoms_unit = bulk("MgO", "rocksalt", a=4.213)
    _set_neutral_singlet(atoms_unit)
    atoms_unit.calc = calc
    forces_unit = atoms_unit.get_forces()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    _set_neutral_singlet(atoms_super)
    atoms_super.calc = calc
    forces_super = atoms_super.get_forces()

    forces_unit_tiled = np.vstack([forces_unit] * multiplier)
    _assert_multiset_close(forces_super, forces_unit_tiled, atol=1e-4)


# --- Isolated-cluster extensivity ---


def test_isolated_extensivity_energy(calc):
    # Two H2O copies separated by 50 A, well beyond the 6 A model cutoff —
    # the combined-system energy must equal twice the single-molecule energy.
    mol_single = molecule("H2O")
    _set_neutral_singlet(mol_single)
    mol_single.calc = calc
    energy_single = mol_single.get_potential_energy()

    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    mol_combined = mol_single.copy() + mol_copy
    _set_neutral_singlet(mol_combined)
    mol_combined.calc = calc
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


def test_isolated_extensivity_forces(calc):
    mol_single = molecule("H2O")
    _set_neutral_singlet(mol_single)
    mol_single.calc = calc
    forces_single = mol_single.get_forces()

    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    mol_combined = mol_single.copy() + mol_copy
    _set_neutral_singlet(mol_combined)
    mol_combined.calc = calc
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
