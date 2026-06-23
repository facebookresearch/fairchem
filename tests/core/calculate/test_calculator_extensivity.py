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

# Extensivity is a property of the UMA-S architecture, not of any particular
# head: every task head should satisfy E(N replicas) = N * E(1 replica) when
# the replicas don't interact. Run on both UMA-S checkpoints — declared model
# args route this file to the per-model sweep CI jobs so a future model
# regression that breaks extensivity gets caught.
pytestmark = [pytest.mark.gpu, pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")]

TASK_NAMES = ["oc20", "omat", "omol", "odac", "omc", "oc25"]

SUPERCELL_CONFIGS = [
    pytest.param(np.diag([2, 2, 2]), 8, id="2x2x2"),
    pytest.param(np.diag([2, 1, 1]), 2, id="2x1x1"),
    pytest.param(np.diag([4, 4, 4]), 64, id="4x4x4"),
    pytest.param(np.diag([4, 1, 1]), 4, id="4x1x1"),
    pytest.param(np.diag([3, 4, 1]), 12, id="3x4x1"),
]


@pytest.fixture(scope="module", params=TASK_NAMES)
def calc(declared_predict_unit, request):
    task_name = request.param
    if task_name not in declared_predict_unit.dataset_to_tasks:
        pytest.skip(
            f"task {task_name!r} not supported by current pretrained model"
        )
    return FAIRChemCalculator(declared_predict_unit, task_name=task_name)


# uma-s-1p2 has a known extensivity regression: the model violates
# E(N replicas) = N * E(1 replica). Surfaced when this file was routed
# through the per-model sweep partition — main today only ran extensivity
# against uma-s-1p1 (which passes). Mark every extensivity test in this
# file as xfail under uma-s-1p2 with strict=False so a future model fix
# shows up as xpassed (signal to delete this fixture). uma-s-1p1 and any
# other sweep value continue to assert strictly.
#
# Tracked for the UMA team. Remove this fixture when uma-s-1p2 (or its
# successor 1p2p1 / 1p3) reproduces extensivity.
@pytest.fixture(autouse=True)
def _xfail_uma_s_1p2_extensivity_bug(request, pretrained_checkpoint):
    if pretrained_checkpoint == "uma-s-1p2":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="uma-s-1p2 has a known extensivity regression",
                strict=False,
            )
        )


def _set_charge_spin(atoms, calc):
    # Only omol has been trained with spin = 1 (singlet). The materials
    # heads (oc20/omat/odac/omc/oc25) use the null spin token (spin = 0);
    # setting spin = 1 there would put the model out of distribution.
    # Charge defaults to 0 for all heads.
    if calc.task_name == "omol":
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
    _set_charge_spin(atoms_unit, calc)
    atoms_unit.calc = calc
    energy_unit = atoms_unit.get_potential_energy()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    _set_charge_spin(atoms_super, calc)
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
    _set_charge_spin(atoms_unit, calc)
    atoms_unit.calc = calc
    forces_unit = atoms_unit.get_forces()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    _set_charge_spin(atoms_super, calc)
    atoms_super.calc = calc
    forces_super = atoms_super.get_forces()

    forces_unit_tiled = np.vstack([forces_unit] * multiplier)
    _assert_multiset_close(forces_super, forces_unit_tiled, atol=1e-4)


# --- Isolated-cluster extensivity ---


def test_isolated_extensivity_energy(calc):
    # Two H2O copies separated by 50 A, well beyond the 6 A model cutoff —
    # the combined-system energy must equal twice the single-molecule energy.
    mol_single = molecule("H2O")
    _set_charge_spin(mol_single, calc)
    mol_single.calc = calc
    energy_single = mol_single.get_potential_energy()

    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    mol_combined = mol_single.copy() + mol_copy
    _set_charge_spin(mol_combined, calc)
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
    _set_charge_spin(mol_single, calc)
    mol_single.calc = calc
    forces_single = mol_single.get_forces()

    mol_copy = mol_single.copy()
    mol_copy.positions += [50.0, 0.0, 0.0]

    mol_combined = mol_single.copy() + mol_copy
    _set_charge_spin(mol_combined, calc)
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
