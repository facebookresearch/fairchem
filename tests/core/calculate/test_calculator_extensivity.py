"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Extensivity of UMA-S forces and energies — E(N replicas) =
        N * E(1 replica) and the equivalent force-tiling check, on
        both PBC supercells (rocksalt MgO) and isolated clusters
        (H2O at >50 A separation). Parametrized over fp32/fp64
        (numerical-floor check) and 6 task heads (oc20, omat, omol,
        odac, omc, oc25). Structures are rotated by a fixed seeded
        SO(3) matrix to break axis-aligned-edge degeneracies in the
        eSCN spherical-harmonic basis.
Models: uma-s-1p1, uma-s-1p2 (module-level pytestmark). uma-s-1p2 is
        xfail(strict=False) here via _xfail_uma_s_1p2_extensivity_bug
        for a known regression.
CI:     test_gpu_sweep (models shard).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import torch
from ase.build import bulk, make_supercell, molecule
from scipy.spatial.transform import Rotation

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import InferenceSettings
from tests.conftest import get_predict_unit_for_test, sweep_refs_from

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


# Run every extensivity check in both fp32 and fp64. fp32 is what production
# inference uses; fp64 is a precision-floor check that rules out summation-
# order float drift when the test fails. Pattern mirrors
# tests/core/units/mlip_unit/test_extensivity.py and test_equivariance.py.
DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float64, id="fp64"),
]

# Per-dtype tolerances. fp64 is tight to the float-precision floor
# (empirical max_abs on 4x4x4 supercell ≈ 7e-13). fp32 tolerances are set
# above the largest legitimate float-noise observed on rotated MgO
# supercells across all UMA-S task heads, so that real extensivity
# violations are still caught.
PBC_ENERGY_RTOL = {torch.float32: 1e-4, torch.float64: 1e-10}
PBC_FORCES_ATOL = {torch.float32: 5e-4, torch.float64: 1e-10}
ISOLATED_ENERGY_ATOL = {torch.float32: 1e-4, torch.float64: 1e-10}
ISOLATED_FORCES_ATOL = {torch.float32: 5e-4, torch.float64: 1e-10}


# Apply a fixed random rotation to every structure before evaluation.
# Axis-aligned edges (parallel to a Cartesian axis) hit exactly 0 or ±1
# in many eSCN spherical-harmonic channels, giving the model a degenerate
# sample of its basis. Rocksalt MgO's primitive cell sits in such a
# degenerate orientation by default — its Mg-O bonds are along
# [100]-family directions. A generic SO(3) rotation moves the structure
# to a non-degenerate orientation that exercises every basis channel.
#
# UMA-S is SO(3) equivariant by construction (eSCN + Wigner-D), so the
# rotation must not change passing tests. The seed is fixed so the
# perturbation is a stable, reproducible orientation rather than a noise
# source; bump it and re-baseline if a future model regression depends
# on the specific rotation chosen.
_ROTATION_SEED = 42
_ROTATION_MATRIX = Rotation.random(
    random_state=np.random.default_rng(_ROTATION_SEED)
).as_matrix()


def _rotate(atoms):
    """
    Apply the module's deterministic random rotation to ``atoms`` in
    place, returning the same object for chaining.

    For periodic structures both the cell and atomic positions rotate
    together (``scale_atoms=True``). For non-periodic structures only
    positions rotate. The rotation preserves all distances and angles,
    so chemistry, neighbor lists, and extensivity are unchanged.
    """
    if atoms.cell.any():
        atoms.set_cell(atoms.cell @ _ROTATION_MATRIX.T, scale_atoms=True)
    else:
        atoms.positions = atoms.positions @ _ROTATION_MATRIX.T
    return atoms


# (pretrained_checkpoint, dtype) -> MLIPPredictUnit cache. Loading a UMA
# checkpoint takes ~5s on GPU; with TASK_NAMES * SUPERCELL_CONFIGS * DTYPES
# we hit each predictor for ~70 tests, so a per-key cache turns a 70-load
# loop into a single load per (checkpoint, dtype) pair.
_PREDICTOR_CACHE: dict = {}


def _predictor_for(pretrained_checkpoint, dtype, refs_from):
    key = (pretrained_checkpoint, dtype)
    if key not in _PREDICTOR_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        settings = InferenceSettings(
            activation_checkpointing=False, base_precision_dtype=dtype
        )
        _PREDICTOR_CACHE[key] = get_predict_unit_for_test(
            pretrained_checkpoint,
            device=device,
            refs_from=refs_from,
            inference_settings=settings,
        )
    return _PREDICTOR_CACHE[key]


@pytest.fixture(params=TASK_NAMES)
def calc(request, pretrained_checkpoint, dtype):
    """
    FAIRChemCalculator wrapping a (checkpoint, dtype)-cached predictor
    for the parametrized task. Skips tasks not supported by the
    checkpoint (e.g. uma-s-1p1 lacking oc25).
    """
    task_name = request.param
    pu = _predictor_for(pretrained_checkpoint, dtype, sweep_refs_from(request.config))
    if task_name not in pu.dataset_to_tasks:
        pytest.skip(f"task {task_name!r} not supported by current pretrained model")
    return FAIRChemCalculator(pu, task_name=task_name)


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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("supercell_matrix, multiplier", SUPERCELL_CONFIGS)
def test_pbc_extensivity_energy(supercell_matrix, multiplier, calc, dtype):
    atoms_unit = _rotate(bulk("MgO", "rocksalt", a=4.213))
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
        rtol=PBC_ENERGY_RTOL[dtype],
        err_msg=(
            f"Energy does not scale by {multiplier}x for supercell. "
            f"E_unit={energy_unit:.6f}, E_super={energy_super:.6f}"
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("supercell_matrix, multiplier", SUPERCELL_CONFIGS)
def test_pbc_extensivity_forces(supercell_matrix, multiplier, calc, dtype):
    atoms_unit = _rotate(bulk("MgO", "rocksalt", a=4.213))
    _set_charge_spin(atoms_unit, calc)
    atoms_unit.calc = calc
    forces_unit = atoms_unit.get_forces()

    atoms_super = make_supercell(atoms_unit, supercell_matrix)
    _set_charge_spin(atoms_super, calc)
    atoms_super.calc = calc
    forces_super = atoms_super.get_forces()

    forces_unit_tiled = np.vstack([forces_unit] * multiplier)
    _assert_multiset_close(forces_super, forces_unit_tiled, atol=PBC_FORCES_ATOL[dtype])


# --- Isolated-cluster extensivity ---


@pytest.mark.parametrize("dtype", DTYPES)
def test_isolated_extensivity_energy(calc, dtype):
    # Two H2O copies separated by 50 A, well beyond the 6 A model cutoff —
    # the combined-system energy must equal twice the single-molecule energy.
    mol_single = _rotate(molecule("H2O"))
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
        atol=ISOLATED_ENERGY_ATOL[dtype],
        err_msg=(
            f"Energy is not extensive for two separated molecules. "
            f"E_single={energy_single:.6f}, E_combined={energy_combined:.6f}"
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_isolated_extensivity_forces(calc, dtype):
    mol_single = _rotate(molecule("H2O"))
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
        atol=ISOLATED_FORCES_ATOL[dtype],
        err_msg="Forces on first molecule don't match isolated molecule.",
    )
    npt.assert_allclose(
        forces_combined[n:],
        forces_single,
        atol=ISOLATED_FORCES_ATOL[dtype],
        err_msg="Forces on second molecule don't match isolated molecule.",
    )
