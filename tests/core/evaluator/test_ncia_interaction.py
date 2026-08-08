"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests: NCIA two-body interaction-energy evaluator math. Pure numpy/ase/pymatgen
       (no model), so these run in normal CI. Covers the monomer-subtraction
       machinery (interaction_energy_and_forces), the ncia_interaction_energy MAE,
       and the exact-partition invariant guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from pymatgen.io.ase import MSONAtoms

from fairchem.data.omol.modules.evaluator import (
    interaction_energy_and_forces,
    ncia_interaction_energy,
)


def _payload(symbols, positions, energy, forces, charge=0, spin=1):
    """Build one component result dict (MSONAtoms + energy + forces)."""
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info.update({"charge": charge, "spin": spin})
    return {
        "atoms": MSONAtoms(atoms).as_dict(),
        "energy": energy,
        "forces": forces,
    }


# A tiny 3-atom "dimer": He at origin, two H stacked along z.
# monomer_a = He (dimer index 0); monomer_b = the two H (dimer indices 1, 2).
_DIMER_POS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
_DIMER_FORCES = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
_MONO_A_FORCES = [[0.0, 0.0, 0.5]]
_MONO_B_FORCES = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]


def _mlip_results(mono_b_shift=0.0):
    """MLIP-style per-component results for a single identifier.

    mono_b_shift perturbs monomer_b's positions so they no longer match the
    dimer geometry (used to exercise the partition-invariant guard).
    """
    mono_b_pos = [[0.0, 0.0, 1.0 + mono_b_shift], [0.0, 0.0, 2.0 + mono_b_shift]]
    return {
        "id": {
            "dimer": _payload("HeHH", _DIMER_POS, -10.0, _DIMER_FORCES),
            "monomer_a": _payload("He", [[0.0, 0.0, 0.0]], -3.0, _MONO_A_FORCES),
            "monomer_b": _payload("HH", mono_b_pos, -6.0, _MONO_B_FORCES),
        }
    }


def test_interaction_energy_and_forces():
    """E_int = E_dimer - E_a - E_b, forces mapped into the dimer indices."""
    ixn_energy, ixn_forces = interaction_energy_and_forces(_mlip_results(), "dimer")

    # -10 - (-3) - (-6) = -1.0
    assert ixn_energy["id"] == pytest.approx(-1.0)

    # dimer forces minus each monomer force at its mapped dimer index:
    # He -> idx0, H@z=1 -> idx1, H@z=2 -> idx2
    expected = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])
    np.testing.assert_allclose(ixn_forces["id"], expected)


def test_ncia_interaction_energy_mae():
    """MAE is |reference E_int - MLIP E_int|, averaged over systems."""
    orca_results = {"id": {"interaction_energy": -0.5}}
    metrics = ncia_interaction_energy(
        orca_results, _mlip_results(), supersystem="dimer"
    )

    # MLIP E_int = -1.0, reference = -0.5 -> |(-0.5) - (-1.0)| = 0.5
    assert metrics["interaction_energy_mae"] == pytest.approx(0.5)
    assert metrics["n_systems"] == 1


def test_partition_invariant_guard():
    """A monomer atom that isn't an exact match in the dimer raises ValueError."""
    with pytest.raises(ValueError):
        interaction_energy_and_forces(_mlip_results(mono_b_shift=1e-3), "dimer")
