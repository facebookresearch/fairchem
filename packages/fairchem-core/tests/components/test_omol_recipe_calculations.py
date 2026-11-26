"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

from fairchem.core.components.calculate.recipes import omol


class HarmonicCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(
                np.square(atoms.positions).sum()
                + atoms.info.get("charge", 0) ** 2
                + 0.1 * atoms.info.get("spin", 0)
            ),
            "forces": -2 * atoms.positions,
        }


def _atoms(charge=0, spin=0):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.2]])
    atoms.info.update(charge=charge, spin=spin)
    return atoms


@pytest.fixture()
def optimization(monkeypatch):
    flags = {
        "optimizer": FIRE,
        "optimizer_kwargs": {"logfile": None},
        "fmax": 0.0,
        "max_steps": 1,
    }
    monkeypatch.setattr(omol, "EVAL_OPT_PARAMETERS", flags)
    return flags


def test_single_point_clears_cached_charge_and_spin_results():
    atoms = _atoms()
    calculator = HarmonicCalculator()
    for charge in (0, 1, 0):
        atoms.info["charge"] = charge
        result = omol.single_point_job(atoms, calculator)
        assert result["energy"] == pytest.approx(0.04 + charge**2)
        assert np.asarray(result["forces"]) == pytest.approx(-2 * atoms.positions)
        assert atoms.calc is None


def test_relaxation_records_initial_final_and_last_valid_failed_state(optimization):
    calculator = HarmonicCalculator()
    result = omol.relax_job(_atoms(), calculator, optimization)
    assert result["initial"]["energy"] == pytest.approx(0.04)
    assert result["final"]["energy"] < result["initial"]["energy"]
    assert np.asarray(result["final"]["forces"]).shape == (2, 3)
    failed = omol.relax_job(
        _atoms(),
        calculator,
        {**optimization, "optimizer": Mock(side_effect=RuntimeError("failed"))},
    )
    assert failed["initial"]["energy"] == failed["final"]["energy"]


def test_charge_spin_and_complex_recipes_preserve_result_keys():
    calculator = HarmonicCalculator()
    result = omol.ieea({"a": {1: {2: {"atoms": _atoms(1, 2)}}}}, calculator)
    assert result["a"][1][2]["energy"] == pytest.approx(1.24)
    spins = omol.spin_gap({"a": {2: {"atoms": _atoms(spin=2)}}}, calculator)
    assert spins["a"][2]["energy"] == pytest.approx(0.24)
    complex_result = omol.ligand_pocket(
        {"a": {key: _atoms() for key in ("ligand", "pocket", "ligand_pocket")}},
        calculator,
    )
    assert set(complex_result["a"]) == {"ligand", "pocket", "ligand_pocket"}
    distances = omol.distance_scaling(
        {"vertical": {"a": {"short": _atoms(), "long": _atoms()}}}, calculator
    )
    assert set(distances["vertical"]["a"]) == {"short", "long"}
    assert set(omol.singlepoint({"a": _atoms()}, calculator)) == {"a"}


def test_conformer_protonation_and_ligand_relaxation_routes(optimization):
    calculator = HarmonicCalculator()
    conformers = omol.conformers(
        {"family": [{"sid": "a", "initial_atoms": _atoms()}]}, calculator
    )
    assert conformers["family"]["a"]["final"]["energy"] < 0.04
    protonation = omol.protonation(
        {"family": {"neutral": {"initial_atoms": _atoms()}}}, calculator
    )
    assert set(protonation["family"]) == {"neutral"}
    ligands = omol.ligand_strain(
        {"a": {"bioactive_conf": _atoms(), "conformers": [_atoms()]}}, calculator
    )
    assert set(ligands["a"]) == {"bioactive", "gas_phase"}
    assert set(ligands["a"]["gas_phase"]) == {0}
