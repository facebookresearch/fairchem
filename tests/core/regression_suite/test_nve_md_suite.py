"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Regression-suite measurements hold NumPy arrays, not pandas objects.
# ruff: noqa: PD011
import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from fairchem.core.regression_suite.spec.transforms import apply_case_transform
from fairchem.core.regression_suite.suites import load_suite
from fairchem.core.regression_suite.suites.nve_md import suite as nve_module

_DRIFT_STEP_ANGSTROM = 1.0e-5


class _SlopeCalculator(Calculator):
    """
    Energy equal to the sum of the atoms' x coordinates, in eV.
    """

    implemented_properties = ("energy", "forces", "stress")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(np.sum(self.atoms.positions[:, 0])),
            "forces": np.zeros((len(self.atoms), 3)),
            "stress": np.zeros(6),
        }


class _DriftingVerlet:
    """
    Translate every atom along x by a fixed step, so the energy of
    _SlopeCalculator climbs linearly in time.
    """

    def __init__(self, atoms, timestep):
        self.atoms = atoms
        self.steps = 0
        self.observers = []

    def attach(self, callback, interval):
        self.observers.append((callback, interval))

    def get_number_of_steps(self):
        return self.steps

    def run(self, steps):
        start = self.atoms.positions.copy()
        for step in range(steps + 1):
            self.steps = step
            self.atoms.positions = start + [_DRIFT_STEP_ANGSTROM * step, 0.0, 0.0]
            for callback, interval in self.observers:
                if step % interval == 0:
                    callback()


@pytest.mark.parametrize("system", ["DHA", "Fe"])
def test_linear_energy_ramp_reports_its_exact_drift(monkeypatch, system):
    suite = load_suite("nve_md")
    spec = next(
        case
        for case in suite.plan().cases
        if case.case_id.endswith(f"/{system}/seed-0")
    )
    case = apply_case_transform(suite.source_atoms(spec), spec)

    def zero_momenta(atoms, **_options):
        atoms.set_momenta(np.zeros((len(atoms), 3)))

    monkeypatch.setattr(nve_module, "initialize_momenta", zero_momenta)
    monkeypatch.setattr(nve_module, "VelocityVerlet", _DriftingVerlet)
    measurements = suite.evaluate(_SlopeCalculator(), case)

    ev_per_atom_per_ps = _DRIFT_STEP_ANGSTROM * 1000.0 / spec.metadata["timestep_fs"]
    assert measurements["energy_drift"] == pytest.approx(1000.0 * ev_per_atom_per_ps)


def test_gate_curves_measure_gap_movement_and_spread():
    # Three seeds whose energy climbs at rates 0, 1, and 2 per frame, each a
    # constant 0.3 above its FP64 reference.
    rates = np.array([0.0, 1.0, 2.0])
    run = rates[:, None] * np.arange(40.0)[None, :]

    curves = nve_module.gate_curves(run, run - 0.3, window=5)

    frames = np.arange(36.0)
    assert curves["fp64_gap"] == pytest.approx(np.full(36, 0.3))
    assert curves["drift"] == pytest.approx(frames)
    assert curves["spread"] == pytest.approx(np.var(rates) * frames**2)
