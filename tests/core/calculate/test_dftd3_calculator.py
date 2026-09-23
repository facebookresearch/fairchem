"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the ASE DFT-D3(BJ) calculator wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.testing as npt
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.data.s22 import create_s22_system
from dftd3.ase import DFTD3 as ReferenceDFTD3

from fairchem.core import DFTD3Calculator

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms

pytestmark = pytest.mark.serial


class _ZeroCalculator(Calculator):
    """Base calculator that isolates the physical D3 contribution."""

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.zeros((len(self.atoms), 3)),
            "stress": np.zeros(6),
        }


@pytest.fixture(scope="session")
def dftd3_parameter_file(tmp_path_factory) -> Path:
    """Exercise the public first-use download into a clean temporary cache."""

    destination = tmp_path_factory.mktemp("dftd3") / "parameters.pt"
    assert not destination.exists()
    DFTD3Calculator(
        _ZeroCalculator(),
        functional="pbe",
        device="cpu",
        param_file=destination,
        auto_download=True,
    )
    assert destination.is_file()
    return destination


def _reference_calculator(functional: str) -> ReferenceDFTD3:
    # Loading the functional from dftd3's independent parameter database tests
    # FairChem's damping parameters as well as the kernel outputs. dftd3 leaves
    # ATM disabled by default, matching nvalchemiops' two-body implementation.
    # Its cutoff widths use the same C5 switching window: 3 A is the outer 20%
    # of FairChem's default 15 A cutoff.
    return ReferenceDFTD3(
        method=functional,
        damping="d3bj",
        realspace_cutoff={
            "disp2": 15.0,
            "disp3": 15.0,
            "cn": 15.0,
            "width2": 3.0,
            "width3": 3.0,
        },
    )


def _fairchem_calculator(functional: str, param_file: Path) -> DFTD3Calculator:
    return DFTD3Calculator(
        _ZeroCalculator(),
        functional=functional,
        device="cpu",
        param_file=param_file,
        auto_download=False,
    )


def _s22_structure(name: str, periodic: bool) -> Atoms:
    atoms = create_s22_system(name)
    if periodic:
        # A low-symmetry molecular cell exercises periodic images and all six
        # independent components of the analytic virial/stress conversion.
        atoms.cell = [[14.0, 0.4, 0.1], [0.0, 13.5, 0.5], [0.2, 0.0, 12.8]]
        atoms.center()
        atoms.pbc = True
    return atoms


@pytest.mark.parametrize("functional", ["pbe", "r2scan"])
@pytest.mark.parametrize(
    ("structure_name", "periodic"),
    [
        ("Water_dimer", False),
        ("Benzene_dimer_parallel_displaced", True),
    ],
)
def test_matches_reference_dftd3_on_s22_structures(
    dftd3_parameter_file, functional, structure_name, periodic
):
    atoms = _s22_structure(structure_name, periodic)
    actual_atoms = atoms.copy()
    actual_atoms.calc = _fairchem_calculator(functional, dftd3_parameter_file)
    reference_atoms = atoms.copy()
    reference_atoms.calc = _reference_calculator(functional)

    npt.assert_allclose(
        actual_atoms.get_potential_energy(),
        reference_atoms.get_potential_energy(),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    npt.assert_allclose(
        actual_atoms.get_forces(),
        reference_atoms.get_forces(),
        rtol=2.0e-5,
        atol=1.0e-7,
    )
    if periodic:
        npt.assert_allclose(
            actual_atoms.get_stress(),
            reference_atoms.get_stress(),
            rtol=2.0e-5,
            atol=1.0e-9,
        )


def test_matches_reference_after_position_and_cell_changes(dftd3_parameter_file):
    """Catch stale positions, cells, or periodic-image shifts during NPT."""

    actual_atoms = _s22_structure("Benzene_dimer_parallel_displaced", periodic=True)
    reference_atoms = actual_atoms.copy()
    actual_atoms.calc = _fairchem_calculator("r2scan", dftd3_parameter_file)
    reference_atoms.calc = _reference_calculator("r2scan")

    for displacement, cell_scale in [
        (np.zeros(3), 1.0),
        (np.array([0.03, -0.02, 0.01]), 1.0),
        (np.zeros(3), 1.015),
    ]:
        actual_atoms.positions[0] += displacement
        reference_atoms.positions[0] += displacement
        actual_atoms.set_cell(actual_atoms.cell * cell_scale, scale_atoms=True)
        reference_atoms.set_cell(reference_atoms.cell * cell_scale, scale_atoms=True)

        npt.assert_allclose(
            actual_atoms.get_potential_energy(),
            reference_atoms.get_potential_energy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        npt.assert_allclose(
            actual_atoms.get_forces(),
            reference_atoms.get_forces(),
            rtol=2.0e-5,
            atol=1.0e-7,
        )
        npt.assert_allclose(
            actual_atoms.get_stress(),
            reference_atoms.get_stress(),
            rtol=2.0e-5,
            atol=1.0e-9,
        )
