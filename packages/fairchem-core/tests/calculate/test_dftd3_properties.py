"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import torch
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.outputs import all_outputs
from ase.stress import voigt_6_to_full_3x3_stress

from fairchem.core import DFTD3Calculator
from fairchem.core.components.calculate import SinglePointRunner

pytestmark = pytest.mark.serial


@pytest.fixture()
def synthetic_dftd3_parameter_file(tmp_path):
    """
    Create a local parameter table with a nonzero Cu dispersion term.
    """
    parameters = {
        "rcov": torch.zeros(95),
        "r4r2": torch.zeros(95),
        "c6ab": torch.zeros(95, 95, 5, 5),
        "cn_ref": torch.full((95, 95, 5, 5), -1.0),
    }
    parameters["rcov"][29] = 1.0
    parameters["r4r2"][29] = 1.0
    parameters["c6ab"][29, 29, 0, 0] = 10.0
    parameters["cn_ref"][29, :, 0, :] = 0.0
    destination = tmp_path / "dftd3_parameters.pt"
    torch.save(parameters, destination)
    return destination


@pytest.fixture()
def dftd3_atoms(synthetic_dftd3_parameter_file):
    """
    Wrap EMT with the real CPU D3 kernel using only local parameters.
    """
    atoms = bulk("Cu", cubic=True)
    atoms.calc = DFTD3Calculator(
        EMT(),
        functional="r2scan",
        device="cpu",
        cutoff=6.0,
        param_file=synthetic_dftd3_parameter_file,
        auto_download=False,
    )
    return atoms


def test_results_keys_are_ase_outputs(dftd3_atoms):
    dftd3_atoms.get_potential_energy()
    assert set(dftd3_atoms.calc.results) <= all_outputs.keys()


def test_get_properties_roundtrip(dftd3_atoms):
    properties = dftd3_atoms.get_properties(["energy", "forces"])
    calculator = dftd3_atoms.calc
    assert calculator.dispersion_calculator.results["energy"] != 0.0
    for name in ("energy", "free_energy", "forces"):
        assert np.isfinite(properties[name]).all()
        npt.assert_allclose(
            properties[name],
            calculator.base_calculator.results[name]
            + calculator.dispersion_calculator.results[name],
        )


def test_contributions_attribute(dftd3_atoms):
    calculator = dftd3_atoms.calc
    assert calculator.contributions == {}
    dftd3_atoms.get_potential_energy()
    previous_contributions = calculator.contributions
    previous_energy = calculator.results["energy"]
    dftd3_atoms.positions[0] += [0.1, -0.05, 0.02]
    dftd3_atoms.get_potential_energy()

    assert calculator.contributions is not previous_contributions
    assert calculator.results["energy"] != pytest.approx(previous_energy)
    assert set(calculator.contributions) == set(calculator.results)
    for name, pair in calculator.contributions.items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        base, dispersion = pair
        npt.assert_allclose(base, calculator.base_calculator.results[name])
        npt.assert_allclose(dispersion, calculator.dispersion_calculator.results[name])
        npt.assert_allclose(base + dispersion, calculator.results[name])

    # A later calculation without stress must discard its previous contribution.
    dftd3_atoms.pbc = False
    dftd3_atoms.get_potential_energy()
    assert "stress" not in calculator.results
    assert "stress" not in calculator.contributions


class _MatrixStressEMT(EMT):
    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.results["stress"] = voigt_6_to_full_3x3_stress(self.results["stress"])


@pytest.mark.parametrize("matrix_stress", [False, True])
def test_stress_periodic(dftd3_atoms, matrix_stress):
    if matrix_stress:
        dftd3_atoms.calc.base_calculator = _MatrixStressEMT()
    stress = dftd3_atoms.get_properties(["stress"])["stress"]
    assert stress.shape == (6,)
    assert np.isfinite(stress).all()
    base, dispersion = dftd3_atoms.calc.contributions["stress"]
    assert base.shape == dispersion.shape == (6,)
    npt.assert_allclose(stress, base + dispersion)


def test_singlepoint_runner(dftd3_atoms):
    structures = [dftd3_atoms.copy(), dftd3_atoms.copy()]
    structures[1].positions[0] += [0.1, -0.05, 0.02]
    runner = SinglePointRunner(
        calculator=dftd3_atoms.calc,
        input_data=structures,
        calculate_properties=["energy", "forces"],
        save_atoms=False,
    )
    results = runner.calculate()
    assert len(results) == 2
    for atoms, result in zip(structures, results):
        assert result["errors"] == ""
        assert result["traceback"] == ""
        assert np.isfinite(result["energy"])
        assert np.isfinite(result["forces"]).all()
        assert np.shape(result["forces"]) == (len(atoms), 3)
        npt.assert_allclose(result["energy"], atoms.get_potential_energy())
        npt.assert_allclose(result["forces"], atoms.get_forces(), atol=1.0e-10)
