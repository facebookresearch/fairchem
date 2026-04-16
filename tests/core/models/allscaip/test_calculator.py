"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from ase.build import molecule

from fairchem.core import FAIRChemCalculator
from fairchem.core.models.allscaip.AllScAIP import AllScAIPBackbone
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "allscaip_checkpoint.pt"
)

# mark all tests in this module as gpu tests
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def allscaip_predict_unit():
    return MLIPPredictUnit(CHECKPOINT_PATH, device="cuda")


def test_calculator_inference(allscaip_predict_unit):
    """
    Test that AllScaip works end-to-end through FAIRChemCalculator.
    """
    calc = FAIRChemCalculator(allscaip_predict_unit, task_name="omol")

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (len(atoms), 3)


def test_calculator_inference_with_max_atoms():
    """
    Test that max_atoms in InferenceSettings pads inputs correctly
    and produces valid results through FAIRChemCalculator with torch.compile.
    """
    torch.compiler.reset()

    inference_settings = InferenceSettings(compile=True, max_atoms=30)
    predict_unit = MLIPPredictUnit(
        CHECKPOINT_PATH, device="cuda", inference_settings=inference_settings
    )
    calc = FAIRChemCalculator(predict_unit, task_name="omol")

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    # Forces should match the actual number of atoms, not max_atoms
    assert forces.shape == (len(atoms), 3)


def test_calculator_inference_max_atoms_required_with_compile():
    """
    Test that compile=True without max_atoms raises an error.
    """
    with pytest.raises(ValueError, match="max_atoms must be set"):
        AllScAIPBackbone.build_inference_settings(
            InferenceSettings(compile=True, max_atoms=None)
        )
