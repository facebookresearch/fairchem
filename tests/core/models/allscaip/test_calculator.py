"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from ase.build import molecule

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MLIPInferenceCheckpoint,
)
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

DATASET_LIST = ["omol"]
BACKBONE_CONFIG = {
    "model": "fairchem.core.models.allscaip.AllScAIP.AllScAIPBackbone",
    "regress_stress": True,
    "direct_forces": False,
    "regress_forces": True,
    "hidden_size": 8,
    "dataset_list": DATASET_LIST,
    "use_compile": False,
    "use_padding": False,
    "max_num_elements": 100,
    "max_atoms": 30,
    "max_batch_size": 8,
    "max_radius": 6.0,
    "knn_k": 20,
    "knn_pad_size": 30,
    "num_layers": 2,
    "atten_name": "memory_efficient",
    "atten_num_heads": 2,
    "freequency_list": [2, 2],
    "use_freq_mask": True,
    "use_sincx_mask": True,
}
HEADS_CONFIG = {
    "energyandforcehead": {
        "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
        "head_cls": "fairchem.core.models.allscaip.AllScAIP.AllScAIPGradientEnergyForceStressHead",
        "head_kwargs": {"wrap_property": False},
        "dataset_names": DATASET_LIST,
    },
}
MODEL_CONFIG = {
    "_target_": "fairchem.core.models.base.HydraModel",
    "backbone": BACKBONE_CONFIG,
    "heads": HEADS_CONFIG,
    "pass_through_head_outputs": True,
}
TASKS_CONFIG = [
    {
        "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
        "name": "omol_energy",
        "level": "system",
        "property": "energy",
        "out_spec": {"dim": [1], "dtype": "float32"},
        "normalizer": {
            "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
            "mean": 0.0,
            "rmsd": 1.0,
        },
        "datasets": ["omol"],
    },
    {
        "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
        "name": "omol_forces",
        "level": "atom",
        "property": "forces",
        "out_spec": {"dim": [3], "dtype": "float32"},
        "normalizer": {
            "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
            "mean": 0.0,
            "rmsd": 1.0,
        },
        "datasets": ["omol"],
    },
    {
        "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
        "name": "omol_stress",
        "level": "system",
        "property": "stress",
        "out_spec": {"dim": [9], "dtype": "float32"},
        "normalizer": {
            "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
            "mean": 0.0,
            "rmsd": 1.0,
        },
        "datasets": ["omol"],
    },
]


def _create_allscaip_checkpoint(path, model_config=None):
    """
    Create a minimal AllScaip inference checkpoint for testing.
    """
    import hydra

    if model_config is None:
        model_config = MODEL_CONFIG

    model = hydra.utils.instantiate(model_config)
    state_dict = model.state_dict()

    # Create a minimal EMA state dict (copy of model state dict + n_averaged)
    ema_state_dict = {k: v.clone() for k, v in state_dict.items()}
    ema_state_dict["n_averaged"] = torch.tensor(1, dtype=torch.long)

    checkpoint = MLIPInferenceCheckpoint(
        model_config=model_config,
        model_state_dict=state_dict,
        ema_state_dict=ema_state_dict,
        tasks_config=TASKS_CONFIG,
    )
    torch.save(checkpoint, path)


@pytest.fixture(scope="module")
def allscaip_checkpoint(tmp_path_factory):
    """
    Create a temporary AllScaip checkpoint for calculator tests.
    """
    path = str(tmp_path_factory.mktemp("allscaip") / "allscaip_test.pt")
    _create_allscaip_checkpoint(path)
    return path


# mark all tests in this module as gpu tests
pytestmark = pytest.mark.gpu


def test_calculator_inference(allscaip_checkpoint):
    """
    Test that AllScaip works end-to-end through FAIRChemCalculator.
    """
    predict_unit = MLIPPredictUnit(
        allscaip_checkpoint,
        device="cuda",
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
    assert forces.shape == (len(atoms), 3)


def test_calculator_inference_with_max_atoms(allscaip_checkpoint):
    """
    Test that max_atoms in InferenceSettings pads inputs correctly
    and produces valid results through FAIRChemCalculator.
    """
    torch.compiler.reset()

    max_atoms = 30
    inference_settings = InferenceSettings(compile=True, max_atoms=max_atoms)
    predict_unit = MLIPPredictUnit(
        allscaip_checkpoint,
        device="cuda",
        inference_settings=inference_settings,
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


def test_calculator_inference_max_atoms_exceeded():
    """
    Test that compile=True without max_atoms raises an error.
    """
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _create_allscaip_checkpoint(f.name)
        with pytest.raises(ValueError, match="max_atoms must be set"):
            MLIPPredictUnit(
                f.name,
                device="cuda",
                inference_settings=InferenceSettings(compile=True, max_atoms=None),
            )
