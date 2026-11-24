"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  MLIPPredictUnit.refresh_weights loads another checkpoint's weights in
        place: before and after the first prediction, with the general,
        MOLE-merged, compiled, fast-GPU and UMA Nitro models, and rejects a
        checkpoint of a different model without changing the loaded one.
Models: Randomly initialized UMA-S checkpoints; no downloads.
CI:     CPU tests in the units shard; GPU tests in the GPU shard.
"""

from __future__ import annotations

import copy

import pytest
import torch
from ase.build import bulk, molecule

from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit
from tests.models.uma.uma_nitro.nitro_test_utils import (
    batch,
    nitro_settings,
    random_checkpoint,
    requires_nitro_gpu,
)

SYSTEMS = {
    "periodic": (bulk("Cu", "fcc", a=3.6, cubic=True), "omat"),
    "molecule": (molecule("H2O"), "omol", 0, 1),
}


def _perturbed(checkpoint, seed: int):
    """
    The checkpoint with every weight perturbed and an energy normalizer changed.
    """
    generator = torch.Generator().manual_seed(seed)

    def perturb(state):
        return {
            key: (
                value + 0.05 * torch.randn(value.shape, generator=generator)
                if value.is_floating_point()
                else value
            )
            for key, value in state.items()
        }

    other = copy.deepcopy(checkpoint)
    other.ema_state_dict = perturb(checkpoint.ema_state_dict)
    other.model_state_dict = perturb(checkpoint.model_state_dict)
    other.tasks_config[0]["normalizer"]["rmsd"] = 2.0
    return other


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """
    Paths of two checkpoints of one UMA-S model and one of a smaller model.
    """
    root = tmp_path_factory.mktemp("refresh_weights")
    first = random_checkpoint()
    paths = {
        "first": root / "first.pt",
        "second": root / "second.pt",
        "other_model": root / "other_model.pt",
    }
    torch.save(first, paths["first"])
    torch.save(_perturbed(first, seed=1), paths["second"])
    torch.save(random_checkpoint(num_layers=2), paths["other_model"])
    return {name: str(path) for name, path in paths.items()}


def _predict(predictor, system):
    return {
        key: value.detach().cpu()
        for key, value in predictor.predict(batch(system)).items()
    }


def _assert_close(output, expected, rtol=1e-5, atol=1e-6):
    """
    Agreement up to float32 run-to-run variation (CPU reductions are not
    bitwise reproducible across calls, even on one predict unit).
    """
    assert output.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(output[key], expected[key], rtol=rtol, atol=atol)


def _assert_differs(output, reference):
    assert any(not torch.allclose(output[key], reference[key]) for key in reference)


def _check_refresh(checkpoints, device, settings, system, **tolerances):
    predictor = MLIPPredictUnit(
        checkpoints["first"], device, inference_settings=copy.deepcopy(settings)
    )
    before = _predict(predictor, system)
    model = predictor.model

    predictor.refresh_weights(checkpoints["second"])
    after = _predict(predictor, system)

    expected = _predict(
        MLIPPredictUnit(
            checkpoints["second"], device, inference_settings=copy.deepcopy(settings)
        ),
        system,
    )
    assert predictor.model is model, "refresh rebuilt the prepared model"
    _assert_close(after, expected, **tolerances)
    _assert_differs(after, before)
    assert predictor.tasks["omat_energy"].normalizer.rmsd.item() == 2.0


GENERAL = {"merge_mole": False, "compile": False, "execution_mode": "general"}
MERGED = {"merge_mole": True, "compile": False, "execution_mode": "general"}


@pytest.mark.parametrize("system", SYSTEMS.values(), ids=SYSTEMS.keys())
@pytest.mark.parametrize("settings", [GENERAL, MERGED], ids=["general", "merged"])
def test_refresh_matches_a_new_predict_unit(checkpoints, settings, system):
    _check_refresh(checkpoints, "cpu", InferenceSettings(**settings), system)


def test_refresh_reuses_the_compiled_model(checkpoints):
    settings = InferenceSettings(**{**MERGED, "compile": True})
    _check_refresh(
        checkpoints, "cpu", settings, SYSTEMS["periodic"], rtol=1e-4, atol=1e-4
    )


def test_refresh_before_the_first_prediction_loads_the_checkpoint(checkpoints):
    settings = InferenceSettings(**MERGED)
    predictor = MLIPPredictUnit(
        checkpoints["first"], "cpu", inference_settings=settings
    )
    predictor.refresh_weights(checkpoints["second"])

    expected = MLIPPredictUnit(
        checkpoints["second"], "cpu", inference_settings=InferenceSettings(**MERGED)
    )
    _assert_close(
        _predict(predictor, SYSTEMS["periodic"]),
        _predict(expected, SYSTEMS["periodic"]),
    )


def test_refresh_rejects_a_different_model(checkpoints):
    predictor = MLIPPredictUnit(
        checkpoints["first"], "cpu", inference_settings=InferenceSettings(**MERGED)
    )
    reference = _predict(predictor, SYSTEMS["periodic"])

    with pytest.raises(ValueError, match="does not match the loaded model"):
        predictor.refresh_weights(checkpoints["other_model"])
    _assert_close(_predict(predictor, SYSTEMS["periodic"]), reference)
    assert predictor._inference_model_path == checkpoints["first"]


@pytest.mark.gpu()
@pytest.mark.parametrize("system", SYSTEMS.values(), ids=SYSTEMS.keys())
def test_refresh_fast_gpu_compiled(checkpoints, system):
    settings = InferenceSettings(
        merge_mole=True,
        compile=True,
        execution_mode="umas_fast_gpu",
        activation_checkpointing=False,
        tf32=False,
    )
    _check_refresh(checkpoints, "cuda", settings, system, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu()
@requires_nitro_gpu
@pytest.mark.parametrize("system", SYSTEMS.values(), ids=SYSTEMS.keys())
def test_refresh_nitro(checkpoints, system, monkeypatch):
    from fairchem.core.models.uma.uma_nitro import _inference

    # Random checkpoints are not on Nitro's allowlist of UMA-S 1.2.1 files.
    monkeypatch.setattr(_inference, "verify_checkpoint_file", lambda path: None)
    predictor = MLIPPredictUnit(
        checkpoints["first"], "cuda", inference_settings=nitro_settings()
    )
    _predict(predictor, system)
    assert predictor.inference_settings.nitro, "Nitro fell back"
    # A second call replays the captured graph against the old weights.
    _predict(predictor, system)

    predictor.refresh_weights(checkpoints["second"])
    after = _predict(predictor, system)

    expected = _predict(
        MLIPPredictUnit(
            checkpoints["second"], "cuda", inference_settings=nitro_settings()
        ),
        system,
    )
    assert predictor.inference_settings.nitro
    _assert_close(after, expected, rtol=1e-3, atol=1e-3)
