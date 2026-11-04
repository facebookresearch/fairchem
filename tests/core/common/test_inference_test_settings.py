"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit.api.inference import (
    NAME_TO_INFERENCE_SETTING,
    InferenceSettings,
)
from tests import conftest as test_config


@pytest.mark.parametrize("name", list(NAME_TO_INFERENCE_SETTING))
def test_eager_settings_preserve_precision_and_model_options(name, monkeypatch):
    preset = NAME_TO_INFERENCE_SETTING[name]
    original = asdict(preset)
    monkeypatch.setitem(test_config._FIXTURE_SETTINGS, "eager", True)

    eager = test_config.inference_settings_for_test(name)

    assert asdict(eager) == {**original, "compile": False, "nitro": False}
    assert asdict(preset) == original
    assert eager is not preset


def test_unrequested_eager_policy_preserves_interactive_defaults(monkeypatch):
    monkeypatch.setitem(test_config._FIXTURE_SETTINGS, "eager", False)
    monkeypatch.setitem(
        test_config._FIXTURE_SETTINGS, "inference_settings", "high-precision"
    )
    assert test_config.fixture_inference_settings() == "high-precision"
    settings = InferenceSettings(compile=True, nitro=True, tf32=True)
    assert test_config.inference_settings_for_test(settings) is settings


def test_eager_settings_keep_requested_derivatives_and_threads(monkeypatch):
    monkeypatch.setitem(test_config._FIXTURE_SETTINGS, "eager", True)
    settings = InferenceSettings(
        compile=True,
        nitro=True,
        base_precision_dtype=torch.float64,
        execution_mode="general",
        torch_num_threads=1,
        predict_untrained_hessian={"omol"},
        hessian_vmap=False,
    )
    original = asdict(settings)
    eager = test_config.inference_settings_for_test(settings)
    assert asdict(eager) == {**original, "compile": False, "nitro": False}
    assert asdict(settings) == original


def test_shared_predictor_receives_eager_settings_without_changing_explicit_mode(
    monkeypatch,
):
    monkeypatch.setitem(test_config._FIXTURE_SETTINGS, "eager", True)
    monkeypatch.setitem(
        test_config._FIXTURE_SETTINGS, "inference_settings", "high-precision"
    )
    loads = []

    def record_load(name, **kwargs):
        loads.append((name, kwargs))
        return kwargs["inference_settings"]

    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", record_load)
    name = pretrained_mlip.available_models[0]
    eager = test_config.get_predict_unit_for_test(name)
    assert eager.compile is False
    assert eager.nitro is False
    assert eager.tf32 is False
    assert eager.merge_mole is True
    explicit = InferenceSettings(compile=True)
    assert (
        test_config.get_predict_unit_for_test(name, inference_settings=explicit)
        is explicit
    )
    assert len(loads) == 2


def test_eager_guard_rejects_compilation_and_restores_torch(monkeypatch):
    # Replace the original to verify restoration even when pytest already has
    # its session guard installed via --eager-inference.
    original = object()
    monkeypatch.setattr(torch, "compile", original)
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda *a, **kw: True))
    guard = test_config._guard_eager_inference_compilation.__wrapped__(request)
    next(guard)
    try:
        with pytest.raises(AssertionError, match="torch.compile was called"):
            torch.compile(torch.nn.Identity())
    finally:
        with pytest.raises(StopIteration):
            next(guard)
    assert torch.compile is original


@pytest.mark.gpu()
def test_eager_prediction_loads_real_checkpoint_without_compilation(
    tmp_path, monkeypatch
):
    from ase.build import molecule

    from tests.core.models.uma.uma_nitro.nitro_test_utils import (
        batch,
        random_checkpoint,
    )

    checkpoint = tmp_path / "random-eager.pt"
    torch.save(random_checkpoint(num_layers=1), checkpoint)
    monkeypatch.setitem(test_config._FIXTURE_SETTINGS, "eager", True)
    monkeypatch.setitem(
        test_config._FIXTURE_SETTINGS, "inference_settings", "high-precision"
    )

    def reject_compile(*args, **kwargs):
        raise AssertionError("Eager prediction unexpectedly called torch.compile")

    monkeypatch.setattr(torch, "compile", reject_compile)
    predictor = test_config.get_predict_unit_for_test(str(checkpoint), device="cuda")
    assert predictor.inference_settings.compile is False
    assert predictor.inference_settings.nitro is False
    assert predictor.inference_settings.tf32 is False
    assert predictor.inference_settings.merge_mole is True
    result = predictor.predict(batch((molecule("H2O"), "omol", 0, 1)))
    assert torch.isfinite(result["energy"]).all()
    assert torch.isfinite(result["forces"]).all()
    assert result["forces"].shape == (3, 3)
