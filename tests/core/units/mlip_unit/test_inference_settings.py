"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    guess_inference_settings,
    inference_settings_batch,
    inference_settings_default,
    inference_settings_high_precision,
    inference_settings_multigpu,
    inference_settings_nitro,
    inference_settings_traineval,
)
from fairchem.core.units.mlip_unit.predict import (
    _REDUCED_PRECISION_WARNING,
    _describe_settings_change,
    _uses_reduced_precision,
)

# --- __post_init__ ---


def test_default_dtype_is_float32():
    settings = InferenceSettings()
    assert settings.base_precision_dtype is torch.float32


def test_named_modes_use_expected_paths():
    high_precision = inference_settings_high_precision()
    batch = inference_settings_batch()
    multigpu = inference_settings_multigpu()

    assert high_precision.merge_mole is True
    assert high_precision.compile is True
    assert high_precision.tf32 is False
    assert high_precision.activation_checkpointing is False
    assert batch.merge_mole is False
    assert batch.compile is False
    assert batch.tf32 is False
    assert batch.activation_checkpointing is True
    assert multigpu.merge_mole is True
    assert multigpu.compile is True
    assert multigpu.tf32 is True
    assert multigpu.activation_checkpointing is False


def test_batch_is_a_named_inference_mode():
    assert guess_inference_settings("batch") == inference_settings_batch()


def test_turbo_is_no_longer_available_and_points_to_nitro():
    with pytest.raises(ValueError, match="no longer available. Use the default"):
        guess_inference_settings("turbo")


def test_default_mode_is_nitro():
    assert inference_settings_default() == inference_settings_nitro()
    assert guess_inference_settings("default") == inference_settings_nitro()


def test_multigpu_is_the_reduced_precision_tf32_counterpart_of_high_precision():
    """
    The two presets differ only in tf32: multigpu trades precision for speed.
    """
    import dataclasses

    high_precision = guess_inference_settings("high-precision")
    assert high_precision == inference_settings_high_precision()
    assert dataclasses.replace(inference_settings_multigpu(), tf32=False) == (
        high_precision
    )


def test_nitro_mode_is_multigpu_plus_nitro():
    """
    Turning nitro off, as every Nitro fallback does, gives exactly multigpu.
    """
    import dataclasses

    nitro = inference_settings_nitro()

    assert nitro.nitro is True
    assert dataclasses.replace(nitro, nitro=False) == inference_settings_multigpu()
    assert guess_inference_settings("nitro").nitro is True
    assert guess_inference_settings("default").nitro is True
    for name in ("high-precision", "batch", "multigpu", "traineval"):
        assert guess_inference_settings(name).nitro is False


def test_uma_nitro_survives_omegaconf_roundtrip():
    import hydra

    config = inference_settings_nitro().to_omegaconf()
    assert config["nitro"] is True
    assert hydra.utils.instantiate(config) == inference_settings_nitro()


@pytest.mark.parametrize(
    "dtype_str, expected",
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
    ],
)
def test_string_input_converted_to_dtype(dtype_str, expected):
    settings = InferenceSettings(base_precision_dtype=dtype_str)
    assert settings.base_precision_dtype is expected


def test_torch_dtype_input_passes_through():
    settings = InferenceSettings(base_precision_dtype=torch.float64)
    assert settings.base_precision_dtype is torch.float64


def test_invalid_string_raises():
    with pytest.raises(AssertionError):
        InferenceSettings(base_precision_dtype="int8")


# --- to_omegaconf ---


@pytest.mark.parametrize(
    "dtype, expected_str",
    [
        (torch.float32, "float32"),
        (torch.float64, "float64"),
    ],
)
def test_to_omegaconf_dtype_serialized_as_string(dtype, expected_str):
    settings = InferenceSettings(base_precision_dtype=dtype)
    config = settings.to_omegaconf()
    assert config["base_precision_dtype"] == expected_str


def test_to_omegaconf_has_target():
    config = InferenceSettings().to_omegaconf()
    assert config["_target_"] == (
        "fairchem.core.units.mlip_unit.api.inference.InferenceSettings"
    )


def test_to_omegaconf_roundtrip():
    """Hydra can reinstantiate InferenceSettings from to_omegaconf() output."""
    import hydra

    original = InferenceSettings(base_precision_dtype=torch.float64, tf32=True)
    config = original.to_omegaconf()
    restored = hydra.utils.instantiate(config)
    assert isinstance(restored, InferenceSettings)
    assert restored.base_precision_dtype is torch.float64
    assert restored.tf32 is True


# --- reduced-precision warning ---


@pytest.mark.parametrize(
    "settings",
    [
        inference_settings_high_precision(),
        inference_settings_batch(),
        inference_settings_traineval(),
        InferenceSettings(tf32=True, base_precision_dtype=torch.float64),
    ],
)
def test_standard_precision_modes_do_not_warn(settings):
    assert not _uses_reduced_precision(settings, "cuda")


@pytest.mark.parametrize(
    "settings",
    [
        inference_settings_default(),
        inference_settings_multigpu(),
        InferenceSettings(tf32=True),
        inference_settings_nitro(),
    ],
)
def test_reduced_precision_modes_warn_on_cuda_only(settings):
    assert _uses_reduced_precision(settings, "cuda")
    assert _uses_reduced_precision(settings, "cuda:0")
    # TF32 does not apply on the CPU.
    assert not _uses_reduced_precision(settings, "cpu")


def test_reduced_precision_warning_is_short_and_says_how_to_get_fp32():
    assert "higher-order derivatives" in _REDUCED_PRECISION_WARNING
    assert "phonon" in _REDUCED_PRECISION_WARNING
    assert "inference_settings='high-precision'" in _REDUCED_PRECISION_WARNING
    assert len(_REDUCED_PRECISION_WARNING) < 220


# --- fallback messages ---


def test_settings_change_names_changed_fields_and_precision():
    import dataclasses

    nitro = inference_settings_nitro()
    from fairchem.core.models.uma.nn.execution_backends import ExecutionMode

    multigpu = dataclasses.replace(
        nitro, nitro=False, execution_mode=ExecutionMode.UMAS_FAST_GPU
    )
    assert _describe_settings_change(nitro, multigpu, "cuda") == (
        "Now running with nitro=False, execution_mode='umas_fast_gpu' (TF32)."
    )
    general = dataclasses.replace(multigpu, merge_mole=False, compile=False)
    assert _describe_settings_change(multigpu, general, "cpu") == (
        "Now running with merge_mole=False, compile=False (FP32)."
    )
    fp64 = InferenceSettings(base_precision_dtype=torch.float64)
    assert _describe_settings_change(fp64, fp64, "cuda") == (
        "Now running with unchanged settings (FP64)."
    )
