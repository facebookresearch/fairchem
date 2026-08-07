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
    inference_settings_turbo,
)

# --- __post_init__ ---


def test_default_dtype_is_float32():
    settings = InferenceSettings()
    assert settings.base_precision_dtype is torch.float32


def test_named_modes_use_expected_paths_and_only_turbo_enables_tf32():
    default = inference_settings_default()
    batch = inference_settings_batch()
    turbo = inference_settings_turbo()

    assert default.merge_mole is True
    assert default.compile is True
    assert default.tf32 is False
    assert default.activation_checkpointing is False
    assert batch.merge_mole is False
    assert batch.compile is False
    assert batch.tf32 is False
    assert batch.activation_checkpointing is True
    assert turbo.merge_mole is True
    assert turbo.compile is True
    assert turbo.tf32 is True
    assert turbo.activation_checkpointing is False


def test_batch_is_a_named_inference_mode():
    assert guess_inference_settings("batch") == inference_settings_batch()


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
