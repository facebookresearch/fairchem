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
)

# --- __post_init__ ---


def test_default_dtype_is_float32():
    settings = InferenceSettings()
    assert settings.base_precision_dtype is torch.float32


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

    original = InferenceSettings(
        base_precision_dtype=torch.float64,
        tf32=True,
        compile_mode="reduce-overhead",
        compile_dynamic=False,
    )
    config = original.to_omegaconf()
    restored = hydra.utils.instantiate(config)
    assert isinstance(restored, InferenceSettings)
    assert restored.base_precision_dtype is torch.float64
    assert restored.tf32 is True
    assert restored.compile_mode == "reduce-overhead"
    assert restored.compile_dynamic is False


@pytest.mark.parametrize(
    ("name", "tf32", "checkpointing", "merge_mole", "compile_mode", "dynamic"),
    [
        ("default", False, True, False, None, True),
        ("turbo", True, False, True, None, True),
        ("turbo-fixed", True, False, True, "reduce-overhead", False),
    ],
)
def test_named_inference_mode_policy(
    name,
    tf32,
    checkpointing,
    merge_mole,
    compile_mode,
    dynamic,
):
    settings = guess_inference_settings(name)

    assert settings.tf32 is tf32
    assert settings.activation_checkpointing is checkpointing
    assert settings.merge_mole is merge_mole
    assert settings.compile is (name != "default")
    assert settings.compile_mode == compile_mode
    assert settings.compile_dynamic is dynamic
