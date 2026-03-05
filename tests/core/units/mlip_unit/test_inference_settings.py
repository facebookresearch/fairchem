"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def test_default_dtype_is_float32():
    settings = InferenceSettings()
    assert settings.base_precision_dtype is torch.float32


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, torch.bfloat16],
)
def test_torch_dtype_passthrough(dtype):
    settings = InferenceSettings(base_precision_dtype=dtype)
    assert settings.base_precision_dtype is dtype


@pytest.mark.parametrize(
    ("dtype_str", "expected"),
    [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_string_to_dtype_conversion(dtype_str, expected):
    settings = InferenceSettings(base_precision_dtype=dtype_str)
    assert settings.base_precision_dtype is expected


def test_invalid_string_raises():
    with pytest.raises(AssertionError):
        InferenceSettings(base_precision_dtype="int8")


def test_hydra_instantiate_with_string_dtype():
    cfg = DictConfig(
        {
            "_target_": "fairchem.core.units.mlip_unit.api.inference.InferenceSettings",
            "base_precision_dtype": "float64",
        }
    )
    settings = instantiate(cfg)
    assert settings.base_precision_dtype is torch.float64
