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
    assert settings.get_torch_dtype(settings.base_precision_dtype_str) is torch.float32


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
    assert settings.get_torch_dtype(settings.base_precision_dtype_str) is torch.float64
