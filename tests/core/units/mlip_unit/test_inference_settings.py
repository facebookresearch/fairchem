"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def test_default_dtype_is_float32():
    settings = InferenceSettings()
    assert settings.base_precision_dtype is torch.float32


def test_string_input_converted_to_dtype():
    settings = InferenceSettings(base_precision_dtype="float64")
    assert settings.base_precision_dtype is torch.float64


def test_invalid_string_raises():
    with pytest.raises(AssertionError):
        InferenceSettings(base_precision_dtype="int8")
