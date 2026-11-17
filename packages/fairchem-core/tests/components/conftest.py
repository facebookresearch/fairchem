"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


@pytest.fixture(scope="session")
def calculator(direct_checkpoint) -> FAIRChemCalculator:
    inference_checkpoint_pt, _ = direct_checkpoint
    predictor = load_predict_unit(inference_checkpoint_pt, device="cpu")
    return FAIRChemCalculator(predictor, task_name="oc20")
