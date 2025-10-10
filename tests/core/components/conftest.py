"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from random import choice

import pytest

from fairchem.core import FAIRChemCalculator, pretrained_mlip


@pytest.fixture(scope="session")
def calculator() -> FAIRChemCalculator:
    uma_sm_models = [
        model for model in pretrained_mlip.available_models if "uma-s" in model
    ]
    return FAIRChemCalculator.from_model_checkpoint(
        choice(uma_sm_models), task_name="omat"
    )
