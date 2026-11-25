"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fairchem.core.regression_suite.execution import settings as settings_module
from fairchem.core.regression_suite.execution.settings import selected_suites
from fairchem.core.regression_suite.spec.profiles import (
    build_inference_settings,
    json_settings,
    profile_config,
    unsupported_suites_for_profile,
)
from fairchem.core.regression_suite.suites import load_suite

_PROFILES = sorted(
    path.stem
    for path in Path(settings_module.__file__).parents[1].glob("configs/profile/*.yaml")
)


@pytest.mark.parametrize("profile", _PROFILES)
def test_profiles_build_valid_settings_and_declare_unsupported_suites(profile):
    config = profile_config(profile)
    unsupported = unsupported_suites_for_profile(profile)

    assert config.label == profile.replace("_", "-")
    for suite in selected_suites(None, profile):
        overrides = load_suite(suite).inference_settings_overrides
        settings = build_inference_settings(config, overrides)
        json.dumps(json_settings(settings))
        fast = settings.execution_mode in ("umas_fast_gpu", "umas_fast_pytorch")
        # Fast backends reject activation checkpointing; umas_fast_gpu needs merging.
        assert not (fast and settings.activation_checkpointing)
        assert settings.merge_mole or settings.execution_mode != "umas_fast_gpu"
        # A merged MOLE model serves one composition, so structures cannot share it.
        assert (config.get("predictor_scope") == "structure") == settings.merge_mole
    elastic = build_inference_settings(
        config, load_suite("elastic").inference_settings_overrides
    )
    # Autograd Hessians need eager, vmap-capable backward operators.
    assert ("autograd-frequencies" in unsupported) == (
        elastic.compile or elastic.execution_mode == "umas_fast_gpu"
    )
    assert "finite-difference-frequencies" not in unsupported
