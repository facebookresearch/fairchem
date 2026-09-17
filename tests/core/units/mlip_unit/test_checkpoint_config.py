"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from fairchem.core.units.mlip_unit.mlip_unit import (
    _get_train_eval_unit_config,
    _set_model_id_in_config,
)


@pytest.mark.parametrize("ray_wrapped", [False, True])
def test_set_model_id_in_checkpoint_config(ray_wrapped):
    train_runner = {"train_eval_unit": {"model": {"_target_": "model"}}}
    runner = {"runner_config": train_runner} if ray_wrapped else train_runner
    config = OmegaConf.create({"runner": runner})
    OmegaConf.set_struct(config, True)

    _set_model_id_in_config(config, "UMA-generated")

    train_eval_unit_config = _get_train_eval_unit_config(config)
    assert train_eval_unit_config.model.model_id == "UMA-generated"
