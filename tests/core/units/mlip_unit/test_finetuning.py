"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import hydra
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from fairchem.core._cli import get_hydra_config_from_yaml
from fairchem.core.common.distutils import assign_device_for_local_rank, setup_env_local
from fairchem.core.units.mlip_unit.mlip_unit import UNIT_INFERENCE_CHECKPOINT
from fairchem.core.units.mlip_unit.utils import update_configs


def check_backbone_state_equal(old_state: dict, new_state: dict) -> bool:
    old_backbone_keys = [key for key in old_state if "backbone" in key]
    new_backbone_keys = [key for key in new_state if "backbone" in key]
    if set(old_backbone_keys) != set(new_backbone_keys):
        return False
    for key in old_state:
        if "backbone" in key and (not torch.allclose(old_state[key], new_state[key])):
            print(f"key: {key}")
            return False
    return True


@pytest.mark.skip()
def test_traineval_runner_finetuning():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    setup_env_local()
    assign_device_for_local_rank(True, 0)
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    config = "tests/core/units/mlip_unit/test_mlip_train.yaml"
    # remove callbacks for checking loss

    cfg = get_hydra_config_from_yaml(config, ["expected_loss=null"])
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.job.run_dir, cfg.job.timestamp_id), exist_ok=True)
    OmegaConf.save(cfg, cfg.job.metadata.config_path)

    runner = hydra.utils.instantiate(cfg.runner)
    runner.job_config = cfg.job
    runner.run()

    ch_path = cfg.job.metadata.checkpoint_dir
    # if we save state the state, the state object should be identical
    old_state = runner.train_eval_unit.model.state_dict()
    runner.save_state(ch_path)
    assert len(os.listdir(ch_path)) > 0

    # now re-initialize the runner and load_state
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    finetune_config = "tests/core/units/mlip_unit/test_mlip_finetune.yaml"
    finetune_cfg = get_hydra_config_from_yaml(
        finetune_config,
        [
            "expected_loss=null",
            f"model.checkpoint_location={ch_path}/{UNIT_INFERENCE_CHECKPOINT}",
        ],
    )
    finetune_runner = hydra.utils.instantiate(finetune_cfg.runner)
    finetune_runner.job_config = cfg.job
    assert finetune_runner.train_eval_unit.model.module.backbone.max_neighbors == 300

    new_state_before = finetune_runner.train_eval_unit.model.state_dict()

    # backbone state should be the same
    assert check_backbone_state_equal(old_state, new_state_before)

    finetune_runner.run()
    new_state_after = finetune_runner.train_eval_unit.model.state_dict()

    # the backbone states should be different after finetuning
    assert not check_backbone_state_equal(old_state, new_state_after)

    fch_path = finetune_cfg.job.metadata.checkpoint_dir
    finetune_runner.save_state(fch_path)

    starting_ckpt = torch.load(
        f"{ch_path}/{UNIT_INFERENCE_CHECKPOINT}", weights_only=False
    )
    finetune_ckpt = torch.load(
        f"{fch_path}/{UNIT_INFERENCE_CHECKPOINT}", weights_only=False
    )

    # backbone config of finetuned model inference checkpoint should match the starting checkpoint without overrides
    start_model_config = starting_ckpt.model_config
    updated_model_config = update_configs(
        start_model_config, finetune_cfg.runner.train_eval_unit.model.overrides
    )
    finetune_model_config = finetune_ckpt.model_config

    assert updated_model_config["backbone"] == finetune_model_config["backbone"]
