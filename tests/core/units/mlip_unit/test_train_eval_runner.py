"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
import time

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from fairchem.core._cli import get_hydra_config_from_yaml
from fairchem.core.common.distutils import assign_device_for_local_rank, setup_env_local
from fairchem.core.components.train.train_runner import (
    get_most_recent_viable_checkpoint_path,
)


def check_model_state_equal(old_state: dict, new_state: dict) -> bool:
    if set(old_state.keys()) != set(new_state.keys()):
        return False
    for key in old_state:  # noqa: SIM110
        if not torch.allclose(old_state[key], new_state[key]):
            return False
    return True


def test_traineval_runner_save_and_load_checkpoint(fake_uma_dataset):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    assign_device_for_local_rank(True, 0)
    setup_env_local()
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    config = "tests/core/units/mlip_unit/test_mlip_train.yaml"
    # remove callbacks for checking loss
    # TODO mock main to avoid repeating this code in other tests
    cfg = get_hydra_config_from_yaml(
        config,
        [
            "expected_loss=null",
            "checkpoint_every=null",
            f"datasets.data_root_dir={fake_uma_dataset}",
        ],
    )
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.job.run_dir, cfg.job.timestamp_id), exist_ok=True)
    OmegaConf.save(cfg, cfg.job.metadata.config_path)

    runner = hydra.utils.instantiate(cfg.runner)
    runner.job_config = cfg.job
    runner.run()

    ch_path = cfg.job.metadata.checkpoint_dir
    # if we save state the state, the state object should be identical
    old_state = runner.train_eval_unit.state_dict()
    runner.save_state(ch_path)
    assert len(os.listdir(ch_path)) > 0
    # now re-initialize the runner and load_state
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # use a different seed so the runner cannot have the same state
    new_cfg = get_hydra_config_from_yaml(
        config,
        [
            "expected_loss=null",
            "checkpoint_every=null",
            f"datasets.data_root_dir={fake_uma_dataset}",
        ],
    )
    new_cfg.job.seed = 999
    assert new_cfg.job.seed != cfg.job.seed
    new_runner = hydra.utils.instantiate(new_cfg.runner)
    new_runner.job_config = new_cfg.job
    new_runner.config = new_cfg
    new_runner.run()
    new_state = new_runner.train_eval_unit.state_dict()
    # the states should be different here because we started with a different seed
    assert not check_model_state_equal(new_state["model"], old_state["model"])
    # now the states should be the same after loading, we call the _execute_load_state function to force loading
    new_runner.train_eval_unit._execute_load_state(ch_path)
    new_state_loaded = new_runner.train_eval_unit.state_dict()
    assert check_model_state_equal(old_state["model"], new_state_loaded["model"])


def test_get_most_recent_viable_checkpoint_path():
    test_dir = tempfile.TemporaryDirectory()
    dir_with_metadata = os.path.join(test_dir.name, "has_metadata")
    dir_with_metadata_newer = os.path.join(test_dir.name, "has_metadata_newer")
    dir_without_metadata = os.path.join(test_dir.name, "has_no_metadata")
    os.makedirs(dir_with_metadata, exist_ok=True)
    with open(os.path.join(dir_with_metadata, ".metadata"), "w") as f:
        f.write("This is a metadata file.")
    time.sleep(1)
    os.makedirs(dir_with_metadata_newer, exist_ok=True)
    with open(os.path.join(dir_with_metadata_newer, ".metadata"), "w") as f:
        f.write("This is a metadata file.")
    os.makedirs(dir_without_metadata, exist_ok=True)

    result = get_most_recent_viable_checkpoint_path(test_dir.name)
    assert result == dir_with_metadata_newer
    result = get_most_recent_viable_checkpoint_path("some/random/path")
    assert result is None
