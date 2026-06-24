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
import pytest
import torch
from omegaconf import OmegaConf
from torchtnt.framework.unit import TrainUnit

from fairchem.core._cli import get_hydra_config_from_yaml
from fairchem.core.common.distutils import assign_device_for_local_rank
from fairchem.core.common.test_utils import init_local_distributed_process_group
from fairchem.core.components.train.callbacks import (
    StopBeforeTimeoutCallback,
    TrainCheckpointCallback,
)
from fairchem.core.components.train.train_runner import (
    TrainEvalRunner,
    TrainRunner,
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
    init_local_distributed_process_group(backend="gloo")
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


class _Unit:
    pass


class _CountingTrainUnit(TrainUnit[int]):
    def __init__(self):
        super().__init__()
        self.seen = []

    def train_step(self, _state, data):
        self.seen.append(data)


def test_train_eval_runner_accepts_no_callbacks():
    runner = TrainEvalRunner(
        train_dataloader=[1],
        eval_dataloader=[1],
        train_eval_unit=_Unit(),
        callbacks=None,
    )

    assert runner.callbacks == []
    assert runner.checkpoint_callback is None


def test_train_runner_accepts_no_callbacks():
    runner = TrainRunner(
        train_dataloader=[1],
        train_unit=_Unit(),
        callbacks=None,
    )

    assert runner.callbacks == []
    assert runner.checkpoint_callback is None


def test_train_runner_runs_torchtnt_train_loop():
    train_unit = _CountingTrainUnit()
    runner = TrainRunner(
        train_dataloader=[1, 2, 3],
        train_unit=train_unit,
        callbacks=None,
        max_steps=2,
    )

    runner.run()

    assert train_unit.seen == [1, 2]


class _HookCallback:
    def __init__(self):
        self.hooks = None

    def set_runner_callbacks(self, save_callback, load_callback, checkpoint_dir):
        self.hooks = (save_callback, load_callback, checkpoint_dir)


class _NonCallableHookCallback:
    set_runner_callbacks = "not-callable"


def test_train_eval_runner_initializes_generic_runner_callback_hooks(tmp_path):
    hook_callback = _HookCallback()
    runner = TrainEvalRunner(
        train_dataloader=[1],
        eval_dataloader=[1],
        train_eval_unit=_Unit(),
        callbacks=[hook_callback, _NonCallableHookCallback()],
    )
    runner.job_config = OmegaConf.create(
        {"metadata": {"checkpoint_dir": str(tmp_path)}}
    )

    runner._set_runner_callbacks(runner.callbacks)

    assert hook_callback.hooks == (
        runner.save_state,
        runner.load_state,
        str(tmp_path),
    )


def test_train_runner_initializes_generic_runner_callback_hooks(tmp_path):
    hook_callback = _HookCallback()
    runner = TrainRunner(
        train_dataloader=[1],
        train_unit=_Unit(),
        callbacks=[hook_callback, _NonCallableHookCallback()],
    )
    runner.job_config = OmegaConf.create(
        {"metadata": {"checkpoint_dir": str(tmp_path)}}
    )

    runner._set_runner_callbacks(runner.callbacks)

    assert hook_callback.hooks == (
        runner.save_state,
        runner.load_state,
        str(tmp_path),
    )


class _StopState:
    class _EvalState:
        def __init__(self):
            self._max_steps_per_epoch = None

    def __init__(self):
        self.stopped = False
        self.eval_state = self._EvalState()

    def stop(self):
        self.stopped = True


class _StopUnit:
    class _Progress:
        num_steps_completed = 12

    train_progress = _Progress()


class _CheckpointUnit:
    class _Progress:
        num_steps_completed = 0

    train_progress = _Progress()


def test_train_checkpoint_callback_saves_step_zero(tmp_path):
    saved_paths = []
    callback = TrainCheckpointCallback(checkpoint_every_n_steps=5)
    callback.set_runner_callbacks(saved_paths.append, lambda _: None, str(tmp_path))
    unit = _CheckpointUnit()

    callback.on_train_step_start(None, unit)
    assert saved_paths == [os.path.join(tmp_path, "step_0")]

    unit.train_progress.num_steps_completed = 5
    callback.on_train_step_start(None, unit)
    assert saved_paths == [
        os.path.join(tmp_path, "step_0"),
        os.path.join(tmp_path, "step_5"),
    ]


def test_train_checkpoint_callback_requires_positive_retention():
    with pytest.raises(ValueError, match="max_saved_checkpoints"):
        TrainCheckpointCallback(
            checkpoint_every_n_steps=5,
            max_saved_checkpoints=0,
        )


def test_train_checkpoint_callback_updates_latest_and_rotates(tmp_path):
    def save_checkpoint(path):
        os.makedirs(path)
        with open(os.path.join(path, ".metadata"), "w") as f:
            f.write("metadata")

    callback = TrainCheckpointCallback(
        checkpoint_every_n_steps=5,
        max_saved_checkpoints=2,
    )
    callback.set_runner_callbacks(save_checkpoint, lambda _: None, str(tmp_path))
    unit = _CheckpointUnit()

    unit.train_progress.num_steps_completed = 5
    callback.on_train_step_start(None, unit)
    unit.train_progress.num_steps_completed = 10
    callback.on_train_step_start(None, unit)
    unit.train_progress.num_steps_completed = 15
    callback.on_train_step_start(None, unit)

    latest = tmp_path / "latest"
    assert latest.is_symlink()
    assert os.readlink(latest) == "step_15"
    assert not (tmp_path / "step_5").exists()
    assert (tmp_path / "step_10").exists()
    assert (tmp_path / "step_15").exists()

    callback.on_train_end(None, unit)
    assert os.readlink(latest) == "final"
    assert (tmp_path / "final").exists()


def test_stop_before_timeout_callback_requests_graceful_stop():
    now = 0.0

    def clock():
        return now

    callback = StopBeforeTimeoutCallback(
        timeout_hr=1.0,
        stop_before_timeout_min=30.0,
        clock=clock,
    )
    state = _StopState()

    callback.on_train_step_start(state, _StopUnit())
    assert not state.stopped

    now = 31 * 60
    callback.on_train_step_start(state, _StopUnit())
    assert state.stopped
    assert state.eval_state._max_steps_per_epoch == 0


def test_stop_before_timeout_callback_checks_after_train_step():
    now = 0.0

    def clock():
        return now

    callback = StopBeforeTimeoutCallback(
        timeout_hr=1.0,
        stop_before_timeout_min=30.0,
        clock=clock,
    )
    state = _StopState()

    callback.on_train_step_start(state, _StopUnit())
    assert not state.stopped

    now = 31 * 60
    callback.on_train_step_end(state, _StopUnit())
    assert state.stopped
    assert state.eval_state._max_steps_per_epoch == 0
