"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from unittest.mock import Mock

from omegaconf import OmegaConf

from fairchem.core.launchers import slurm_launch
from fairchem.core.launchers.api import RunType, SlurmEnv


def test_environment_report_follows_distributed_setup(tmp_path, monkeypatch) -> None:
    events = []
    runner = Mock()
    config = OmegaConf.create(
        {
            "job": {
                "metadata": {
                    "log_dir": str(tmp_path),
                    "commit": "abc123",
                    "slurm_env": {},
                },
                "timestamp_id": "timestamp",
                "scheduler": {"mode": "local"},
                "graph_parallel": {"group_size": 1},
                "logger": None,
                "debug": False,
                "seed": 0,
                "deterministic": False,
                "runner_state_path": None,
            },
            "runner": {},
        }
    )
    slurm_environment = SlurmEnv(
        job_id="123_4",
        array_job_id="123",
        array_task_id="4",
        restart_count="1",
    )

    monkeypatch.setattr(slurm_launch, "_get_slurm_env", lambda: slurm_environment)
    monkeypatch.setattr(slurm_launch, "setup_env_vars", lambda: None)
    monkeypatch.setattr(slurm_launch, "setup_logging", lambda: None)
    monkeypatch.setattr(slurm_launch, "map_job_config_to_dist_config", lambda _: {})
    monkeypatch.setattr(
        slurm_launch.distutils, "setup", lambda _: events.append("setup")
    )
    monkeypatch.setattr(
        slurm_launch.distutils, "synchronize", lambda: events.append("synchronize")
    )
    monkeypatch.setattr(slurm_launch.distutils, "is_master", lambda: False)
    monkeypatch.setattr(
        slurm_launch.distutils, "cleanup", lambda: events.append("cleanup")
    )

    def record_environment_report(**kwargs):
        events.append("environment")
        assert kwargs == {
            "log_dir": str(tmp_path),
            "run_type": "run",
            "timestamp_id": "timestamp",
            "submission_commit": "abc123",
        }

    monkeypatch.setattr(
        slurm_launch, "write_environment_report", record_environment_report
    )
    monkeypatch.setattr(slurm_launch, "_set_seeds", lambda _: None)
    monkeypatch.setattr(slurm_launch.hydra.utils, "instantiate", lambda _: runner)
    monkeypatch.setattr(slurm_launch.SlurmSPMDProgram, "_init_logger", lambda _: None)

    slurm_launch.SlurmSPMDProgram()(config, RunType.RUN)

    assert events == ["setup", "synchronize", "environment", "cleanup"]
    runner.load_state.assert_called_once_with(None)
    runner.run.assert_called_once_with()
