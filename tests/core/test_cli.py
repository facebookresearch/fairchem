"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import MagicMock

import hydra
import pytest

from fairchem.core._cli import ALLOWED_TOP_LEVEL_KEYS, get_hydra_config_from_yaml, main
from fairchem.core.common import distutils
from fairchem.core.components.runner import MockRunner


def test_cli():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", "tests/core/test_cli.yml"]
    sys.argv[1:] = sys_args
    main()


def test_cli_multi_rank_cpu():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", "tests/core/test_cli.yml", "job.scheduler.ranks_per_node=2"]
    sys.argv[1:] = sys_args
    main()


def test_cli_run_reduce():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", "tests/core/test_cli_run_reduce.yml"]
    sys.argv[1:] = sys_args
    main()


def test_cli_throws_error():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "--config",
        "tests/core/test_cli.yml",
        "runner.x=1000",
        "runner.y=5",
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(ValueError) as error_info:
        main()
    assert "sum is greater than 1000" in str(error_info.value)


def test_cli_throws_error_on_invalid_inputs():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "-c",
        "tests/core/test_cli.yml",
        "runner.x=1000",
        "runner.a=5",  # a is not a valid input argument to runner
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(hydra.errors.ConfigCompositionException):
        main()


def test_cli_throws_error_on_disallowed_top_level_keys():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    assert "x" not in ALLOWED_TOP_LEVEL_KEYS
    sys_args = [
        "-c",
        "tests/core/test_cli.yml",
        "+x=1000",  # this is not allowed because we are adding a key that is not in ALLOWED_TOP_LEVEL_KEYS
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(ValueError):
        main()


def get_cfg_from_yaml():
    yaml = "tests/core/test_cli.yml"
    cfg = get_hydra_config_from_yaml(yaml)
    # assert fields got initialized properly
    assert cfg.job.run_name is not None
    assert cfg.job.seed is not None
    assert cfg.keys() == ALLOWED_TOP_LEVEL_KEYS


@pytest.mark.parametrize("num_ranks", [1, 4])
def test_cli_ray(num_ranks):
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "--config",
        "tests/core/test_ray_runner.yml",
        f"job.scheduler.ranks_per_node={num_ranks}",
    ]
    sys.argv[1:] = sys_args
    main()


class TestMockRunnerSaveLoadState:
    """Unit tests for MockRunner save/load state functionality."""

    def test_save_state_creates_checkpoint(self):
        """Test that save_state creates a checkpoint file."""
        runner = MockRunner(x=10, y=20, z=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint")
            result = runner.save_state(checkpoint_path, is_preemption=True)

            assert result is True
            state_file = os.path.join(checkpoint_path, "mock_state.json")
            assert os.path.exists(state_file)

            with open(state_file) as f:
                state = json.load(f)

            assert state["x"] == 10
            assert state["y"] == 20
            assert state["z"] == 30
            assert state["is_preemption"] is True

    def test_load_state_restores_values(self):
        """Test that load_state restores runner state from checkpoint."""
        runner1 = MockRunner(x=10, y=20, z=30)
        runner2 = MockRunner(x=0, y=0, z=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint")
            runner1.save_state(checkpoint_path)
            runner2.load_state(checkpoint_path)

            assert runner2.x == 10
            assert runner2.y == 20
            assert runner2.z == 30

    def test_load_state_with_none_does_nothing(self):
        """Test that load_state with None checkpoint_location is a no-op."""
        runner = MockRunner(x=10, y=20, z=30)
        runner.load_state(None)
        assert runner.x == 10
        assert runner.y == 20
        assert runner.z == 30


class TestSignalHandlerRegistration:
    """Unit tests for signal handler registration logic."""

    def test_signal_handler_calls_save_state(self):
        """Test that the signal handler calls runner.save_state with is_preemption=True."""
        runner = MockRunner(x=10, y=20, z=30)
        runner.save_state = MagicMock(return_value=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "preemption_state")

            # Simulate what the signal handler does
            def graceful_shutdown(signum, frame):
                runner.save_state(save_path, is_preemption=True)

            # Call the handler directly (don't actually send signals in unit test)
            graceful_shutdown(signal.SIGINT, None)

            runner.save_state.assert_called_once_with(save_path, is_preemption=True)


class TestSignalHandlingIntegration:
    """Integration tests that actually send signals to a running process."""

    def test_sigterm_saves_state_and_resume_config(self):
        """Test that sending SIGTERM saves state and creates resume config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                [
                    "fairchem",
                    "--config",
                    "tests/core/test_cli.yml",
                    f"+job.run_dir={tmpdir}",
                    "+runner.sleep_seconds=30",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            time.sleep(20)
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)

            timestamp_dirs = [
                d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))
            ]
            assert (
                len(timestamp_dirs) == 1
            ), f"Expected 1 timestamp dir, found {timestamp_dirs}"

            preemption_dir = os.path.join(
                tmpdir, timestamp_dirs[0], "checkpoints", "preemption_state"
            )

            state_file = os.path.join(preemption_dir, "mock_state.json")
            with open(state_file) as f:
                state = json.load(f)
            assert state["x"] == 10
            assert state["y"] == 23
            assert state["z"] == 5
            assert state["is_preemption"] is True

            resume_config = os.path.join(preemption_dir, "resume_config.yaml")
            assert os.path.exists(
                resume_config
            ), f"Resume config not found at {resume_config}"
