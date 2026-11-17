"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import ProcessRaisedException

from fairchem.core.common.test_utils import (
    PGConfig,
    init_env_rank_and_launch_test,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)

pytestmark = pytest.mark.serial


def _rank_result(value, *, timeout):
    """
    Return rank data and verify test-method timeout kwargs are forwarded.
    """
    return int(os.environ["RANK"]), value, timeout


def _reduce_result():
    """
    Exercise a real CPU distributed collective before worker cleanup.
    """
    value = torch.tensor(dist.get_rank() + 1)
    dist.all_reduce(value)
    return value.item()


def _raise_in_child():
    """
    Raise an identifiable exception that the parent must preserve.
    """
    raise RuntimeError("distributed child failure probe")


def _stall_in_child(directory):
    """
    Ignore TERM to exercise the final bounded kill and join.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(directory, str(os.getpid())).touch()
    time.sleep(120)


def _child_pids():
    """
    Include the Manager server in our cleanup assertion.
    """
    return {process.pid for process in multiprocessing.active_children()}


def test_spawn_success_cleans_manager_and_init_file():
    before = _child_pids()
    config = PGConfig("gloo", 2, use_gp=False)
    assert spawn_multi_process(
        config,
        _rank_result,
        init_env_rank_and_launch_test,
        42,
        spawn_timeout_seconds=30,
        timeout="forwarded",
    ) == [(0, 42, "forwarded"), (1, 42, "forwarded")]
    assert _child_pids() == before
    assert not Path(config.init_file).exists()


@pytest.mark.parametrize(
    "initializer", [init_env_rank_and_launch_test, init_pg_and_rank_and_launch_test]
)
def test_spawn_preserves_child_exception_and_cleans_up(initializer):
    before = _child_pids()
    config = PGConfig("gloo", 2, use_gp=False)
    with pytest.raises(ProcessRaisedException, match="distributed child failure probe"):
        spawn_multi_process(
            config,
            _raise_in_child,
            initializer,
            spawn_timeout_seconds=30,
        )
    assert _child_pids() == before
    assert not Path(config.init_file).exists()


def test_spawn_distributed_success_cleans_up():
    before = _child_pids()
    config = PGConfig("gloo", 2, use_gp=False)
    assert spawn_multi_process(
        config,
        _reduce_result,
        init_pg_and_rank_and_launch_test,
        spawn_timeout_seconds=30,
    ) == [3, 3]
    assert _child_pids() == before
    assert not Path(config.init_file).exists()


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX signal cleanup")
def test_spawn_stall_kills_children_and_manager(tmp_path):
    before = _child_pids()
    config = PGConfig("gloo", 2, use_gp=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="unfinished child PIDs"):
        spawn_multi_process(
            config,
            _stall_in_child,
            init_env_rank_and_launch_test,
            str(tmp_path),
            # The deadline includes fresh-process package imports before the stall.
            spawn_timeout_seconds=30,
        )
    assert time.monotonic() - started < 40
    assert list(tmp_path.iterdir()), "children must reach the stall before timeout"
    for path in tmp_path.iterdir():
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.name), 0)
    assert _child_pids() == before
    assert not Path(config.init_file).exists()
