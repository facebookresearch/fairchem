"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import multiprocessing
import os
import pdb
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torchtnt.utils.distributed import get_file_init_method

from fairchem.core.common.gp_utils import setup_gp


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used from a forked multiprocessing child.
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess/23654936#23654936

    example usage to debug a torch distributed run on rank 0:
    if torch.distributed.get_rank() == 0:
        from fairchem.core.common.test_utils import ForkedPdb
        ForkedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin") as f:
                sys.stdin = f
                pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


@dataclass
class PGConfig:
    backend: str
    world_size: int
    gp_group_size: int = 1
    use_gp: bool = True


def init_env_rank_and_launch_test(
    rank: int,
    pg_setup_params: PGConfig,
    mp_output_dict: dict[int, object],
    test_method: callable,
    args: list[object],
    kwargs: dict[str, object],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["WORLD_SIZE"] = str(pg_setup_params.world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    mp_output_dict[rank] = test_method(*args, **kwargs)  # pyre-fixme


def init_pg_and_rank_and_launch_test(
    rank: int,
    pg_setup_params: PGConfig,
    mp_output_dict: dict[int, object],
    test_method: callable,
    args: list[object],
    kwargs: dict[str, object],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["WORLD_SIZE"] = str(pg_setup_params.world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    init_method = get_file_init_method(
        world_size=pg_setup_params.world_size,
        rank=rank,
        filename=pg_setup_params.init_file,
    )
    # setup default process group
    dist.init_process_group(
        init_method=init_method,
        backend=pg_setup_params.backend,
        timeout=timedelta(seconds=120),
    )
    try:
        # setup gp
        if pg_setup_params.use_gp:
            config = {
                "gp_gpus": pg_setup_params.gp_group_size,
                "distributed_backend": pg_setup_params.backend,
            }
            setup_gp(config)
        mp_output_dict[rank] = test_method(*args, **kwargs)  # pyre-fixme
    finally:
        dist.destroy_process_group()


def spawn_multi_process(
    config: PGConfig,
    test_method: callable,
    init_and_launch: callable,
    *test_method_args: Any,
    spawn_timeout_seconds: float = 600,
    **test_method_kwargs: Any,
) -> list[Any]:
    """
    Spawn single node, multi-rank function.
    Uses a shared file for process group initialization to avoid port races.

    IMPORTANT: ``test_method`` must return CPU tensors or plain Python values.
    Results are passed back through a ``multiprocessing.Manager()`` dict whose
    server process is forked from the caller, so unpickling a CUDA tensor there
    fails with "Cannot re-initialize CUDA in forked subprocess" (the CUDA IPC
    rebuild needs a CUDA context the forked server cannot create). Call
    ``.detach().cpu()`` on any tensor before returning it from a NCCL worker.

    Args:
        config: Process group configuration, including world size and backend.
        test_method: callable to spawn. first 3 arguments are rank, world_size and mp output dict
        init_and_launch: Child initializer that invokes the test method.
        test_method_args: args for the test method
        spawn_timeout_seconds: Maximum parent wait, including child startup.
        test_method_kwargs: kwargs for the test method

    Returns:
        A list, l, where l[i] is the return value of test_method on rank i
    """
    if spawn_timeout_seconds <= 0:
        raise ValueError("spawn_timeout_seconds must be positive")

    with tempfile.NamedTemporaryFile(delete=False) as f:
        config.init_file = f.name

    context = None
    try:
        with multiprocessing.Manager() as manager:
            mp_output_dict = manager.dict()
            deadline = time.monotonic() + spawn_timeout_seconds
            context = torch.multiprocessing.spawn(
                # torch.multiprocessing.spawn sends rank as the first param
                # https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
                init_and_launch,
                args=(
                    config,
                    mp_output_dict,
                    test_method,
                    test_method_args,
                    test_method_kwargs,
                ),
                nprocs=config.world_size,
                join=False,
            )
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        pids = [
                            process.pid
                            for process in context.processes
                            if process.is_alive()
                        ]
                        raise TimeoutError(
                            f"Distributed test exceeded {spawn_timeout_seconds}s; "
                            f"unfinished child PIDs: {pids}"
                        )
                    if context.join(timeout=min(remaining, 1), grace_period=1):
                        break
                return [mp_output_dict[i] for i in range(config.world_size)]
            finally:
                # Reap workers before closing their shared Manager server.
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                cleanup_deadline = time.monotonic() + 5
                for process in context.processes:
                    process.join(timeout=max(0, cleanup_deadline - time.monotonic()))
                for process in context.processes:
                    if process.is_alive():
                        process.kill()
                    process.join(timeout=1)
    finally:
        if os.path.exists(config.init_file):
            os.unlink(config.init_file)
        if context is not None:
            for filename in context.error_files:
                if os.path.exists(filename):
                    os.unlink(filename)


def init_local_distributed_process_group(backend="nccl"):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    init_method = get_file_init_method(world_size=1, rank=0, filename=init_file)
    dist.init_process_group(
        init_method=init_method,
        backend=backend,
        timeout=timedelta(seconds=100),
    )
