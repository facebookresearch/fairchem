"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import os
import pickle
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from submitit import AutoExecutor
from submitit.helpers import DelayedSubmission

from fairchem.core.regression_suite.spec.schema import CommandError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fairchem.core.launchers.api import SlurmConfig


def create_slurm_executor(
    folder: str | Path,
    *,
    name: str,
    resources: SlurmConfig,
    gpus_per_node: int,
    device: str = "cuda",
    array_parallelism: int | None = None,
    dependency: str | None = None,
    after: str = "afterok",
) -> AutoExecutor:
    """
    Create a one-node Submitit executor; ``dependency`` names jobs that must
    finish first, cleanly unless ``after`` is ``afterany``. No job is
    submitted.

    ``gpus_per_node`` GPUs are requested for CUDA launches and none for CPU
    launches.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    executor = AutoExecutor(folder=str(folder), slurm_max_num_timeout=3)
    parameters = {
        "name": name,
        "mem_gb": resources.mem_gb,
        "timeout_min": resources.timeout_hr * 60,
        "slurm_partition": resources.partition,
        "gpus_per_node": gpus_per_node if device == "cuda" else 0,
        "cpus_per_task": resources.cpus_per_task,
        "tasks_per_node": 1,
        "nodes": 1,
        "slurm_qos": resources.qos,
        "slurm_account": resources.account,
        "slurm_additional_parameters": dict(resources.additional_parameters or {}),
    }
    if array_parallelism is not None:
        parameters["slurm_array_parallelism"] = array_parallelism
    if dependency is not None:
        parameters["slurm_dependency"] = f"{after}:{dependency}"
        parameters["slurm_additional_parameters"]["kill-on-invalid-dep"] = "yes"
    executor.update_parameters(**parameters)
    return executor


class Rerunnable(functools.partial):
    """
    A partial that submitit requeues to rerun from the start after a timeout.

    Wrap only calls that are safe to repeat after being killed at any point.
    A SLURM job is requeued at most ``slurm_max_num_timeout`` times.
    """

    def checkpoint(self, *args: Any, **kwargs: Any) -> DelayedSubmission:
        return DelayedSubmission(self, *args, **kwargs)


# Each task runs in a fresh interpreter that imports only what its pickled
# callable needs, resolved through the inherited PYTHONPATH.
_WORKER = """
import pickle
import sys

with open(sys.argv[1], "rb") as handle:
    function, arguments = pickle.load(handle)
function(*arguments)
"""


@dataclass(frozen=True)
class LocalJob:
    """
    Identifier of one completed local task.
    """

    job_id: str


def visible_gpus() -> list[str]:
    """
    Return the CUDA devices this process may assign to local tasks.
    """
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [device for device in configured.split(",") if device.strip()]
    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


_CPU_TASK_THREADS = 4


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class LocalExecutor:
    """
    Run regression tasks on this machine.

    Tasks of phases that use the launch device run one per visible GPU, or,
    for CPU launches, one per four CPU cores with ``OMP_NUM_THREADS=4`` unless
    it is set. Other phases, such as merges and finalization, run one task at
    a time. Every task without a GPU has CUDA hidden. Each task runs in a fresh
    Python process that inherits ``PYTHONPATH``, so a launch's source snapshot
    is imported exactly as it is by SLURM jobs. Calls block until every task
    finishes and raise if any task failed.
    """

    def __init__(
        self,
        folder: str | Path,
        *,
        name: str,
        gpus_per_node: int,
        device: str = "cuda",
        **_scheduling,
    ):
        # Phases run in submission order and each call blocks, so SLURM
        # resources and dependencies are unnecessary.
        self.folder = Path(folder)
        self.name = name
        self.gpus_per_node = gpus_per_node
        self.device = device

    def map_array(self, function: Callable[..., Any], tasks: Sequence[Any]):
        return self._run(function, [(task,) for task in tasks])

    def submit(self, function: Callable[..., Any]):
        return self._run(function, [()])[0]

    def _slots(self, task_count: int) -> list[dict[str, str]]:
        """
        Return the environment overrides of each concurrent worker.
        """
        if not self.gpus_per_node:
            return [{"CUDA_VISIBLE_DEVICES": ""}]
        if self.device == "cpu":
            workers = min(task_count, max(1, _cpu_count() // _CPU_TASK_THREADS))
            threads = os.environ.get("OMP_NUM_THREADS", str(_CPU_TASK_THREADS))
            return [{"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": threads}] * workers
        gpus = visible_gpus()
        if not gpus:
            raise RuntimeError("local GPU tasks require a visible CUDA device")
        return [{"CUDA_VISIBLE_DEVICES": gpu} for gpu in gpus]

    def _run(self, function, calls: list[tuple[Any, ...]]) -> list[LocalJob]:
        slots = self._slots(len(calls))
        self.folder.mkdir(parents=True, exist_ok=True)
        pending: queue.Queue[int] = queue.Queue()
        for index in range(len(calls)):
            pending.put(index)
        failures: dict[int, Path] = {}

        def worker(slot: dict[str, str]) -> None:
            while True:
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                payload = self.folder / f"{self.name}_{index}.pkl"
                log = self.folder / f"{self.name}_{index}.log"
                with payload.open("wb") as handle:
                    pickle.dump((function, calls[index]), handle)
                # Array variables would give every task the same run ID.
                environment = {
                    **{
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith("SLURM_ARRAY_")
                    },
                    **slot,
                }
                with log.open("w", encoding="utf-8") as output:
                    result = subprocess.run(
                        [sys.executable, "-c", _WORKER, str(payload)],
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                state = "done" if result.returncode == 0 else "FAILED"
                if result.returncode:
                    failures[index] = log
                print(
                    f"[local] {self.name} {index + 1}/{len(calls)} {state}: {log}",
                    flush=True,
                )

        threads = [threading.Thread(target=worker, args=(slot,)) for slot in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            logs = ", ".join(str(failures[index]) for index in sorted(failures))
            raise CommandError(
                f"{len(failures)} of {len(calls)} local {self.name} tasks failed; "
                f"see {logs}"
            )
        # Like SLURM array tasks, the jobs of one call share an ID prefix.
        return [LocalJob(f"local-{self.name}_{index}") for index in range(len(calls))]


EXECUTORS = {"slurm": create_slurm_executor, "local": LocalExecutor}
