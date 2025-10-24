"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Optional

import clusterscope

from fairchem.core.common.utils import (
    StrEnum,
    get_commit_hash,
    get_timestamp_uid,
)

ALLOWED_TOP_LEVEL_KEYS = {"job", "runner", "reducer"}

LOG_DIR_NAME = "logs"
CHECKPOINT_DIR_NAME = "checkpoints"
RESULTS_DIR = "results"
CONFIG_FILE_NAME = "canonical_config.yaml"
PREEMPTION_STATE_DIR_NAME = "preemption_state"


class SchedulerType(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class RunType(StrEnum):
    RUN = "run"
    REDUCE = "reduce"


class DistributedInitMethod(StrEnum):
    TCP = "tcp"
    FILE = "file"


@dataclass
class SlurmConfig:
    mem_gb: int = 80
    timeout_hr: int = 168
    cpus_per_task: int = 8
    partition: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    qos: Optional[str] = None  # omegaconf in python 3.9 does not backport annotations
    account: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    additional_parameters: Optional[dict] = None


@dataclass
class RayClusterConfig:
    head_gpus: int = 0


@dataclass
class SchedulerConfig:
    mode: SchedulerType = SchedulerType.LOCAL
    distributed_init_method: DistributedInitMethod = DistributedInitMethod.TCP
    ranks_per_node: int = 1
    num_nodes: int = 1
    num_array_jobs: int = 1
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig())
    # if not None, will launch a ray cluster on slurm instead of using submitit directly to launch the job
    use_ray: bool = False
    ray_cluster: RayClusterConfig = field(default_factory=lambda: RayClusterConfig())


@dataclass
class SlurmEnv:
    # reflects the job_id given by submitit (slurm id with array job id and array task id if they exist)
    job_id: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # reflects SLURM_JOB_ID only
    raw_job_id: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # SLURM_ARRAY_JOB_ID
    array_job_id: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # SLURM_ARRAY_TASK_ID
    array_task_id: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # reflects SLURM_RESTART_COUNT env variable
    restart_count: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )


@dataclass
class Metadata:
    # read-only metadata about the job, not user inputs
    commit: str
    log_dir: str
    checkpoint_dir: str
    results_dir: str
    config_path: str
    preemption_checkpoint_dir: str
    cluster_name: str
    array_job_num: int = 0
    slurm_env: SlurmEnv = field(default_factory=lambda: SlurmEnv())


@dataclass
class JobConfig:
    run_name: str = field(
        default_factory=lambda: get_timestamp_uid() + uuid.uuid4().hex.upper()[0:4]
    )
    timestamp_id: str = field(default_factory=lambda: get_timestamp_uid())
    run_dir: str = field(default_factory=lambda: tempfile.TemporaryDirectory().name)
    device_type: DeviceType = DeviceType.CUDA
    debug: bool = False
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig)
    logger: Optional[dict] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    seed: int = 0
    deterministic: bool = False
    runner_state_path: Optional[str] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    # read-only metadata about the job, not user inputs
    metadata: Optional[Metadata] = (
        None  # omegaconf in python 3.9 does not backport annotations
    )
    graph_parallel_group_size: Optional[int] = None

    def __post_init__(self) -> None:
        self.run_dir = os.path.abspath(self.run_dir)
        try:
            cluster = clusterscope.cluster()
        except RuntimeError:
            cluster = ""
        self.metadata = Metadata(
            commit=get_commit_hash(),
            log_dir=os.path.join(self.run_dir, self.timestamp_id, LOG_DIR_NAME),
            checkpoint_dir=os.path.join(
                self.run_dir, self.timestamp_id, CHECKPOINT_DIR_NAME
            ),
            results_dir=os.path.join(self.run_dir, self.timestamp_id, RESULTS_DIR),
            config_path=os.path.join(self.run_dir, self.timestamp_id, CONFIG_FILE_NAME),
            preemption_checkpoint_dir=os.path.join(
                self.run_dir,
                self.timestamp_id,
                CHECKPOINT_DIR_NAME,
                PREEMPTION_STATE_DIR_NAME,
            ),
            cluster_name=cluster,
        )
