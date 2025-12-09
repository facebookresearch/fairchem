from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from submitit import AutoExecutor
from submitit.core.utils import JobPaths, cloudpickle_dump
from submitit.helpers import Checkpointable, DelayedSubmission
from submitit.slurm.slurm import SlurmJobEnvironment
from torch.distributed.elastic.utils.distributed import get_free_port

from fairchem.core.common import distutils
from fairchem.core.common.gp_utils import setup_graph_parallel_groups
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.utils import (
    setup_env_vars,
    setup_logging,
)
from fairchem.core.launchers.api import (
    DeviceType,
    JobConfig,
    RunType,
    SchedulerType,
    SlurmEnv,
)

if TYPE_CHECKING:
    from fairchem.core.components.reducer import Reducer
    from fairchem.core.components.runner import Runner


def _get_slurm_env() -> SlurmEnv:
    slurm_job_env = SlurmJobEnvironment()
    try:
        slurm_env = SlurmEnv(
            job_id=slurm_job_env.job_id,
            raw_job_id=slurm_job_env.raw_job_id,
            array_job_id=slurm_job_env.array_job_id,
            array_task_id=slurm_job_env.array_task_id,
            restart_count=os.environ.get("SLURM_RESTART_COUNT"),
        )
    except KeyError:
        # slurm environment variables are undefined, running locally
        slurm_env = SlurmEnv()

    return slurm_env


def map_job_config_to_dist_config(job_cfg: JobConfig) -> dict:
    scheduler_config = job_cfg.scheduler
    return {
        "world_size": scheduler_config.num_nodes * scheduler_config.ranks_per_node,
        "distributed_backend": (
            "gloo" if job_cfg.device_type == DeviceType.CPU else "nccl"
        ),
        "submit": scheduler_config.mode == SchedulerType.SLURM,
        "cpu": job_cfg.device_type == DeviceType.CPU,
        "init_method": scheduler_config.distributed_init_method,
        # for distributed shared file initialization
        "shared_file_dir": os.path.join(job_cfg.run_dir, job_cfg.timestamp_id),
        "array_job_num": job_cfg.metadata.array_job_num,
    }


def remove_runner_state_from_submission(log_folder: str, job_id: str) -> None:
    # (HACK) Decouple the job from the runner state by manually modifying it
    # this ensures the saved runner state is not re-submitted in the event of a node failure
    # ie: if the job was started at state t=T, a requeue during node failure would resubmit the job
    # starting at state t=T again without calling the checkpoint callback, losing all progress in between.
    job_path = JobPaths(folder=log_folder, job_id=job_id)
    if os.path.isfile(job_path.submitted_pickle):
        submission_obj = DelayedSubmission.load(job_path.submitted_pickle)
        submission_obj.args[0].job.runner_state_path = None
        cloudpickle_dump(submission_obj, job_path.submitted_pickle)


def runner_wrapper(config: DictConfig, run_type: RunType = RunType.RUN):
    # This is needed when using elastic_launch for local runs since it looks for
    # the __name__ attribute of the function, Submitit.__call__ does not have one
    SlurmSPMDProgram()(config, run_type)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_deterministic_mode() -> None:
    # this is required for full cuda deterministic mode
    logging.info("Setting deterministic mode!")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class SlurmSPMDProgram(Checkpointable):
    """
    Entrypoint for a SPMD program launched via submitit on slurm.
    This assumes all ranks run the identical copy of this code
    """

    def __init__(self) -> None:
        self.config = None
        self.runner = None
        self.reducer = None

    def __call__(
        self, dict_config: DictConfig, run_type: RunType = RunType.RUN
    ) -> None:
        self.config = dict_config
        self.run_type = run_type
        # modify the config metadata to add slurm info if they exist
        self.config.job.metadata.slurm_env = _get_slurm_env()

        setup_env_vars()
        setup_logging()

        dist_config = map_job_config_to_dist_config(self.config.job)
        logging.info("Setting up distributed backend...")
        distutils.setup(dist_config)
        distutils.synchronize()
        if (
            distutils.is_master()
            and self.config.job.scheduler.mode == SchedulerType.SLURM
        ):
            # this pickle file is shared across all processes so can only modify this on the main rank
            remove_runner_state_from_submission(
                dict_config.job.metadata.log_dir,
                self.config.job.metadata.slurm_env.job_id,
            )

        if self.config.job.graph_parallel_group_size is not None:
            logging.info("Setting up graph parallel...")
            setup_graph_parallel_groups(
                self.config.job.graph_parallel_group_size,
                dist_config["distributed_backend"],
            )

        self._init_logger()

        _set_seeds(self.config.job.seed)
        if self.config.job.deterministic:
            _set_deterministic_mode()

        if run_type == RunType.RUN:
            logging.info("Calling runner.run() ...")
            self.runner: Runner = hydra.utils.instantiate(self.config.runner)
            self.runner.job_config = self.config.job
            # must call resume state AFTER the runner has been initialized
            self.runner.load_state(self.config.job.runner_state_path)
            self.runner.run()
        elif run_type == RunType.REDUCE:
            logging.info("Calling reducer.reduce() ...")
            self.reducer: Reducer = hydra.utils.instantiate(self.config.reducer)
            self.reducer.job_config = self.config.job
            self.reducer.runner_config = self.config.runner
            # must call resume state AFTER the runner has been initialized
            self.reducer.load_state(self.config.job.runner_state_path)
            self.reducer.reduce()
        else:
            raise ValueError(f"run type {run_type} is not recognized!")

        distutils.cleanup()

    def _init_logger(self) -> None:
        if (
            self.config.job.logger
            and distutils.is_master()
            and not self.config.job.debug
            and self.config.job.metadata.array_job_num == 0
        ):
            # get a partial function from the config and instantiate wandb with it
            # currently code assumes that we only use the WandBSingletonLogger
            logger_initializer = hydra.utils.instantiate(self.config.job.logger)
            simple_config = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            )
            logger_initializer(
                config=simple_config,
                run_id=self.config.job.timestamp_id,
                run_name=self.config.job.run_name,
                log_dir=self.config.job.metadata.log_dir,
            )

    def checkpoint(self, *args, **kwargs) -> DelayedSubmission:
        logging.error("Submitit checkpointing callback is triggered")
        save_path = self.config.job.metadata.preemption_checkpoint_dir
        cfg_copy = self.config.copy()
        # only assign if the save was successful
        cfg_copy.job.runner_state_path = None

        if (
            self.run_type == RunType.RUN
            and self.runner.save_state(save_path, is_preemption=True)
        ) or (
            self.run_type == RunType.REDUCE
            and self.reducer.save_state(save_path, is_preemption=True)
        ):
            cfg_copy.job.runner_state_path = save_path

        if WandBSingletonLogger.initialized():
            WandBSingletonLogger.get_instance().mark_preempting()
        logging.info(
            f"Submitit checkpointing callback is completed, resuming with use the following state: {save_path}"
        )
        return DelayedSubmission(SlurmSPMDProgram(), cfg_copy)


def slurm_launch(cfg: DictConfig, log_dir: str) -> None:
    scheduler_cfg = cfg.job.scheduler
    executor = AutoExecutor(folder=log_dir, slurm_max_num_timeout=3)
    executor.update_parameters(
        name=cfg.job.run_name,
        mem_gb=scheduler_cfg.slurm.mem_gb,
        timeout_min=scheduler_cfg.slurm.timeout_hr * 60,
        slurm_partition=scheduler_cfg.slurm.partition,
        gpus_per_node=scheduler_cfg.ranks_per_node,
        cpus_per_task=scheduler_cfg.slurm.cpus_per_task,
        tasks_per_node=scheduler_cfg.ranks_per_node,
        nodes=scheduler_cfg.num_nodes,
        slurm_qos=scheduler_cfg.slurm.qos,
        slurm_account=scheduler_cfg.slurm.account,
        slurm_additional_parameters=scheduler_cfg.slurm.additional_parameters,
    )
    if scheduler_cfg.num_array_jobs == 1:
        job = executor.submit(SlurmSPMDProgram(), cfg)
        logging.info(
            f"Submitted job id: {cfg.job.timestamp_id}, slurm id: {job.job_id}, logs: {cfg.job.metadata.log_dir}"
        )
        jobs = [job]
    elif scheduler_cfg.num_array_jobs > 1:
        executor.update_parameters(
            slurm_array_parallelism=scheduler_cfg.num_array_jobs,
        )

        jobs = []
        with executor.batch():
            for job_number in range(scheduler_cfg.num_array_jobs):
                _cfg = cfg.copy()
                _cfg.job.metadata.array_job_num = job_number
                job = executor.submit(SlurmSPMDProgram(), _cfg)
                jobs.append(job)
        logging.info(f"Submitted {len(jobs)} jobs: {jobs[0].job_id.split('_')[0]}")

    if "reducer" in cfg:
        job_id = jobs[0].job_id.split("_")[0]
        executor.update_parameters(
            name=f"{cfg.job.run_name}_reduce",
            # set a single node, or do we want the same config as the Runner or a separate JobConfig
            nodes=1,
            slurm_dependency=f"afterok:{job_id}",
            slurm_additional_parameters={
                "kill-on-invalid-dep": "yes"
            },  # kill the reducer if run fails
        )
        executor.submit(SlurmSPMDProgram(), cfg, RunType.REDUCE)


def local_launch(cfg: DictConfig, log_dir: str):
    """
    Launch locally with torch elastic (for >1 workers) or just single process
    """
    scheduler_cfg = cfg.job.scheduler
    if scheduler_cfg.ranks_per_node > 1:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=scheduler_cfg.ranks_per_node,
            rdzv_backend="c10d",
            max_restarts=0,
            rdzv_endpoint=f"127.0.0.1:{get_free_port()}",
        )
        elastic_launch(launch_config, runner_wrapper)(cfg)
        if "reducer" in cfg:
            elastic_launch(launch_config, runner_wrapper)(cfg, RunType.REDUCE)
    else:
        logging.info("Running in local mode without elastic launch")
        distutils.setup_env_local()
        runner_wrapper(cfg)
        if "reducer" in cfg:
            runner_wrapper(cfg, RunType.REDUCE)
