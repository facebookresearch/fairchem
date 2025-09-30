from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from fairchem.core.launchers.cluster.ray_cluster import RayCluster

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner
    from fairchem.core.launchers.api import SchedulerConfig, SlurmConfig


def ray_on_slurm_launch(config: DictConfig):
    scheduler_config: SchedulerConfig = config.job.scheduler
    slurm_config: SlurmConfig = scheduler_config.slurm
    runner: Runner = hydra.utils.instantiate(config.runner)
    cluster = RayCluster(log_dir=Path(config.job.run_dir))
    cluster_reqs = {
        "slurm_account": slurm_config.account,
        "slurm_qos": slurm_config.qos,
        "timeout_min": slurm_config.timeout_hr * 60,
        "mem_gb": slurm_config.mem_gb,
    }
    try:
        cluster.start_head(
            requirements=cluster_reqs | {"cpus_per_task": slurm_config.cpus_per_task},
            executor="slurm",
        )
        logging.info("Ray head started")

        # allocate the a ray cluster that is the same size and resources as the slurm job
        # todo, if multiple, launch multiple jobs instead of 1?
        cluster.start_workers(
            1,
            requirements=cluster_reqs
            | {
                "nodes": scheduler_config.num_nodes,
                "gpus_per_task": scheduler_config.ranks_per_node,
                "cpus_per_task": slurm_config.cpus_per_task
                * scheduler_config.ranks_per_node,
                "tasks_per_node": 1,
            },
        )
        logging.info("Ray workers started")

        # launch a payload on ray, move this to run on the head node
        # add a gpu for now
        cluster.submit_driver(
            runner.run,
            requirements=cluster_reqs
            | {
                "cpus_per_task": slurm_config.cpus_per_task,
                "gpus_per_task": 1,
                "tasks_per_node": 1,
            },
            executor="slurm",
            block=True,
        )
    finally:
        logging.info("Ray job finished, shutting down cluster ...")
        # TODO find way to shutdown without the `block` above? (maybe have some socket and signal between head/worker/driver)
        cluster.shutdown()
