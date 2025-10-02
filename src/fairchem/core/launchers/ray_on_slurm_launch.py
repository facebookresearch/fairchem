from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import (
    assign_device_for_local_rank,
    setup_env_local_multi_gpu,
)
from fairchem.core.common.utils import setup_env_vars
from torch.distributed.elastic.utils.distributed import get_free_port
import hydra
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch.distributed as dist

from fairchem.core.launchers.cluster.ray_cluster import RayCluster
from fairchem.core.components.runner import Runner


if TYPE_CHECKING:
    from omegaconf import DictConfig
    from fairchem.core.launchers.api import SchedulerConfig, SlurmConfig


@ray.remote
class RaySPMDWorker:
    def __init__(
        self,
        job_config: DictConfig,
        runner_config: DictConfig,
        worker_id: int,
        world_size: int,
        device: str,
        gp_size: int | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
    ):
        self.runner_config = runner_config
        # master address and port is not passed in, initialize it here
        self.master_address = (
            ray.util.get_node_ip_address() if master_addr is None else master_addr
        )
        self.master_port = get_free_port() if master_port is None else master_port
        self.worker_id = worker_id
        self.device = device
        self.gp_size = gp_size
        self.world_size = world_size
        self.job_config = job_config
        setup_env_vars()
        self.distributed_setup = False

    def _distributed_setup(
        self,
        worker_id: int,
        world_size: int,
        master_address: str,
        master_port: int,
        device: str,
        gp_size: int | None,
    ):
        setup_env_local_multi_gpu(worker_id, master_port, master_address)
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=worker_id,
            world_size=world_size,
        )
        if gp_size is not None:
            gp_utils.setup_graph_parallel_groups(gp_size, backend)

    def get_master_address_and_port(self):
        return (self.master_address, self.master_port)

    def run(self):
        if not self.distributed_setup:
            # initialize distributed environment
            self._distributed_setup(
                worker_id=self.worker_id,
                world_size=self.world_size,
                master_address=self.master_address,
                master_port=self.master_port,
                device=self.device,
                gp_size=self.gp_size,
            )
            self.runner: Runner = hydra.utils.instantiate(self.runner_config)
            self.runner.job_config = self.job_config
            self.distributed_setup = True
        self.runner.run()


class SPMDController(Runner):
    # this is equivalent to the fairchem Submitit routine that runs the runner on every worker
    def __init__(self, job_config: DictConfig, runner_config: DictConfig):
        self.job_config = job_config
        self.runner_config = runner_config
        self.device = job_config.device_type.value
        self.world_size = (
            job_config.scheduler.num_nodes * job_config.scheduler.ranks_per_node
        )
        self.gp_group_size = job_config.graph_parallel_group_size
        self.ranks_per_node = job_config.scheduler.ranks_per_node
        self.num_nodes = job_config.scheduler.num_nodes

        bundle_gpus = {"GPU": self.ranks_per_node, "CPU": self.ranks_per_node * 8}
        placement_groups = []
        # first create one placement group for each node
        for _ in range(self.num_nodes):
            pg = ray.util.placement_group([bundle_gpus], strategy="STRICT_PACK")
            placement_groups.append(pg)
        ray.get(pg.ready())  # Wait for each placement group to be scheduled

        logging.info(f"{len(placement_groups)} placement groups are ready")
        rank0_worker = RaySPMDWorker.options(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
            ),
        ).remote(
            self.job_config,
            self.runner_config,
            0,
            self.world_size,
            self.device,
            self.gp_group_size,
            None,
            None,
        )
        master_addr, master_port = ray.get(
            rank0_worker.get_master_address_and_port.remote()
        )
        logging.info(f"Started rank0 on {master_addr}:{master_port}")
        self.workers = [rank0_worker]

        # next place all ranks in order and pack them on placement groups
        # ie: rank0-7 -> placement group 0, 8->15 -> placement group 1 etc.
        for pg_idx, pg in enumerate(placement_groups):
            print(f"Launching workers for placement group {pg_idx} (Node {pg_idx})")

            for gpu_rank_on_node in range(self.ranks_per_node):
                if pg_idx == 0 and gpu_rank_on_node == 0:
                    continue
                # Each actor requests 1 GPU and uses the specific placement group
                actor = RaySPMDWorker.options(
                    num_gpus=1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                        placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
                    ),
                ).remote(
                    self.job_config,
                    self.runner_config,
                    pg_idx * self.ranks_per_node + gpu_rank_on_node,
                    self.world_size,
                    self.device,
                    self.gp_group_size,
                    master_addr,
                    master_port,
                )
                self.workers.append(actor)

    def run(self):
        logging.info("Running RaySPMDWrapper payload ...")
        futures = [w.run.remote() for w in self.workers]
        ray.get(futures)

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass


def ray_entrypoint(runner_config: DictConfig):
    runner = hydra.utils.instantiate(runner_config, _recursive_=False)
    runner.run()


def ray_on_slurm_launch(config: DictConfig, log_dir: str):
    scheduler_config: SchedulerConfig = config.job.scheduler
    slurm_config: SlurmConfig = scheduler_config.slurm
    cluster = RayCluster(log_dir=Path(log_dir))
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
            ray_entrypoint,
            requirements=cluster_reqs
            | {
                "cpus_per_task": slurm_config.cpus_per_task,
                "gpus_per_task": 1,
                "tasks_per_node": 1,
            },
            executor="slurm",
            block=True,
            runner_config=config.runner,
        )
    finally:
        logging.info("Ray job finished, shutting down cluster ...")
        # TODO find way to shutdown without the `block` above? (maybe have some socket and signal between head/worker/driver)
        cluster.shutdown()
