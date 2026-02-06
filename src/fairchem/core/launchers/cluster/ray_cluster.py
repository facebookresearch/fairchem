# ruff: noqa
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Callable, Optional, TypeVar
import uuid
from contextlib import closing
from pathlib import Path

import psutil
from fairchem.core.common.distutils import os_environ_get_or_throw
import submitit
from submitit.helpers import Checkpointable, DelayedSubmission


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def scancel(job_ids: list[str]):
    """
    Cancel the SLURM jobs with the given job IDs.

    This function takes a list of job IDs.

    Args:
        job_ids (List[str]): A list of job IDs to cancel.
    """
    root_ids = list(set([i.split("_", maxsplit=2)[0] for i in job_ids]))
    subprocess.check_call(["scancel"] + root_ids)


start_ip_pattern = r"ray start --address='([0-9\.]+):([0-9]+)'"

PayloadReturnT = TypeVar("PayloadReturnT")


def mk_symlinks(target_dir: Path, job_type: str, paths: submitit.core.utils.JobPaths):
    """Create symlinks for the job's stdout and stderr in the target directory with a nicer name."""
    (target_dir / f"{job_type}.err").symlink_to(paths.stderr)
    (target_dir / f"{job_type}.out").symlink_to(paths.stdout)


@dataclasses.dataclass
class HeadInfo:
    """
    information about the head node that we can share to workers
    """

    hostname: Optional[str] = None
    port: Optional[int] = None
    temp_dir: Optional[str] = None


class RayClusterState:
    """
    This class is responsible for managing the state of the Ray cluster. It is useful to keep track
    of the head node and the workers, and to make sure they are all ready before starting the payload.

    It relies on storing info in a rendezvous directory so they can be shared async between jobs.

    Args:
        rdv_dir (Path): The directory where the rendezvous information will be stored. Defaults to ~/.fairray.
        cluster_id (str): A unique identifier for the cluster. Defaults to a random UUID. You only want to set this if you want to connect to an existing cluster.
    """

    def __init__(
        self,
        rdv_dir: Optional[Path] = None,
        cluster_id: Optional[str] = None,
    ):
        self.rendezvous_rootdir = (
            rdv_dir if rdv_dir is not None else (Path.home() / ".fairray")
        )
        self._cluster_id = (
            uuid.uuid4().hex if cluster_id is None else cluster_id
        )  # maybe use something more readable
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cluster_id(self) -> str:
        """Returns the unique identifier for the cluster."""
        return self._cluster_id

    @property
    def rendezvous_dir(self) -> Path:
        """Returns the path to the directory where the rendezvous information is stored."""
        return self.rendezvous_rootdir / self.cluster_id

    @property
    def jobs_dir(self) -> Path:
        """Returns the path to the directory where job information is stored."""
        return self.rendezvous_dir / "jobs"

    @property
    def _head_json(self) -> Path:
        """Returns the path to the JSON file containing head node information."""
        return self.rendezvous_dir / "head.json"

    def is_head_ready(self) -> bool:
        """Checks if the head node information is available and ready."""
        return self._head_json.exists()

    def head_info(self) -> Optional[HeadInfo]:
        """
        Retrieves the head node information from the stored JSON file.

        Returns:
            Optional[HeadInfo]: The head node information if available, otherwise None.
        """
        try:
            with self._head_json.open("r") as f:
                return HeadInfo(**json.load(f))
        except Exception as ex:
            print(f"failed to load head info: {ex}. Maybe it's not ready yet?")
            return None

    def save_head_info(self, head_info: HeadInfo):
        """
        Saves the head node information to a JSON file.

        Args:
            head_info (HeadInfo): The head node information to save.
        """
        with self._head_json.open("w") as f:
            json.dump(dataclasses.asdict(head_info), f)

    def reset_state(self):
        """Resets the head node information by removing the stored JSON file, useful for preemption resumes"""
        if self._head_json.exists():
            self._head_json.unlink()

    def clean(self):
        """Removes the rendezvous directory and all its contents."""
        shutil.rmtree(self.rendezvous_dir)

    def add_job(self, job: submitit.Job):
        """
        Adds a job to the jobs directory by creating a JSON file with the job's information.

        Args:
            job (submitit.Job): The job to add.
        """
        with (self.jobs_dir / f"{job.job_id}.json").open("w") as f:
            json.dump(
                {
                    "job_id": job.job_id,
                },
                fp=f,
            )

    def list_job_ids(self) -> list[str]:
        """Lists all job IDs stored in the jobs directory."""
        return [f.stem for f in self.jobs_dir.iterdir()]


class CheckpointableRayJob(Checkpointable):
    """
    A checkpointable Ray job that can restart itself upon failure or preemption.
    It gang schedules the head and worker nodes together to keep preemption logic simple.
    """

    def __init__(
        self,
        cluster_state: RayClusterState,
        worker_wait_timeout_seconds: int,
        payload: Optional[Callable[..., PayloadReturnT]],
        **kwargs,
    ):
        self.cluster_state = cluster_state
        self.worker_wait_timeout_seconds = worker_wait_timeout_seconds
        self.payload = payload
        self.kwargs = kwargs

    def __call__(self):
        # if we are the head node, start head
        # the worker nodes need to get the head address from the cluster state
        node_id = int(os_environ_get_or_throw("SLURM_NODEID"))
        if node_id == 0:
            _ray_head_script(
                cluster_state=self.cluster_state,
                worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
                payload=self.payload,
                **self.kwargs,
            )
        else:
            worker_script(
                cluster_state=self.cluster_state,
                worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
            )

    def checkpoint(self) -> DelayedSubmission:
        logging.error(f"CheckpointableRayJob checkpointing callback is triggered")
        # reset head info so that on restart we can create a new head
        self.cluster_state.reset_state()
        job = CheckpointableRayJob(
            cluster_state=self.cluster_state,
            worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
            payload=self.payload,
            **self.kwargs,
        )
        return DelayedSubmission(job)


def _ray_head_script(
    cluster_state: RayClusterState,
    worker_wait_timeout_seconds: int,
    payload: Optional[Callable[..., PayloadReturnT]] = None,
    **kwargs,
):
    """Start the head node of the Ray cluster on slurm."""
    hostname = socket.gethostname()
    head_env = os.environ.copy()
    num_cpus = os.environ.get("SLURM_CPUS_ON_NODE", 1)
    num_gpus = os.environ.get("SLURM_GPUS_ON_NODE", 0)
    # using 0 as the port for the head will make ray search for an open port instead of
    # always using the same one.
    port = find_free_port()
    head_env["RAY_ADDRESS"] = f"{hostname}:{port}"
    head_env["RAY_gcs_server_request_timeout_seconds"] = str(
        worker_wait_timeout_seconds
    )
    print(f"host {hostname}:{port}")
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        # ray workers have the same tempdir name (even on a different host)
        # as the head. This is a problem when we use  /scratch/slurm_tmpdir/JOBID as
        # the tempdir of the head job will not be accessible/visible from other workers if they
        # are scheduled on the same host. We are forced to use a different tempdir than /scratch
        # TODO ideally, we would still have a /scratch dir that everyone can share.
        process = subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--port={port}",
                f"--temp-dir={temp_dir}",
                "--num-cpus",
                f"{num_cpus}",
                "--num-gpus",
                f"{num_gpus}",
                "--dashboard-host=0.0.0.0",
            ],
            env=head_env,
            stdout=subprocess.PIPE,
            text=True,
        )
        started = False
        for line in process.stdout:
            if "ray start --address=" in line:
                # this is a bit flaky, we search the stdout of the head job to
                # find this specific message and extract the address, it might be
                # better to not rely on ray printing this as it might change outside of our control.
                # Search for the pattern
                started = True
        assert (
            started
        ), "couldn't find head address in stdout. Check head.err for details"
        print(f"Head started, ip: {hostname}:{port} ({cluster_state.cluster_id})")
        info = HeadInfo(hostname=hostname, port=int(port), temp_dir=temp_dir)
        cluster_state.save_head_info(info)
        os.environ.update(head_env)
        if payload is not None:
            payload(**kwargs)
        else:
            while True:
                # practically, we should wait from driver signal to die here
                time.sleep(60)


def worker_script(
    cluster_state: RayClusterState,
    worker_wait_timeout_seconds: int,
    start_wait_time_seconds: int = 60,  # TODO pass this around properly
):
    """start an array of worker nodes for the Ray cluster on slurm. Waiting on the head node first."""
    print(f"Waiting for head node. {cluster_state.cluster_id}")
    while not cluster_state.is_head_ready():
        # wait for head to have started
        time.sleep(5)
    print("Head node found.")
    head_info = cluster_state.head_info()
    assert head_info is not None, "something went wrong getting head information."
    worker_env = os.environ.copy()
    worker_env["RAY_ADDRESS"] = f"{head_info.hostname}:{head_info.port}"
    worker_env["RAY_gcs_server_request_timeout_seconds"] = str(
        worker_wait_timeout_seconds
    )
    worker_env["RAY_raylet_start_wait_time_s"] = str(start_wait_time_seconds)
    num_cpus = os.environ.get("SLURM_CPUS_ON_NODE", 1)
    num_gpus = os.environ.get("SLURM_GPUS_ON_NODE", 0)

    try:
        subprocess.run(
            [
                "ray",
                "start",
                "--address",
                "auto",
                "--block",
                "--num-cpus",
                f"{num_cpus}",
                "--num-gpus",
                f"{num_gpus}",
            ],
            env=worker_env,
            check=False,
        )
    finally:
        if head_info.temp_dir:
            shutil.rmtree(Path(head_info.temp_dir))


# TODO deal with ports better: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats
# TODO: reqs are just dicts, maybe we want to be more specific (in particular for qos/partition)
# TODO: need better naming too
# TODO: better log messages
# TODO checkpointing to recover worker nodes after timeout/preemption https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md
# TODO have a ray autoscaler nodeprovider based on this, e.g. https://github.com/TingkaiLiu/Ray-SLURM-autoscaler/blob/main/slurm/node_provider.py
class RayCluster:
    """
    A RayCluster offers tools to start a Ray cluster (head and wokers) on slurm with the correct settings.

    args:

    log_dir: Path to the directory where logs will be stored. Defaults to "raycluster_logs" in the working directory. All slurm logs will go there,
    and it also creates symlinks to the stdout/stderr of each jobs with nicer name (head, worker_0, worker_1, ..., driver_0, etc). There interesting
    logs will be in the driver_N.err file, you should tail that.
    rdv_dir: Path to the directory where the rendezvous information will be stored. Defaults to ~/.fairray. Useful if you are trying to recover an existing cluster.
    cluster_id: A unique identifier for the cluster. Defaults to a random UUID. You only want to set this if you want to connect to an existing cluster.
    worker_wait_timeout_seconds (int): The number of seconds ray will wait for a worker to be ready before giving up. Defaults to 60 seconds. If you are scheduling
        workers in a queue that takes time for allocation, you might want to increase this otherwise your ray payload will fail, not finding resources.

    """

    def __init__(
        self,
        log_dir: Path = Path("raycluster_logs"),
        rdv_dir: Optional[Path] = None,
        cluster_id: Optional[str] = None,
        worker_wait_timeout_seconds: int = 60,
    ):
        self.state = RayClusterState(rdv_dir, cluster_id)
        print(f"cluster {self.state.cluster_id}")
        self.output_dir = log_dir
        self.log_dir = Path(log_dir) / self.state.cluster_id
        self.state.rendezvous_dir.mkdir(parents=True, exist_ok=True)
        self.worker_wait_timeout_seconds = worker_wait_timeout_seconds
        self.is_shutdown = False
        self.num_worker_groups = 0
        self.num_drivers = 0
        self.head_started = False
        self.jobs: list[submitit.Job] = []
        print(f"logs will be in {self.log_dir.resolve()}")

    def start_head_and_workers(
        self,
        requirements: dict[str, int | str],
        name: str = "default",
        executor: str = "slurm",
        payload: Optional[Callable[..., PayloadReturnT]] = None,
        **kwargs,
    ):
        assert not self.head_started, "head already started"
        # start the head node
        self.head_started = True
        s_executor = submitit.AutoExecutor(
            folder=str(self.log_dir),
            cluster=executor,
        )
        s_executor.update_parameters(
            name=f"ray_{name}_{self.state.cluster_id}",
            **requirements,
        )
        ray_job = CheckpointableRayJob(
            cluster_state=self.state,
            worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
            payload=payload,
            **kwargs,
        )
        slurm_job = s_executor.submit(ray_job)
        self.state.add_job(slurm_job)
        self.jobs.append(slurm_job)
        mk_symlinks(self.log_dir, "job", slurm_job.paths)
        print("slurm job id:", slurm_job.job_id)
        return slurm_job.job_id

    def start_head(
        self,
        requirements: dict[str, int | str],
        name: str = "default",
        executor: str = "slurm",
        payload: Optional[Callable[..., PayloadReturnT]] = None,
        **kwargs,
    ) -> str:
        """
        Start the head node of the Ray cluster on slurm. You should do this first. Interesting requirements: qos, partition, time, gpus, cpus-per-task, mem-per-gpu, etc.
        """
        assert not self.head_started, "head already started"
        # start the head node
        self.head_started = True
        s_executor = submitit.AutoExecutor(
            folder=str(self.log_dir),
            cluster=executor,
        )
        s_executor.update_parameters(
            name=f"ray_head_{name}_{self.state.cluster_id}",
            **requirements,
        )
        head_job = s_executor.submit(
            _ray_head_script,
            self.state,
            self.worker_wait_timeout_seconds,
            payload,
            **kwargs,
        )
        self.state.add_job(head_job)
        self.jobs.append(head_job)
        mk_symlinks(self.log_dir, "head", head_job.paths)
        print("head slurm job id:", head_job.job_id)
        return head_job.job_id

    def start_workers(
        self,
        num_workers: int,
        requirements: dict[str, int | str],
        name: str = "default",
        executor: str = "slurm",
    ) -> list[str]:
        """
        Start an array of worker nodes of the Ray cluster on slurm. You should do this after starting a head.
        Interesting requirements: qos, partition, time, gpus, cpus-per-task, mem-per-gpu, etc.
        You can call this multiple times to start an heterogeneous cluster.
        """
        # start the workers
        s_executor = submitit.AutoExecutor(folder=str(self.log_dir), cluster=executor)
        s_executor.update_parameters(
            name=f"ray_worker_{name}_{self.num_worker_groups}_{self.state.cluster_id}",  # TODO name should probably include more details (cluster_id)
            **requirements,
        )

        jobs = []
        with s_executor.batch():  # TODO set slurm array max parallelism here, because we really want all jobs to be scheduled at the same time
            for i in range(num_workers):
                jobs.append(
                    s_executor.submit(
                        worker_script,
                        self.state,
                        self.worker_wait_timeout_seconds,
                    )
                )

        for idx, j in enumerate(jobs):
            mk_symlinks(self.log_dir, f"worker_{self.num_worker_groups}_{idx}", j.paths)
        print("workers slurm job ids:", [job.job_id for job in jobs])
        for j in jobs:
            self.state.add_job(j)
            self.jobs.append(j)
        self.num_worker_groups += 1
        return [job.job_id for job in jobs]

    def shutdown(self):
        """
        Cancel all slurms jobs and get rid of rdv directory.
        """
        self.is_shutdown = True
        scancel(self.state.list_job_ids())
        kill_proc_tree(
            os.getpid(), including_parent=False
        )  # kill local job started by submitit as subprocess TODO that's not going to work when this is not the main process (e.g. recovering on cli)
        self.state.clean()
        print(f"cluster {self.state.cluster_id} shutdown")

    def __enter__(self):
        # only use as a context if you have something blocking waiting on the driver
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
