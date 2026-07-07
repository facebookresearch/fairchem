"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# ruff: noqa
from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from typing import TYPE_CHECKING, Callable, Optional, TypeVar
import uuid
from contextlib import closing, suppress
from pathlib import Path

import psutil
from fairchem.core.common.distutils import os_environ_get_or_throw
import submitit
from submitit.helpers import Checkpointable, DelayedSubmission

if TYPE_CHECKING:
    from fairchem.core.launchers.api import RayMetricsConfig

logger = logging.getLogger(__name__)


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def find_free_port(
    preferred: int = 0,
    num_random_attempts: int = 10,
    bind_address: str = "",
) -> int:
    """
    Find an available port.

    Random sampling in the ephemeral range is used (instead of relying on the
    OS to pick) so that multiple raylets / object managers / client servers
    started in tight succession on the same node don't collide.

    Args:
        preferred: Try this port first. If 0 or unavailable, try random ports.
        num_random_attempts: Number of random ports to try before letting OS pick.
        bind_address: Interface to probe. Default ``""`` (all interfaces) for
            Ray ports that must be reachable from other nodes. Pass
            ``"127.0.0.1"`` for ports that only need local reachability.
    """
    if preferred:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((bind_address, preferred))
                return preferred
        except OSError:
            pass

    # Try random ports in the ephemeral range (49152-65535)
    for _ in range(num_random_attempts):
        port = random.randint(49152, 65535)
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((bind_address, port))
                return port
        except OSError:
            continue

    # Fall back to letting the OS pick
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((bind_address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


RAY_DEFAULT_DASHBOARD_PORT = 8265

# Default directory for per-cluster rendezvous / head.json files.
DEFAULT_HEAD_FILE_DIR = Path.home() / ".fairray"

# Name of the human/machine readable file written to the log directory with the
# Ray head and dashboard connection details.
RAY_CLUSTER_INFO_FILENAME = "ray_cluster_info.json"

# Pinned binary versions used only for the explicit metrics auto-download fallback.
# Users who need specific versions should install the binaries themselves.
PROMETHEUS_AUTODOWNLOAD_VERSION = "3.1.0"
GRAFANA_AUTODOWNLOAD_VERSION = "11.4.0"

# How long to wait for Ray to generate the Prometheus/Grafana config files under
# the session dir before giving up on starting the metrics servers.
METRICS_CONFIG_WAIT_SECONDS = 60


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
    head_nodename: Optional[str] = (
        None  # SLURM node name (or gethostname) of the head machine
    )
    port: Optional[int] = None  # Ray GCS port
    client_port: Optional[int] = None  # Ray Client server port (if enabled)
    dashboard_port: Optional[int] = None  # Ray dashboard port
    prometheus_port: Optional[int] = None  # Prometheus server port (if metrics enabled)
    grafana_port: Optional[int] = None  # Grafana server port (if metrics enabled)
    temp_dir: Optional[str] = None
    namespace_serve_fairchem: Optional[str] = None

    @property
    def ray_address(self) -> Optional[str]:
        """The Ray head address (``hostname:gcs_port``) or None if unknown."""
        if self.hostname and self.port:
            return f"{self.hostname}:{self.port}"
        return None

    @property
    def dashboard_url(self) -> Optional[str]:
        """The Ray dashboard URL or None if unknown."""
        if self.hostname and self.dashboard_port:
            return f"http://{self.hostname}:{self.dashboard_port}"
        return None

    @property
    def prometheus_url(self) -> Optional[str]:
        """
        The Prometheus URL (via the SSH-tunnel-friendly node name) or None if
        metrics are not running.
        """
        if self.head_nodename and self.prometheus_port:
            return f"http://{self.head_nodename}:{self.prometheus_port}"
        return None

    @property
    def grafana_url(self) -> Optional[str]:
        """
        The Grafana URL (via the SSH-tunnel-friendly node name) or None if
        metrics are not running.
        """
        if self.head_nodename and self.grafana_port:
            return f"http://{self.head_nodename}:{self.grafana_port}"
        return None


class RayClusterState:
    """
    This class is responsible for managing the state of the Ray cluster. It is useful to keep track
    of the head node and the workers, and to make sure they are all ready before starting the payload.

    It relies on storing info in a rendezvous directory so they can be shared async between jobs.

    Args:
        rdv_dir (Path): The directory where the rendezvous information will be stored. Defaults to ~/.fairray.
        cluster_id (str): A unique identifier for the cluster. Defaults to a random UUID. You only want to set this if you want to connect to an existing cluster.
        log_dir (Path): Job log directory where a human-facing copy of the Ray/dashboard
            connection details is written. If None, only the rendezvous ``head.json`` is written.
    """

    def __init__(
        self,
        rdv_dir: Optional[Path] = None,
        cluster_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ):
        self.rendezvous_rootdir = (
            rdv_dir if rdv_dir is not None else DEFAULT_HEAD_FILE_DIR
        )
        self._cluster_id = (
            uuid.uuid4().hex if cluster_id is None else cluster_id
        )  # maybe use something more readable
        self.log_dir = log_dir
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

    @property
    def cluster_info_file(self) -> Optional[Path]:
        """
        Path to the human-facing Ray/dashboard connection info file in the job
        log directory, or None if no log directory is configured.
        """
        if self.log_dir is None:
            return None
        return Path(self.log_dir) / RAY_CLUSTER_INFO_FILENAME

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
            logger.info(f"failed to load head info: {ex}. Maybe it's not ready yet?")
            return None

    def save_head_info(self, head_info: HeadInfo):
        """
        Saves the head node information to the rendezvous ``head.json`` and, if a
        job log directory is configured, also writes a human-facing copy with the
        Ray/dashboard connection details there.

        Both files are (re)written on every head start, including preemption
        resumes, so they always reflect the currently running cluster.

        Args:
            head_info (HeadInfo): The head node information to save.
        """
        with self._head_json.open("w") as f:
            json.dump(dataclasses.asdict(head_info), f)

        info_file = self.cluster_info_file
        if info_file is not None:
            info_file.parent.mkdir(parents=True, exist_ok=True)
            with info_file.open("w") as f:
                json.dump(
                    {
                        "cluster_id": self.cluster_id,
                        "hostname": head_info.hostname,
                        "head_nodename": head_info.head_nodename,
                        "ray_gcs_port": head_info.port,
                        "ray_address": head_info.ray_address,
                        "ray_client_port": head_info.client_port,
                        "dashboard_host": head_info.hostname,
                        "dashboard_port": head_info.dashboard_port,
                        "dashboard_url": head_info.dashboard_url,
                        "prometheus_port": head_info.prometheus_port,
                        "prometheus_url": head_info.prometheus_url,
                        "grafana_port": head_info.grafana_port,
                        "grafana_url": head_info.grafana_url,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Wrote Ray cluster info to {info_file}")

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
        temp_dir_template: Optional[str] = None,
        **kwargs,
    ):
        self.cluster_state = cluster_state
        self.worker_wait_timeout_seconds = worker_wait_timeout_seconds
        self.payload = payload
        self.temp_dir_template = temp_dir_template
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
                temp_dir_template=self.temp_dir_template,
                **self.kwargs,
            )
        else:
            worker_script(
                cluster_state=self.cluster_state,
                worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
                temp_dir_template=self.temp_dir_template,
            )

    def checkpoint(self) -> DelayedSubmission:
        logging.error(f"CheckpointableRayJob checkpointing callback is triggered")
        # reset head info so that on restart we can create a new head
        self.cluster_state.reset_state()
        job = CheckpointableRayJob(
            cluster_state=self.cluster_state,
            worker_wait_timeout_seconds=self.worker_wait_timeout_seconds,
            payload=self.payload,
            temp_dir_template=self.temp_dir_template,
            **self.kwargs,
        )
        return DelayedSubmission(job)


def _os_arch_for_download() -> tuple[str, str]:
    """Return (os_type, arch) strings used in Prometheus/Grafana release URLs."""
    os_type = platform.system().lower()  # 'linux' / 'darwin'
    machine = platform.machine().lower()
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    return os_type, arch


def _download_and_extract(url: str, dest_dir: str) -> Path:
    """
    Download a ``.tar.gz`` from ``url`` and extract it into ``dest_dir``.

    Returns the top-level extracted directory.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    tar_path = Path(dest_dir) / "download.tar.gz"
    logger.info(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tar:
        members = tar.getnames()
        tar.extractall(dest_dir)
    tar_path.unlink(missing_ok=True)
    top = members[0].split("/")[0]
    return Path(dest_dir) / top


def _download_prometheus(dest_dir: str) -> Optional[str]:
    """Download a static Prometheus release; return the binary path or None."""
    os_type, arch = _os_arch_for_download()
    ver = PROMETHEUS_AUTODOWNLOAD_VERSION
    name = f"prometheus-{ver}.{os_type}-{arch}"
    url = (
        "https://github.com/prometheus/prometheus/releases/"
        f"download/v{ver}/{name}.tar.gz"
    )
    binary = _download_and_extract(url, dest_dir) / "prometheus"
    return str(binary) if binary.exists() else None


def _download_grafana(dest_dir: str) -> Optional[tuple[str, str]]:
    """Download a Grafana OSS release; return (binary, homepath) or None."""
    os_type, arch = _os_arch_for_download()
    ver = GRAFANA_AUTODOWNLOAD_VERSION
    url = f"https://dl.grafana.com/oss/release/grafana-{ver}.{os_type}-{arch}.tar.gz"
    root = _download_and_extract(url, dest_dir)
    binary = root / "bin" / "grafana"
    if not binary.exists():
        binary = root / "bin" / "grafana-server"
    return (str(binary), str(root)) if binary.exists() else None


def _resolve_prometheus_binary(
    metrics_config: RayMetricsConfig, download_dir: str
) -> Optional[str]:
    """
    Resolve a Prometheus binary: explicit config path, then PATH, then (only if
    ``auto_download`` is set) an internet download. Returns None if unavailable.
    """
    if metrics_config.prometheus_binary:
        if Path(metrics_config.prometheus_binary).exists():
            return metrics_config.prometheus_binary
        logger.warning(
            f"Configured prometheus_binary {metrics_config.prometheus_binary!r} "
            "not found; falling back to PATH."
        )
    found = shutil.which("prometheus")
    if found:
        return found
    if metrics_config.auto_download:
        try:
            return _download_prometheus(f"{download_dir}/prometheus")
        except Exception as ex:
            logger.warning(f"Prometheus auto-download failed: {ex}")
    return None


def _grafana_homepath(binary: str, configured: Optional[str]) -> Optional[str]:
    """Locate a Grafana homepath (dir containing ``public/``) for an installed binary."""
    candidates = []
    if configured:
        candidates.append(configured)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(os.path.join(conda, "share", "grafana"))
    bin_dir = os.path.dirname(os.path.realpath(binary))
    candidates.append(os.path.join(bin_dir, "..", "share", "grafana"))
    candidates.append("/usr/share/grafana")
    for candidate in candidates:
        if Path(candidate, "public").is_dir():
            return os.path.abspath(candidate)
    return None


def _resolve_grafana(
    metrics_config: RayMetricsConfig, download_dir: str
) -> Optional[tuple[str, str]]:
    """
    Resolve a Grafana (binary, homepath): explicit config, then PATH, then (only
    if ``auto_download`` is set) an internet download. Returns None if unavailable.
    """
    binary = metrics_config.grafana_binary
    if binary and not Path(binary).exists():
        logger.warning(
            f"Configured grafana_binary {binary!r} not found; falling back to PATH."
        )
        binary = None
    if binary is None:
        binary = shutil.which("grafana") or shutil.which("grafana-server")
    if binary:
        homepath = _grafana_homepath(binary, metrics_config.grafana_homepath)
        if homepath:
            return binary, homepath
        logger.warning(
            "Found a grafana binary but could not locate its homepath (a dir with "
            "a public/ subfolder); set metrics.grafana_homepath explicitly."
        )
    if metrics_config.auto_download:
        try:
            return _download_grafana(f"{download_dir}/grafana")
        except Exception as ex:
            logger.warning(f"Grafana auto-download failed: {ex}")
    return None


@dataclasses.dataclass
class _MetricsPlan:
    """Resolved binaries + ports for the head-node metrics servers."""

    prometheus_bin: Optional[str] = None
    grafana_bin: Optional[str] = None
    grafana_homepath: Optional[str] = None
    prometheus_port: Optional[int] = None
    grafana_port: Optional[int] = None

    @property
    def run_prometheus(self) -> bool:
        return self.prometheus_bin is not None

    @property
    def run_grafana(self) -> bool:
        # Grafana's datasource points at Prometheus, so only run it if both exist.
        return (
            self.run_prometheus
            and self.grafana_bin is not None
            and self.grafana_homepath is not None
        )


def _prepare_metrics(
    metrics_config: RayMetricsConfig, download_dir: str
) -> _MetricsPlan:
    """Allocate ports and resolve binaries for the metrics servers."""
    plan = _MetricsPlan(
        prometheus_port=metrics_config.prometheus_port or find_free_port(),
        grafana_port=metrics_config.grafana_port or find_free_port(),
    )
    plan.prometheus_bin = _resolve_prometheus_binary(metrics_config, download_dir)
    grafana = _resolve_grafana(metrics_config, download_dir)
    if grafana is not None:
        plan.grafana_bin, plan.grafana_homepath = grafana
    return plan


def _metrics_env_updates(
    head_nodename: str, plan: _MetricsPlan, metrics_config: RayMetricsConfig
) -> dict[str, str]:
    """
    Env vars the Ray dashboard reads (must be set before ``ray start --head``),
    set only for servers we will actually start.
    """
    env: dict[str, str] = {}
    if plan.run_prometheus:
        env["RAY_PROMETHEUS_HOST"] = f"http://{head_nodename}:{plan.prometheus_port}"
        env["RAY_PROMETHEUS_NAME"] = "Prometheus"
    if plan.run_grafana:
        env["RAY_GRAFANA_HOST"] = f"http://{head_nodename}:{plan.grafana_port}"
        env["RAY_GRAFANA_IFRAME_HOST"] = (
            metrics_config.grafana_iframe_host
            or f"http://localhost:{plan.grafana_port}"
        )
    return env


def _wait_for_metrics_configs(
    session_dir: str, timeout: int = METRICS_CONFIG_WAIT_SECONDS
) -> bool:
    """Wait for Ray to generate the Prometheus/Grafana configs under the session dir."""
    prom_cfg = Path(session_dir) / "metrics" / "prometheus" / "prometheus.yml"
    graf_dir = Path(session_dir) / "metrics" / "grafana"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if prom_cfg.exists() and graf_dir.exists():
            return True
        time.sleep(1)
    return prom_cfg.exists() and graf_dir.exists()


def _start_prometheus(
    binary: str, session_dir: str, port: int, data_dir: str, retention: str
) -> subprocess.Popen:
    """Start Prometheus pointed at Ray's generated config; return the process."""
    config_file = Path(session_dir) / "metrics" / "prometheus" / "prometheus.yml"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "--config.file",
        str(config_file),
        f"--web.listen-address=0.0.0.0:{port}",
        f"--storage.tsdb.path={data_dir}",
        f"--storage.tsdb.retention.time={retention}",
    ]
    logger.info(f"Starting Prometheus: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _start_grafana(
    binary: str,
    homepath: str,
    session_dir: str,
    port: int,
    prometheus_url: str,
    data_dir: str,
) -> subprocess.Popen:
    """Start Grafana with Ray's generated provisioning; return the process."""
    grafana_dir = Path(session_dir) / "metrics" / "grafana"
    config_file = grafana_dir / "grafana.ini"
    provisioning = grafana_dir / "provisioning"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GF_SERVER_HTTP_ADDR": "0.0.0.0",
            "GF_SERVER_HTTP_PORT": str(port),
            "GF_PATHS_PROVISIONING": str(provisioning),
            "GF_PATHS_DATA": str(data_dir),
            # Ray's provisioned datasource interpolates this env var.
            "RAY_PROMETHEUS_HOST": prometheus_url,
        }
    )
    cmd = [binary]
    if os.path.basename(binary) == "grafana":
        cmd.append("server")  # modern grafana uses the `server` subcommand
    cmd += ["--homepath", str(homepath), "--config", str(config_file)]
    logger.info(f"Starting Grafana: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


@dataclasses.dataclass
class MetricsServers:
    """Handles + ports for the metrics servers running on the head node."""

    prometheus_proc: Optional[subprocess.Popen] = None
    grafana_proc: Optional[subprocess.Popen] = None
    prometheus_port: Optional[int] = None
    grafana_port: Optional[int] = None

    def terminate(self) -> None:
        """Best-effort terminate both server subprocesses."""
        for proc in (self.prometheus_proc, self.grafana_proc):
            if proc is None:
                continue
            with suppress(Exception):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _start_metrics_servers(
    plan: _MetricsPlan,
    session_dir: str,
    data_dir: str,
    metrics_config: RayMetricsConfig,
) -> MetricsServers:
    """
    Start the metrics servers described by ``plan`` on the head node.

    Best-effort: any failure logs a warning and returns whatever started; it must
    never raise, so a metrics problem can never fail the training job.
    """
    servers = MetricsServers(
        prometheus_port=plan.prometheus_port if plan.run_prometheus else None,
        grafana_port=plan.grafana_port if plan.run_grafana else None,
    )
    if not plan.run_prometheus:
        logger.warning("Prometheus binary unavailable; skipping Ray dashboard metrics.")
        return servers
    if not _wait_for_metrics_configs(session_dir):
        logger.warning(
            f"Ray metrics configs not found under {session_dir}; skipping metrics."
        )
        servers.prometheus_port = None
        servers.grafana_port = None
        return servers
    try:
        servers.prometheus_proc = _start_prometheus(
            plan.prometheus_bin,
            session_dir,
            plan.prometheus_port,
            f"{data_dir}/prometheus",
            metrics_config.prometheus_retention,
        )
    except Exception as ex:
        logger.warning(f"Failed to start Prometheus: {ex}")
        servers.prometheus_port = None
        servers.grafana_port = None
        return servers
    if plan.run_grafana:
        try:
            servers.grafana_proc = _start_grafana(
                plan.grafana_bin,
                plan.grafana_homepath,
                session_dir,
                plan.grafana_port,
                # Grafana queries Prometheus server-side on the same node.
                f"http://localhost:{plan.prometheus_port}",
                f"{data_dir}/grafana",
            )
        except Exception as ex:
            logger.warning(f"Failed to start Grafana: {ex}")
            servers.grafana_port = None
    else:
        logger.warning(
            "Grafana unavailable; the dashboard Metrics tab needs Grafana to render "
            "panels (Prometheus is still collecting metrics)."
        )
    return servers


def _ray_head_script(
    cluster_state: RayClusterState,
    worker_wait_timeout_seconds: int,
    payload: Optional[Callable[..., PayloadReturnT]] = None,
    dashboard_port: Optional[int] = None,
    enable_client_server: bool = False,
    temp_dir_template: Optional[str] = None,
    metrics_config: Optional[RayMetricsConfig] = None,
    **kwargs,
):
    """Start the head node of the Ray cluster on slurm.

    Args:
        cluster_state: State object for cluster coordination
        worker_wait_timeout_seconds: Timeout for workers to connect
        payload: Optional function to run after head starts
        dashboard_port: Port for Ray dashboard. If provided, this exact port is
            used so you can reliably connect to the dashboard. If None, a free
            port is auto-assigned.
        enable_client_server: If True, start Ray Client server for remote connections
        temp_dir_template: Template path for Ray temp files. Supports environment variable
            expansion (e.g., "/scratch/$SLURM_JOB_ID"). Defaults to system temp directory.
        metrics_config: Optional RayMetricsConfig. If enabled, start Prometheus +
            Grafana on the head so the dashboard Metrics tab works. Best-effort:
            failures here never fail the job.
        **kwargs: Additional arguments passed to payload
    """
    # SLURM node name of the head machine (useful for SSH tunneling to the
    # dashboard); falls back to the local hostname when not running under SLURM.
    head_nodename = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    head_env = os.environ.copy()
    num_cpus = os.environ.get("SLURM_CPUS_ON_NODE", 1)
    num_gpus = os.environ.get("SLURM_GPUS_ON_NODE", 0)

    port = find_free_port()
    # Respect an explicitly requested dashboard port so users can reliably
    # connect to the dashboard; otherwise auto-assign a free one.
    if dashboard_port is None:
        dashboard_port = find_free_port()
    else:
        requested_dashboard_port = dashboard_port
        dashboard_port = find_free_port(preferred=requested_dashboard_port)
        if dashboard_port != requested_dashboard_port:
            logger.warning(
                f"Requested dashboard port {requested_dashboard_port} is "
                f"unavailable; using {dashboard_port} instead."
            )

    head_env["RAY_gcs_server_request_timeout_seconds"] = str(
        worker_wait_timeout_seconds
    )
    # Use specified temp directory with environment variable expansion, or system temp
    if temp_dir_template is None:
        temp_dir_template = tempfile.gettempdir()
    else:
        temp_dir_template = os.path.expandvars(temp_dir_template)
    temp_dir = f"{temp_dir_template}/ray_head"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Optionally set up Prometheus + Grafana for the dashboard Metrics tab. Ports
    # and binaries must be resolved BEFORE ``ray start`` so the dashboard picks up
    # the RAY_* env vars; the servers themselves start after Ray generates their
    # configs. Entirely best-effort: never let a metrics issue fail the job.
    metrics_servers = MetricsServers()
    metrics_plan = None
    if metrics_config is not None and metrics_config.enabled:
        try:
            metrics_plan = _prepare_metrics(metrics_config, f"{temp_dir}/metrics_bin")
            head_env.update(
                _metrics_env_updates(head_nodename, metrics_plan, metrics_config)
            )
        except Exception as ex:
            logger.warning(
                f"Metrics setup (pre-start) failed; continuing without metrics: {ex}"
            )
            metrics_plan = None
    try:
        ray_cmd = [
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
            f"--dashboard-port={dashboard_port}",
            f"--object-manager-port={find_free_port()}",
            "--min-worker-port=0",
            "--max-worker-port=0",
        ]
        client_port = None
        if enable_client_server:
            client_port = find_free_port()
            ray_cmd.append(f"--ray-client-server-port={client_port}")
        process = subprocess.Popen(
            ray_cmd,
            env=head_env,
            stdout=subprocess.PIPE,
            text=True,
        )

        # Wait for ray to start by checking stdout
        started = False
        for line in process.stdout:
            if "ray start --address=" in line:
                started = True
        assert (
            started
        ), "couldn't find head address in stdout. Check head.err for details"

        # Read the actual address from ray_current_cluster file
        current_cluster_file = Path(temp_dir) / "ray_current_cluster"
        assert (
            current_cluster_file.exists()
        ), f"ray_current_cluster file not found at {current_cluster_file}"
        address = current_cluster_file.read_text().strip()
        # Address format is "hostname:port"
        head_hostname, port_str = address.rsplit(":", 1)
        port = int(port_str)

        head_env["RAY_ADDRESS"] = address
        logger.info(f"host {address}")
        logger.info(f"Head started, ip: {address} ({cluster_state.cluster_id})")
        logger.info(
            f"Ray dashboard running at http://{head_hostname}:{dashboard_port} "
            f"(bound to 0.0.0.0:{dashboard_port})"
        )

        info = HeadInfo(
            hostname=head_hostname,
            head_nodename=head_nodename,
            port=port,
            client_port=client_port,
            dashboard_port=dashboard_port,
            prometheus_port=(
                metrics_plan.prometheus_port
                if metrics_plan is not None and metrics_plan.run_prometheus
                else None
            ),
            grafana_port=(
                metrics_plan.grafana_port
                if metrics_plan is not None and metrics_plan.run_grafana
                else None
            ),
            temp_dir=temp_dir,
            namespace_serve_fairchem=cluster_state.cluster_id,
        )
        cluster_state.save_head_info(info)
        os.environ.update(head_env)

        # Start the metrics servers now that Ray has generated their configs.
        if metrics_plan is not None:
            try:
                metrics_servers = _start_metrics_servers(
                    metrics_plan,
                    session_dir=str(Path(temp_dir) / "session_latest"),
                    data_dir=f"{temp_dir}/metrics_data",
                    metrics_config=metrics_config,
                )
                if metrics_servers.prometheus_port:
                    logger.info(
                        f"Prometheus running at "
                        f"http://{head_nodename}:{metrics_servers.prometheus_port}"
                    )
                if metrics_servers.grafana_port:
                    logger.info(
                        f"Grafana running at "
                        f"http://{head_nodename}:{metrics_servers.grafana_port}"
                    )
            except Exception as ex:
                logger.warning(
                    f"Failed to start metrics servers; continuing without them: {ex}"
                )

        if payload is not None:
            payload(**kwargs)
        else:
            while True:
                # practically, we should wait from driver signal to die here
                time.sleep(60)
    finally:
        # Stop the metrics servers, then clean up the temp directory.
        metrics_servers.terminate()
        shutil.rmtree(Path(temp_dir), ignore_errors=True)


def worker_script(
    cluster_state: RayClusterState,
    worker_wait_timeout_seconds: int,
    start_wait_time_seconds: int = 60,  # TODO pass this around properly
    temp_dir_template: Optional[str] = None,
):
    """start an array of worker nodes for the Ray cluster on slurm. Waiting on the head node first.

    Args:
        cluster_state: State object for cluster coordination
        worker_wait_timeout_seconds: Timeout for workers to connect
        start_wait_time_seconds: Time to wait for head node to start
        temp_dir_template: Template path for Ray temp files. Supports environment variable
            expansion (e.g., "/scratch/$SLURM_JOB_ID"). Defaults to system temp directory.
    """
    # Sleep a little to avoid race conditions on start and subsequent port collision
    time.sleep(random.uniform(0, 60.0))

    logger.info(f"Waiting for head node. {cluster_state.cluster_id}")
    while not cluster_state.is_head_ready():
        # wait for head to have started
        time.sleep(5)

    logger.info("Head node found.")

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

    # Use specified temp directory with environment variable expansion, or system temp
    if temp_dir_template is None:
        temp_dir_template = tempfile.gettempdir()
    else:
        temp_dir_template = os.path.expandvars(temp_dir_template)
    temp_dir = f"{temp_dir_template}/ray_worker"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ray",
                "start",
                "--address",
                "auto",
                "--block",
                "--temp-dir",
                temp_dir,
                "--num-cpus",
                f"{num_cpus}",
                "--num-gpus",
                f"{num_gpus}",
                f"--object-manager-port={find_free_port()}",
                f"--node-manager-port={find_free_port()}",
                "--min-worker-port=0",
                "--max-worker-port=0",
            ],
            env=worker_env,
            check=False,
        )
    except Exception as ex:
        logger.error(f"Worker failed to start: {ex}")
        raise ex
    finally:
        # Clean up worker temp directory
        shutil.rmtree(Path(temp_dir), ignore_errors=True)


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
    temp_dir_template: Template path for Ray temp files. Supports environment variable expansion
        (e.g., "/scratch/slurm_tmpdir/$SLURM_JOB_ID"). If None, uses the system temp directory.
    cancel_on_exit: If True, register an ``atexit`` hook that scancels submitted slurm jobs
        when the interpreter exits. Defaults to False (fire-and-forget). Note that using the
        cluster as a context manager always cancels jobs on block exit and additionally
        registers the atexit hook as a crash safety net for the duration of the ``with``
        block, independent of this flag.

    """

    def __init__(
        self,
        log_dir: Path = Path("raycluster_logs"),
        rdv_dir: Optional[Path] = None,
        cluster_id: Optional[str] = None,
        worker_wait_timeout_seconds: int = 60,
        temp_dir_template: Optional[str] = None,
        cancel_on_exit: bool = False,
    ):
        self.state = RayClusterState(rdv_dir, cluster_id, log_dir=log_dir)
        logger.info(f"cluster {self.state.cluster_id}")
        self.output_dir = log_dir
        self.log_dir = Path(log_dir) / self.state.cluster_id
        self.state.rendezvous_dir.mkdir(parents=True, exist_ok=True)
        self.worker_wait_timeout_seconds = worker_wait_timeout_seconds
        self.temp_dir_template = temp_dir_template
        self.is_shutdown = False
        self.num_worker_groups = 0
        self.num_drivers = 0
        self.head_started = False
        self.jobs: list[submitit.Job] = []
        logger.info(f"logs will be in {self.log_dir.resolve()}")

        # Optional safety net: scancel SLURM jobs at interpreter exit if the
        # caller forgot to (or couldn't) call shutdown(). Off by default so
        # that submit-and-exit ("fire and forget") scripts don't lose their
        # allocations. Context-manager use registers its own hook in __enter__.
        self._cancel_on_exit = cancel_on_exit
        if cancel_on_exit:
            atexit.register(self._atexit_cancel)

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
            temp_dir_template=self.temp_dir_template,
            **kwargs,
        )
        slurm_job = s_executor.submit(ray_job)
        self.state.add_job(slurm_job)
        self.jobs.append(slurm_job)
        mk_symlinks(self.log_dir, "job", slurm_job.paths)
        logger.info(f"slurm job id: {slurm_job.job_id}")
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
            temp_dir_template=self.temp_dir_template,
            **kwargs,
        )
        self.state.add_job(head_job)
        self.jobs.append(head_job)
        mk_symlinks(self.log_dir, "head", head_job.paths)
        logger.info(f"head slurm job id: {head_job.job_id}")
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
                        temp_dir_template=self.temp_dir_template,
                    )
                )

        for idx, j in enumerate(jobs):
            mk_symlinks(self.log_dir, f"worker_{self.num_worker_groups}_{idx}", j.paths)
        logger.info(f"workers slurm job ids: {[job.job_id for job in jobs]}")
        for j in jobs:
            self.state.add_job(j)
            self.jobs.append(j)
        self.num_worker_groups += 1
        return [job.job_id for job in jobs]

    def shutdown(self):
        """
        Cancel all slurms jobs and get rid of rdv directory.
        """
        if self.is_shutdown:
            return
        self.is_shutdown = True
        scancel(self.state.list_job_ids())
        kill_proc_tree(
            os.getpid(), including_parent=False
        )  # kill local job started by submitit as subprocess TODO that's not going to work when this is not the main process (e.g. recovering on cli)
        self.state.clean()
        logger.info(f"cluster {self.state.cluster_id} shutdown")
        atexit.unregister(self._atexit_cancel)

    def _atexit_cancel(self):
        """
        Minimal scancel for interpreter exit. Does not kill the local proc tree
        or wipe the rendezvous dir (preserved for postmortem on crash).
        """
        if self.is_shutdown:
            return
        with suppress(Exception):
            scancel(self.state.list_job_ids())

    def __enter__(self):
        # only use as a context if you have something blocking waiting on the driver
        # Register a crash safety net for the lifetime of the with-block, so an
        # uncaught exception or KeyboardInterrupt before __exit__ still cancels jobs.
        if not self._cancel_on_exit:
            atexit.register(self._atexit_cancel)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
