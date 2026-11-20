"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# ruff: noqa
from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fairchem.core.common.utils import download_and_extract, os_arch_for_download

if TYPE_CHECKING:
    from fairchem.core.launchers.api import RayMetricsConfig

logger = logging.getLogger(__name__)

# Pinned binary versions used only for the explicit metrics auto-download fallback.
# Users who need specific versions should install the binaries themselves.
PROMETHEUS_AUTODOWNLOAD_VERSION = "3.1.0"
GRAFANA_AUTODOWNLOAD_VERSION = "11.4.0"

# How long to wait for Ray to generate the Prometheus/Grafana config files under
# the session dir before giving up on starting the metrics servers.
METRICS_CONFIG_WAIT_SECONDS = 60


def _download_prometheus(dest_dir: str) -> Optional[str]:
    """Download a static Prometheus release; return the binary path or None."""
    os_type, arch = os_arch_for_download()
    ver = PROMETHEUS_AUTODOWNLOAD_VERSION
    name = f"prometheus-{ver}.{os_type}-{arch}"
    url = (
        "https://github.com/prometheus/prometheus/releases/"
        f"download/v{ver}/{name}.tar.gz"
    )
    binary = download_and_extract(url, dest_dir) / "prometheus"
    return str(binary) if binary.exists() else None


def _download_grafana(dest_dir: str) -> Optional[tuple[str, str]]:
    """Download a Grafana OSS release; return (binary, homepath) or None."""
    os_type, arch = os_arch_for_download()
    ver = GRAFANA_AUTODOWNLOAD_VERSION
    url = f"https://dl.grafana.com/oss/release/grafana-{ver}.{os_type}-{arch}.tar.gz"
    root = download_and_extract(url, dest_dir)
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
    metrics_config: RayMetricsConfig,
    download_dir: str,
    prometheus_port: int,
    grafana_port: int,
) -> _MetricsPlan:
    """
    Resolve the metrics-server binaries, given pre-allocated ports.

    Ports are passed in (rather than allocated here) so this module stays free of
    any dependency on the cluster's port allocation.
    """
    plan = _MetricsPlan(
        prometheus_port=prometheus_port,
        grafana_port=grafana_port,
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
