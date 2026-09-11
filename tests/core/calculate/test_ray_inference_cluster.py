"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fairchem.core.calculate._ray_inference_cluster import (
    RayClusterStartupError,
    RayWorkerStartupError,
    _wait_for_expected_gpu_capacity,
    start_ray_cluster,
)


def _cluster_config(num_workers: int) -> dict:
    return {
        "num_workers": num_workers,
        "partition": "test",
        "cpus_per_node": 8,
        "gpus_per_node": 1,
        "time_minutes": 10,
        "mem_gb": 32,
    }


@pytest.mark.parametrize(
    ("total_jobs", "additional_worker_jobs"),
    [(1, 0), (32, 31)],
)
@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_treats_num_workers_as_total_jobs(
    ray_cluster_cls,
    total_jobs,
    additional_worker_jobs,
    tmp_path,
):
    cluster = ray_cluster_cls.return_value
    head_file = tmp_path / "head.json"
    cluster.state._head_json = head_file
    cluster.state.is_head_ready.return_value = True
    cluster.state.head_info.return_value = SimpleNamespace(
        hostname="head-node",
        port=6379,
    )

    returned_head_file, returned_cluster = start_ray_cluster(
        _cluster_config(total_jobs), return_cluster=True
    )

    assert returned_head_file == str(head_file)
    assert returned_cluster is cluster
    cluster.start_head.assert_called_once()
    if additional_worker_jobs:
        cluster.start_workers.assert_called_once_with(
            num_workers=additional_worker_jobs,
            requirements={
                "slurm_partition": "test",
                "cpus_per_task": 8,
                "slurm_time": 10,
                "mem_gb": 32,
                "gpus_per_node": 1,
            },
            name="ray_cluster",
        )
    else:
        cluster.start_workers.assert_not_called()


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_rejects_zero_total_jobs(ray_cluster_cls):
    with pytest.raises(ValueError, match="num_workers must be at least 1"):
        start_ray_cluster(_cluster_config(0))

    ray_cluster_cls.assert_not_called()


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_reports_failed_head_and_cleans_up(ray_cluster_cls, tmp_path):
    cluster = ray_cluster_cls.return_value
    cluster.state._head_json = tmp_path / "head.json"
    cluster.state.is_head_ready.return_value = False
    head = MagicMock(job_id="head-1")
    head.done.return_value = True
    head.state = "FAILED"
    head.stderr.return_value = "head scratch unavailable"
    cluster.jobs = [head]

    with pytest.raises(RayClusterStartupError) as exc_info:
        start_ray_cluster(_cluster_config(1), return_cluster=True)

    message = str(exc_info.value)
    assert "head-1 [FAILED]" in message
    assert "head scratch unavailable" in message
    cluster.shutdown.assert_called_once_with()


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_head_wait_is_bounded_and_cleans_up(
    ray_cluster_cls, tmp_path
):
    cluster = ray_cluster_cls.return_value
    cluster.state._head_json = tmp_path / "head.json"
    cluster.state.is_head_ready.return_value = False
    head = MagicMock(job_id="head-1")
    head.done.return_value = False
    head.state = "PENDING"
    cluster.jobs = [head]
    config = _cluster_config(1)
    config["worker_wait_timeout_seconds"] = 0

    with pytest.raises(RayClusterStartupError, match="Timed out.*head-1"):
        start_ray_cluster(config, return_cluster=True)

    cluster.shutdown.assert_called_once_with()


def test_wait_for_expected_gpu_capacity_waits_before_returning():
    ray_module = MagicMock()
    ray_module.cluster_resources.side_effect = [{"GPU": 1.0}, {"GPU": 2.0}]
    worker = MagicMock(job_id="worker-1")
    worker.done.return_value = False
    cluster = SimpleNamespace(jobs=[MagicMock(job_id="head"), worker])

    with patch("fairchem.core.calculate._ray_inference_cluster.time.sleep") as sleep:
        _wait_for_expected_gpu_capacity(
            ray_module=ray_module,
            cluster=cluster,
            expected_gpus=2,
            timeout_seconds=30,
        )

    sleep.assert_called_once_with(5.0)
    worker.done.assert_called_once_with(force_check=True)


def test_wait_for_expected_gpu_capacity_reports_exited_worker():
    ray_module = MagicMock()
    ray_module.cluster_resources.return_value = {"GPU": 31.0}
    worker = MagicMock(job_id="worker-31")
    worker.done.return_value = True
    worker.state = "FAILED"
    worker.stderr.return_value = "OSError: [Errno 28] No space left on device"
    cluster = SimpleNamespace(jobs=[MagicMock(job_id="head"), worker])

    with pytest.raises(RayWorkerStartupError) as exc_info:
        _wait_for_expected_gpu_capacity(
            ray_module=ray_module,
            cluster=cluster,
            expected_gpus=32,
            timeout_seconds=300,
        )

    message = str(exc_info.value)
    assert "31/32" in message
    assert "worker-31 [FAILED]" in message
    assert "No space left on device" in message


def test_wait_for_expected_gpu_capacity_has_bounded_wait():
    ray_module = MagicMock()
    ray_module.cluster_resources.return_value = {"GPU": 31.0}
    worker = MagicMock(job_id="worker-31")
    worker.done.return_value = False
    worker.state = "PENDING"
    cluster = SimpleNamespace(jobs=[MagicMock(job_id="head"), worker])

    with pytest.raises(RayClusterStartupError, match="31/32"):
        _wait_for_expected_gpu_capacity(
            ray_module=ray_module,
            cluster=cluster,
            expected_gpus=32,
            timeout_seconds=0,
        )
