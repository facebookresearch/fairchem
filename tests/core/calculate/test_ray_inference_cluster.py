"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fairchem.core.calculate._ray_inference_cluster import start_ray_cluster


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
