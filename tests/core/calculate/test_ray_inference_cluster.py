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
    RayServeStartupError,
    RayWorkerStartupError,
    _get_serve_setup_result,
    _wait_for_expected_gpu_capacity,
    get_slurm_inference_raycluster,
    start_ray_cluster,
)
from fairchem.core.components.batch_server import (
    RayServeHandleUnavailableError,
    get_app_handle_with_retry,
)
from fairchem.core.launchers.cluster.ray_cluster import RayClusterCleanupError


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


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_wraps_raw_submission_failure(ray_cluster_cls):
    cluster = ray_cluster_cls.return_value
    cluster.start_head.side_effect = OSError("sbatch unavailable")

    with pytest.raises(RayClusterStartupError, match="submitting or starting") as exc:
        start_ray_cluster(_cluster_config(1), return_cluster=True)

    assert isinstance(exc.value.__cause__, OSError)
    cluster.shutdown.assert_called_once_with()


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_preserves_deterministic_submission_failure(
    ray_cluster_cls,
):
    cluster = ray_cluster_cls.return_value
    cluster.start_head.side_effect = ValueError("invalid SLURM configuration")

    with pytest.raises(ValueError, match="invalid SLURM configuration"):
        start_ray_cluster(_cluster_config(1), return_cluster=True)

    cluster.shutdown.assert_called_once_with()


@patch("fairchem.core.calculate._ray_inference_cluster.RayCluster")
def test_start_ray_cluster_surfaces_cleanup_failure(ray_cluster_cls):
    cluster = ray_cluster_cls.return_value
    cluster.start_head.side_effect = OSError("sbatch unavailable")
    cluster.shutdown.side_effect = RayClusterCleanupError("allocation still live")

    with pytest.raises(RayClusterCleanupError, match="allocation still live") as exc:
        start_ray_cluster(_cluster_config(1), return_cluster=True)

    assert exc.value.__cause__ is None
    assert isinstance(exc.value.__context__, OSError)


def test_stale_external_head_file_fails_before_submission(monkeypatch, tmp_path):
    missing = tmp_path / "missing-head.json"
    monkeypatch.setenv("RAY_HEAD_FILE", str(missing))

    with (
        patch(
            "fairchem.core.calculate._ray_inference_cluster.start_ray_cluster"
        ) as start,
        pytest.raises(FileNotFoundError, match="external cluster"),
        get_slurm_inference_raycluster(),
    ):
        pass

    start.assert_not_called()


def test_serve_setup_failure_has_typed_boundary():
    ray_module = MagicMock()
    ray_module.get.side_effect = RuntimeError("Application failed to deploy")
    object_ref = object()
    submit = MagicMock(return_value=object_ref)

    with pytest.raises(RayServeStartupError, match="readiness"):
        _get_serve_setup_result(
            ray_module=ray_module,
            submit=submit,
            deployment_name="predict-server",
            operation="readiness check",
            timeout_seconds=7.5,
        )

    submit.assert_called_once_with()
    ray_module.get.assert_called_once_with(object_ref, timeout=7.5)


def test_serve_setup_preserves_remote_deterministic_failure():
    class RayTaskError(RuntimeError):
        def __init__(self, cause):
            super().__init__(str(cause))
            self.cause = cause

    ray_module = MagicMock()
    ray_module.get.side_effect = RayTaskError(ValueError("invalid model config"))

    with pytest.raises(RayTaskError, match="invalid model config"):
        _get_serve_setup_result(
            ray_module=ray_module,
            submit=lambda: object(),
            deployment_name="predict-server",
            operation="deployment",
            timeout_seconds=7.5,
        )


def test_serve_submission_failure_has_typed_boundary():
    ray_module = MagicMock()
    submit = MagicMock(side_effect=ConnectionError("Ray client disconnected"))

    with pytest.raises(RayServeStartupError, match="deployment"):
        _get_serve_setup_result(
            ray_module=ray_module,
            submit=submit,
            deployment_name="predict-server",
            operation="deployment submission",
            timeout_seconds=7.5,
        )

    ray_module.get.assert_not_called()


def test_invalid_serve_timeout_does_not_submit():
    submit = MagicMock()

    with pytest.raises(ValueError, match="must be positive"):
        _get_serve_setup_result(
            ray_module=MagicMock(),
            submit=submit,
            deployment_name="predict-server",
            operation="deployment",
            timeout_seconds=0,
        )

    submit.assert_not_called()


def test_serve_handle_lookup_exhaustion_has_typed_boundary():
    lookup_error = RuntimeError("There is no Serve instance in the namespace")

    with (
        patch(
            "fairchem.core.components.batch_server.serve.get_app_handle",
            side_effect=lookup_error,
        ),
        patch(
            "fairchem.core.components.batch_server.time.monotonic",
            side_effect=[0.0, 1.0],
        ),
        pytest.raises(RayServeHandleUnavailableError, match="remained unavailable"),
    ):
        get_app_handle_with_retry("predict-server", timeout_seconds=0)


def test_partial_ray_init_is_released_and_cluster_is_stopped(monkeypatch, tmp_path):
    head_file = tmp_path / "head.json"
    head_file.write_text(
        '{"hostname": "head", "client_port": 10001, '
        '"namespace_serve_fairchem": "ns"}',
        encoding="utf-8",
    )
    cluster = MagicMock(worker_wait_timeout_seconds=1)
    fake_ray = SimpleNamespace(
        is_initialized=MagicMock(return_value=False),
        init=MagicMock(side_effect=ConnectionError("client handshake failed")),
        shutdown=MagicMock(),
    )
    monkeypatch.setenv("FAIRCHEM_RAY_INIT_MAX_ATTEMPTS", "1")
    config = {
        "start_inference_server": True,
        "num_workers": 1,
        "gpus_per_node": 0,
    }

    with (
        patch.dict("sys.modules", {"ray": fake_ray}),
        patch(
            "fairchem.core.calculate._ray_inference_cluster._build_cluster_config",
            return_value=config,
        ),
        patch(
            "fairchem.core.calculate._ray_inference_cluster.start_ray_cluster",
            return_value=(str(head_file), cluster),
        ),
        pytest.raises(RayClusterStartupError, match="Failed to connect"),
        get_slurm_inference_raycluster(start_inference_server=True),
    ):
        pass

    fake_ray.shutdown.assert_called_once_with()
    cluster.shutdown.assert_called_once_with()


def test_ray_client_cleanup_failure_still_stops_cluster(monkeypatch, tmp_path):
    head_file = tmp_path / "head.json"
    head_file.write_text(
        '{"hostname": "head", "client_port": 10001, '
        '"namespace_serve_fairchem": "ns"}',
        encoding="utf-8",
    )
    cluster = MagicMock(worker_wait_timeout_seconds=1)
    fake_ray = SimpleNamespace(
        is_initialized=MagicMock(return_value=False),
        init=MagicMock(),
        shutdown=MagicMock(side_effect=RuntimeError("client cleanup failed")),
    )
    config = {
        "start_inference_server": True,
        "num_workers": 1,
        "gpus_per_node": 0,
    }

    with (
        patch.dict("sys.modules", {"ray": fake_ray}),
        patch(
            "fairchem.core.calculate._ray_inference_cluster._build_cluster_config",
            return_value=config,
        ),
        patch(
            "fairchem.core.calculate._ray_inference_cluster.start_ray_cluster",
            return_value=(str(head_file), cluster),
        ),
        patch(
            "fairchem.core.calculate._ray_inference_cluster._resolve_serve_configs",
            side_effect=ValueError("invalid serve config"),
        ),
        pytest.raises(RayClusterCleanupError, match="refusing to replace"),
        get_slurm_inference_raycluster(start_inference_server=True),
    ):
        pass

    fake_ray.shutdown.assert_called_once_with()
    cluster.shutdown.assert_called_once_with()


def test_context_propagates_cluster_cleanup_failure(tmp_path):
    head_file = tmp_path / "head.json"
    head_file.write_text("{}", encoding="utf-8")
    cluster = MagicMock()
    cluster.shutdown.side_effect = RayClusterCleanupError("jobs still active")
    config = {"start_inference_server": False}

    with (
        patch(
            "fairchem.core.calculate._ray_inference_cluster._build_cluster_config",
            return_value=config,
        ),
        patch(
            "fairchem.core.calculate._ray_inference_cluster.start_ray_cluster",
            return_value=(str(head_file), cluster),
        ),
        pytest.raises(RayClusterCleanupError, match="jobs still active"),
        get_slurm_inference_raycluster(),
    ):
        pass

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
