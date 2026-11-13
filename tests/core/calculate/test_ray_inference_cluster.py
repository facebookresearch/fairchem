"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from fairchem.core.calculate import _ray_inference_cluster as cluster


@pytest.fixture()
def fake_ray(monkeypatch):
    """
    Run orchestration callbacks synchronously without starting Ray processes.
    """
    runtime = SimpleNamespace(
        init=Mock(),
        shutdown=Mock(),
        is_initialized=Mock(return_value=False),
        get=lambda obj: obj,
        put=lambda obj: obj,
        remote=lambda fn: SimpleNamespace(remote=fn),
        LoggingConfig=lambda **kwargs: kwargs,
        get_runtime_context=lambda: SimpleNamespace(gcs_address="127.0.0.1:9000"),
    )
    monkeypatch.setattr(cluster, "ray", runtime)
    monkeypatch.setattr(cluster.serve, "shutdown", Mock())
    monkeypatch.setattr(cluster, "setup_batch_predict_server", Mock())
    monkeypatch.setattr(cluster, "setup_multiplexed_batch_predict_server", Mock())
    monkeypatch.setattr(cluster, "wait_for_serve_ready", Mock())
    monkeypatch.setattr(cluster, "_find_free_localhost_port", lambda: 9001)
    monkeypatch.setattr(cluster.time, "sleep", Mock())
    return runtime


@pytest.mark.parametrize("mode", ["single", "multiplexed", "no_server"])
def test_local_cluster_writes_connection_info_and_cleans_up(
    tmp_path, monkeypatch, fake_ray, mode
):
    monkeypatch.setattr(cluster, "DEFAULT_HEAD_FILE_DIR", tmp_path)
    monkeypatch.setattr(cluster.torch.cuda, "device_count", lambda: 2)
    unit = object() if mode == "single" else None
    start_server = mode != "no_server"
    with cluster.get_local_inference_raycluster(
        predict_unit=unit,
        start_inference_server=start_server,
    ) as head_file:
        info = json.loads(Path(head_file).read_text())
        assert info["local"]
        assert info["bootstrap_address"] == "127.0.0.1:9000"
        assert info["num_gpus"] == 2
        assert info["num_cpus"] == 8
        assert info["dashboard_port"] == 9001
    assert not Path(head_file).exists()
    fake_ray.shutdown.assert_called_once()
    if start_server:
        name = "predict-server" if mode == "single" else "multiplexed-predict-server"
        cluster.wait_for_serve_ready.assert_called_once_with(app_name=name)
        cluster.serve.shutdown.assert_called_once()
    else:
        cluster.wait_for_serve_ready.assert_not_called()


def test_local_cluster_cleans_up_after_body_error(tmp_path, fake_ray):
    path = tmp_path / "cluster" / "head.json"
    with (
        pytest.raises(ValueError, match="consumer failed"),
        cluster.get_local_inference_raycluster(
            head_file=path, num_cpus=1, num_gpus=0, start_inference_server=False
        ),
    ):
        assert "num_gpus" not in fake_ray.init.call_args.kwargs
        raise ValueError("consumer failed")
    assert not path.exists()
    fake_ray.shutdown.assert_called_once()


def test_cluster_configuration_precedence_capacity_and_slurm_requirements(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "partition": "default",
                "num_workers": 2,
                "cpus_per_node": 4,
                "gpus_per_node": 8,
                "time_minutes": 10,
                "mem_gb": 16,
            }
        )
    )
    monkeypatch.setenv("RAY_FAIRRAY_DIR", str(tmp_path / "rdv"))
    result = cluster._build_cluster_config(
        config,
        partition="explicit",
        cpus_per_node=8,
        env_vars={"TEST": "yes"},
        exclude_nodes=["bad1", "bad2"],
        slurm_constraint="h100",
        mem_gb=32,
        time_minutes=20,
        gpus_per_node=8,
        num_workers=2,
        start_inference_server=True,
        serve_log_level="INFO",
        slurm_additional_parameters={"slurm_qos": "test"},
        cluster_config_overrides={
            "partition": "override",
            "deployment_config": {
                "ray_actor_options": {"num_gpus": 2, "num_cpus": 2},
                "autoscaling_config": {"max_replicas": 2.0, "min_replicas": 0.5},
            },
        },
    )
    assert result["partition"] == "override"
    assert result["rdv_dir"] == str(tmp_path / "rdv")
    assert result["head_file"].endswith("/head.json")
    deployment, batching = cluster._resolve_serve_configs(result)
    assert deployment["autoscaling_config"] == {"max_replicas": 16, "min_replicas": 8}
    assert deployment["logging_config"].log_level == "INFO"
    assert batching == {}
    assert result["deployment_config"]["autoscaling_config"]["max_replicas"] == 2.0
    requirements = cluster._build_slurm_requirements(result)
    assert requirements["cpus_per_task"] == 16
    assert requirements["gpus_per_node"] == 8
    assert requirements["slurm_exclude"] == "bad1,bad2"
    assert requirements["slurm_qos"] == "test"
    assert requirements["slurm_constraint"] == "h100"
    assert (
        cluster.load_update_config(config, head_file="existing.json")["head_file"]
        == "existing.json"
    )
    for minimum in (2.0, 2):
        deployment, _ = cluster._resolve_serve_configs(
            {
                "num_workers": 4,
                "deployment_config": {
                    "autoscaling_config": {"max_replicas": 3, "min_replicas": minimum}
                },
            }
        )
        assert deployment["autoscaling_config"]["min_replicas"] == 2


@pytest.mark.parametrize("mode", ["single", "multiplexed", "caller_owned"])
def test_slurm_cluster_deploys_and_releases_only_owned_client(
    tmp_path, monkeypatch, fake_ray, mode
):
    monkeypatch.delenv("RAY_HEAD_FILE", raising=False)
    monkeypatch.setenv("RAY_PREFECT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RAY_FAIRRAY_DIR", str(tmp_path / "rdv"))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"partition": "test", "cpus_per_node": 2, "time_minutes": 1, "mem_gb": 1}
        )
    )
    head_file = tmp_path / "head.json"
    head_file.write_text(
        json.dumps(
            {
                "hostname": "head",
                "client_port": 10001,
                "namespace_serve_fairchem": "test",
            }
        )
    )
    instance = Mock()
    instance.state._head_json = head_file
    instance.state.is_head_ready.side_effect = [False, True]
    instance.state.head_info.return_value = SimpleNamespace(hostname="head", port=9000)
    monkeypatch.setattr(cluster, "RayCluster", Mock(return_value=instance))
    if mode == "caller_owned":
        fake_ray.is_initialized.return_value = True
    unit = object() if mode == "single" else None
    with cluster.get_slurm_inference_raycluster(
        config=config, num_workers=2, start_inference_server=True, predict_unit=unit
    ) as result:
        assert result == str(head_file)
    instance.start_head.assert_called_once()
    instance.start_workers.assert_called_once()
    instance.shutdown.assert_called_once()
    if mode == "caller_owned":
        fake_ray.init.assert_not_called()
        fake_ray.shutdown.assert_not_called()
        cluster.wait_for_serve_ready.assert_not_called()
    else:
        fake_ray.init.assert_called_once_with(
            "ray://head:10001", ignore_reinit_error=True, namespace="test"
        )
        fake_ray.shutdown.assert_called_once()
        name = "predict-server" if mode == "single" else "multiplexed-predict-server"
        cluster.wait_for_serve_ready.assert_called_once_with(app_name=name)


def test_slurm_existing_cluster_is_not_shutdown(tmp_path, monkeypatch, fake_ray):
    path = tmp_path / "head.json"
    path.write_text("{}")
    monkeypatch.setenv("RAY_HEAD_FILE", str(path))
    constructor = Mock()
    monkeypatch.setattr(cluster, "RayCluster", constructor)
    with cluster.get_slurm_inference_raycluster() as result:
        assert result == str(path)
    constructor.assert_not_called()
    fake_ray.shutdown.assert_not_called()
