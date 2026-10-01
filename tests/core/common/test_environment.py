"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from fairchem.core.common import environment


class _StringSubclass(str):
    __slots__ = ()


def _fake_system_environment() -> SimpleNamespace:
    return SimpleNamespace(
        python_version="3.12.0",
        python_platform="Linux-x86_64",
        torch_version=_StringSubclass("2.13.0"),
        is_debug_build="False",
        cuda_compiled_version="13.0",
        rocm_compiled_version="N/A",
        hip_compiled_version="N/A",
        caching_allocator_config={"PYTORCH_CUDA_ALLOC_CONF": "test"},
        os="Test Linux",
        gcc_version="12.1",
        clang_version="18.0",
        cmake_version="3.30",
        cpu_info="CPU(s): 8\nModel name: Test CPU",
        is_cuda_available="True",
        cuda_module_loading="LAZY",
        nvidia_gpu_models="GPU 0: Test GPU",
        is_xpu_available="False",
        libc_version="glibc-2.39",
        cuda_runtime_version="13.0",
        nvidia_driver_version="999.0",
        cudnn_version="9.0",
        hip_runtime_version="N/A",
        miopen_runtime_version="N/A",
        is_xnnpack_available="True",
    )


def _write_report(tmp_path, **overrides):
    arguments = {
        "log_dir": str(tmp_path),
        "run_type": "run",
        "timestamp_id": "timestamp",
        "commit": "abc123",
        "job_id": "123_4",
        "array_job_id": "123",
        "array_task_id": "4",
        "restart_count": "1",
    }
    arguments.update(overrides)
    return environment.write_environment_report(**arguments)


def test_writes_safe_node_report(tmp_path, monkeypatch) -> None:
    for variable in environment.ENVIRONMENT_VARIABLE_ALLOWLIST:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NODEID", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_TOKEN", "secret-huggingface-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-cloud-key")

    monkeypatch.setattr(environment, "get_env_info", _fake_system_environment)
    monkeypatch.setattr(
        environment.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "z-package"}, version="2.0"),
            SimpleNamespace(metadata={"Name": "A-package"}, version="1.0"),
        ],
    )
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 8)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 16)
    monkeypatch.setattr(environment.socket, "gethostname", lambda: "test-host")

    report_path = _write_report(tmp_path)

    assert report_path == (tmp_path / "environment" / "run_123_4_node_2_restart_1.yaml")
    report_text = report_path.read_text()
    report = yaml.safe_load(report_text)
    assert report["rank"] == {
        "global_rank": 8,
        "local_rank": 0,
        "world_size": 16,
        "node_id": "2",
        "hostname": "test-host",
    }
    assert report["environment_variables"] == {
        "SLURM_NODEID": "2",
        "SLURM_LOCALID": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert list(report["python_packages"]) == ["A-package", "z-package"]
    assert report["pytorch"]["version"] == "2.13.0"
    assert report["cpu"]["details"] == "CPU(s): 8\nModel name: Test CPU"
    assert "details: |-" in report_text
    assert "HF_TOKEN" not in report_text
    assert "secret-huggingface-token" not in report_text
    assert "AWS_SECRET_ACCESS_KEY" not in report_text
    assert "secret-cloud-key" not in report_text
    assert not list(report_path.parent.glob("*.tmp"))


def test_non_node_leader_does_not_collect_or_write(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")

    def fail_if_called():
        raise AssertionError("non-node leaders must not collect environment data")

    monkeypatch.setattr(environment, "get_env_info", fail_if_called)

    assert _write_report(tmp_path) is None
    assert not (tmp_path / "environment").exists()


def test_collection_failures_produce_partial_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("SLURM_LOCALID", raising=False)

    def fail_system_collection():
        raise RuntimeError("sensitive failure details")

    def fail_package_collection():
        raise OSError("sensitive package details")

    monkeypatch.setattr(environment, "get_env_info", fail_system_collection)
    monkeypatch.setattr(environment.metadata, "distributions", fail_package_collection)
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 0)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 1)

    report_path = _write_report(tmp_path)
    report_text = report_path.read_text()
    report = yaml.safe_load(report_text)

    assert report["collection_errors"] == {
        "system_environment": ["RuntimeError"],
        "python_packages": ["OSError"],
    }
    assert report["python_packages"] == {}
    assert report["operating_system"]["description"] is None
    assert "sensitive failure details" not in report_text
    assert "sensitive package details" not in report_text


def test_invalid_local_rank_is_nonfatal(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "not-an-integer")

    assert _write_report(tmp_path) is None
    assert "Failed to determine local rank" in caplog.text
