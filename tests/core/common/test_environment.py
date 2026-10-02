"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from click.testing import CliRunner

from fairchem.core.common import environment
from tests.perf import performance_report


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
        pip_packages="z-package==2.0\nA-package==1.0",
        conda_packages="A-package 1.1 pypi_0 pypi\nconda-only 3.0 pypi_0 pypi",
    )


def _write_report(tmp_path, **overrides):
    arguments = {
        "log_dir": str(tmp_path),
        "run_type": "run",
        "timestamp_id": "timestamp",
        "submission_commit": "submission-abc123",
    }
    arguments.update(overrides)
    return environment.write_environment_report(**arguments)


def test_writes_safe_node_report(tmp_path, monkeypatch) -> None:
    for variable in environment.ENVIRONMENT_VARIABLE_ALLOWLIST:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NODEID", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_TOKEN", "secret-huggingface-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-cloud-key")

    monkeypatch.setattr(
        environment,
        "_collect_system_environment",
        lambda: (_fake_system_environment(), []),
    )
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 8)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 16)
    monkeypatch.setattr(environment.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(environment, "get_commit_hash", lambda: "runtime-def456")

    report_path = _write_report(tmp_path)

    assert report_path == (tmp_path / "environment" / "run_123_4_node_2_restart_1.json")
    report_text = report_path.read_text()
    report = json.loads(report_text)
    assert report["schema_version"] == 1
    assert report["measurements"] == {}
    assert report["job"] == {
        "run_type": "run",
        "timestamp_id": "timestamp",
        "commit": "submission-abc123",
        "job_id": "123_4",
        "array_job_id": "123",
        "array_task_id": "4",
        "restart_count": "1",
    }
    assert report["environment"]["git_commit_hash"] == "runtime-def456"
    assert report["rank"] == {
        "global_rank": 8,
        "local_rank": 0,
        "world_size": 16,
        "node_id": "2",
        "hostname": "test-host",
    }
    assert report["environment"]["environment_variables"] == {
        "SLURM_JOB_ID": "456",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "4",
        "SLURM_NODEID": "2",
        "SLURM_LOCALID": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert list(report["environment"]["libraries"]) == [
        "A-package",
        "conda-only",
        "z-package",
    ]
    assert report["environment"]["libraries"]["A-package"] == "1.1"
    assert report["environment"]["pytorch_version"] == "2.13.0"
    assert report["environment"]["num_cpus"] == "8"
    assert report["environment"]["cpu_model"] == "Test CPU"
    assert "cpu_info" not in report["environment"]
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

    monkeypatch.setattr(environment, "_collect_system_environment", fail_if_called)

    assert _write_report(tmp_path) is None
    assert not (tmp_path / "environment").exists()


def test_collection_failure_produces_partial_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("SLURM_LOCALID", raising=False)

    monkeypatch.setattr(
        environment,
        "_collect_system_environment",
        lambda: (None, ["RuntimeError"]),
    )
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 0)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 1)

    report_path = _write_report(tmp_path)
    report_text = report_path.read_text()
    report = json.loads(report_text)

    assert report["collection_errors"] == {
        "system_environment": ["RuntimeError"],
    }
    assert report["environment"]["libraries"] == {}
    assert report["environment"]["os"] is None


def test_written_report_is_accepted_by_performance_cli(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    monkeypatch.setattr(
        environment,
        "_collect_system_environment",
        lambda: (_fake_system_environment(), []),
    )
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 0)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 1)
    report_path = _write_report(tmp_path)
    target_report = json.loads(report_path.read_text())
    target_report["environment"]["environment_variables"]["OMP_NUM_THREADS"] = "4"
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target_report))

    result = CliRunner().invoke(
        performance_report.cli,
        ["compare", str(target_path), "--baseline", str(report_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    comparison = json.loads(result.output)
    assert comparison["environment"]["changed"] == [
        {
            "attribute": "environment_variables.OMP_NUM_THREADS",
            "value": "4",
            "baseline_value": "8",
        }
    ]
    assert comparison["environment"]["added"] == []
    assert comparison["environment"]["removed"] == []
    assert comparison["environment"]["unchanged"]


def test_system_collection_timeout_is_nonfatal(monkeypatch) -> None:
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="collect-env", timeout=30)

    monkeypatch.setattr(environment.subprocess, "run", time_out)

    system_environment, errors = environment._collect_system_environment()

    assert system_environment is None
    assert errors == ["TimeoutExpired"]


def test_system_collection_is_isolated_and_bounded(monkeypatch) -> None:
    def collect(command, **kwargs):
        assert command[:2] == [environment.sys.executable, "-c"]
        assert kwargs["timeout"] == environment.ENVIRONMENT_COLLECTION_TIMEOUT_SECONDS
        assert kwargs["check"] is True
        return SimpleNamespace(stdout='{"python_version": "3.12.0"}')

    monkeypatch.setattr(environment.subprocess, "run", collect)

    system_environment, errors = environment._collect_system_environment()

    assert system_environment == {"python_version": "3.12.0"}
    assert errors == []


def test_package_versions_include_pipless_environment_metadata() -> None:
    packages = environment.get_python_package_versions(
        {
            "pip_packages": None,
            "conda_packages": None,
            "installed_python_packages": {
                "fairchem-core": "2.23.0",
                "torch": "2.13.0+cpu",
            },
        }
    )

    assert packages == {
        "fairchem-core": "2.23.0",
        "torch": "2.13.0+cpu",
    }


def test_slurm_local_rank_takes_precedence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        environment,
        "_collect_system_environment",
        lambda: (_fake_system_environment(), []),
    )
    monkeypatch.setattr(environment.distutils, "get_rank", lambda: 0)
    monkeypatch.setattr(environment.distutils, "get_world_size", lambda: 2)

    report_path = _write_report(tmp_path)

    assert json.loads(report_path.read_text())["rank"]["local_rank"] == 0


def test_slurm_rank_is_used_before_distributed_setup(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setattr(environment.distutils, "initialized", lambda: False)

    assert environment._get_global_rank() == 3
    assert environment._get_world_size() == 8


def test_invalid_local_rank_is_nonfatal(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "not-an-integer")

    assert _write_report(tmp_path) is None
    assert "Failed to determine local rank" in caplog.text
