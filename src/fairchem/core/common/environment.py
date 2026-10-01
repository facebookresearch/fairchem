"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from torch.utils.collect_env import get_env_info

from fairchem.core.common import distutils

ENVIRONMENT_VARIABLE_ALLOWLIST = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_NNODES",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_DISTRIBUTED_DEBUG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOGLEVEL",
)


class _EnvironmentDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.nodes.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_EnvironmentDumper.add_representer(str, _represent_string)


def _get_local_rank() -> int:
    for variable in ("LOCAL_RANK", "SLURM_LOCALID"):
        if variable in os.environ:
            return int(os.environ[variable])
    return 0


def _get_node_id() -> str:
    return os.environ.get("SLURM_NODEID", "0")


def _safe_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _collect_python_packages() -> tuple[dict[str, str], list[str]]:
    packages: list[tuple[str, str]] = []
    errors: set[str] = set()
    try:
        distributions = metadata.distributions()
        for distribution in distributions:
            try:
                name = distribution.metadata.get("Name")
                if name:
                    packages.append((name, distribution.version))
            except Exception as error:
                errors.add(type(error).__name__)
    except Exception as error:
        errors.add(type(error).__name__)

    return (
        dict(
            sorted(
                packages,
                key=lambda package: package[0].casefold(),
            )
        ),
        sorted(errors),
    )


def _system_value(system_environment: Any, name: str) -> Any:
    if system_environment is None:
        return None
    return getattr(system_environment, name, None)


def _to_yaml_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _to_yaml_safe(nested_value) for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe(nested_value) for nested_value in value]
    return str(value)


def collect_environment_report(
    *,
    run_type: str,
    timestamp_id: str,
    commit: str,
    job_id: str | None,
    array_job_id: str | None,
    array_task_id: str | None,
    restart_count: str | None,
) -> dict[str, Any]:
    """
    Collect diagnostic information for a FairChem process.

    Environment variables are restricted to a fixed allow-list so credentials and
    other secrets are not copied into the report.
    """
    collection_errors: dict[str, list[str]] = {}
    try:
        system_environment = get_env_info()
    except Exception as error:
        system_environment = None
        collection_errors["system_environment"] = [type(error).__name__]

    python_packages, package_errors = _collect_python_packages()
    if package_errors:
        collection_errors["python_packages"] = package_errors

    return {
        "schema_version": 1,
        "job": {
            "run_type": run_type,
            "timestamp_id": timestamp_id,
            "commit": commit,
            "job_id": job_id,
            "array_job_id": array_job_id,
            "array_task_id": array_task_id,
            "restart_count": restart_count,
        },
        "rank": {
            "global_rank": distutils.get_rank(),
            "local_rank": _get_local_rank(),
            "world_size": distutils.get_world_size(),
            "node_id": _get_node_id(),
            "hostname": socket.gethostname(),
        },
        "environment_variables": {
            variable: os.environ[variable]
            for variable in ENVIRONMENT_VARIABLE_ALLOWLIST
            if variable in os.environ
        },
        "python": {
            "version": _system_value(system_environment, "python_version"),
            "platform": _system_value(system_environment, "python_platform"),
        },
        "python_packages": python_packages,
        "pytorch": {
            "version": _system_value(system_environment, "torch_version"),
            "debug_build": _system_value(system_environment, "is_debug_build"),
            "cuda_build_version": _system_value(
                system_environment, "cuda_compiled_version"
            ),
            "rocm_build_version": _system_value(
                system_environment, "rocm_compiled_version"
            ),
            "hip_build_version": _system_value(
                system_environment, "hip_compiled_version"
            ),
            "caching_allocator_config": _system_value(
                system_environment, "caching_allocator_config"
            ),
        },
        "operating_system": {
            "description": _system_value(system_environment, "os"),
            "gcc_version": _system_value(system_environment, "gcc_version"),
            "clang_version": _system_value(system_environment, "clang_version"),
            "cmake_version": _system_value(system_environment, "cmake_version"),
        },
        "cpu": {
            "details": _system_value(system_environment, "cpu_info"),
        },
        "accelerators": {
            "cuda_available": _system_value(system_environment, "is_cuda_available"),
            "cuda_module_loading": _system_value(
                system_environment, "cuda_module_loading"
            ),
            "nvidia_gpu_models": _system_value(system_environment, "nvidia_gpu_models"),
            "xpu_available": _system_value(system_environment, "is_xpu_available"),
        },
        "native_libraries": {
            "libc_version": _system_value(system_environment, "libc_version"),
            "cuda_runtime_version": _system_value(
                system_environment, "cuda_runtime_version"
            ),
            "nvidia_driver_version": _system_value(
                system_environment, "nvidia_driver_version"
            ),
            "cudnn_version": _system_value(system_environment, "cudnn_version"),
            "hip_runtime_version": _system_value(
                system_environment, "hip_runtime_version"
            ),
            "miopen_runtime_version": _system_value(
                system_environment, "miopen_runtime_version"
            ),
            "xnnpack_available": _system_value(
                system_environment, "is_xnnpack_available"
            ),
        },
        "collection_errors": collection_errors,
    }


def write_environment_report(
    *,
    log_dir: str,
    run_type: str,
    timestamp_id: str,
    commit: str,
    job_id: str | None,
    array_job_id: str | None,
    array_task_id: str | None,
    restart_count: str | None,
) -> Path | None:
    """
    Write an environment report from local rank zero on each node.

    Returns:
        The report path on node leaders, otherwise ``None``.
    """
    try:
        local_rank = _get_local_rank()
    except Exception as error:
        logging.warning(
            "Failed to determine local rank for environment report (%s)",
            type(error).__name__,
        )
        return None

    if local_rank != 0:
        return None

    try:
        report = _to_yaml_safe(
            collect_environment_report(
                run_type=run_type,
                timestamp_id=timestamp_id,
                commit=commit,
                job_id=job_id,
                array_job_id=array_job_id,
                array_task_id=array_task_id,
                restart_count=restart_count,
            )
        )
        report_dir = Path(log_dir) / "environment"
        report_dir.mkdir(parents=True, exist_ok=True)

        execution_id = (
            f"{array_job_id}_{array_task_id}"
            if array_job_id is not None and array_task_id is not None
            else job_id or timestamp_id
        )
        filename = "_".join(
            (
                _safe_filename_component(run_type),
                _safe_filename_component(execution_id),
                f"node_{_safe_filename_component(_get_node_id())}",
                f"restart_{_safe_filename_component(restart_count or '0')}",
            )
        )
        report_path = report_dir / f"{filename}.yaml"

        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=report_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as report_file:
                yaml.dump(
                    report,
                    report_file,
                    Dumper=_EnvironmentDumper,
                    default_flow_style=False,
                    sort_keys=False,
                )
            os.replace(temporary_path, report_path)
        except Exception:
            Path(temporary_path).unlink(missing_ok=True)
            raise

        logging.info(f"Wrote environment report to {report_path}")
        return report_path
    except Exception as error:
        logging.warning("Failed to write environment report (%s)", type(error).__name__)
        return None
