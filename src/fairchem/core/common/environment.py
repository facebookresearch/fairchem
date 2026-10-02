"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import InitVar, asdict, dataclass, field, fields
from functools import cache
from pathlib import Path
from typing import Any

from submitit.slurm.slurm import SlurmJobEnvironment
from torch.utils.collect_env import SystemEnv, get_env_info

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_commit_hash
from fairchem.core.launchers.api import SlurmEnv

ENVIRONMENT_COLLECTION_TIMEOUT_SECONDS = 30

_COLLECT_ENVIRONMENT_SCRIPT = """
import json
from importlib import metadata

from torch.utils.collect_env import get_env_info

system_environment = get_env_info()._asdict()
system_environment["installed_python_packages"] = {
    name: distribution.version
    for distribution in metadata.distributions()
    if (name := distribution.metadata.get("Name"))
}
print(json.dumps(system_environment))
"""

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


def _get_local_rank() -> int:
    for variable in ("SLURM_LOCALID", "LOCAL_RANK"):
        if variable in os.environ:
            return int(os.environ[variable])
    return 0


def _get_node_id() -> str:
    return os.environ.get("SLURM_NODEID", "0")


def _get_global_rank() -> int:
    if not distutils.initialized() and "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    return distutils.get_rank()


def _get_world_size() -> int:
    if not distutils.initialized() and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return distutils.get_world_size()


def _get_slurm_env() -> SlurmEnv:
    """Return normalized Slurm metadata, or empty metadata for a local run."""
    slurm_job_environment = SlurmJobEnvironment()
    try:
        return SlurmEnv(
            job_id=slurm_job_environment.job_id,
            raw_job_id=slurm_job_environment.raw_job_id,
            array_job_id=slurm_job_environment.array_job_id,
            array_task_id=slurm_job_environment.array_task_id,
            restart_count=os.environ.get("SLURM_RESTART_COUNT"),
        )
    except KeyError:
        # Slurm environment variables are undefined for local runs.
        return SlurmEnv()


def _safe_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _collect_system_environment() -> tuple[dict[str, Any] | None, list[str]]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COLLECT_ENVIRONMENT_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=ENVIRONMENT_COLLECTION_TIMEOUT_SECONDS,
        )
        return json.loads(result.stdout), []
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
    ) as error:
        return None, [type(error).__name__]


def get_python_package_versions(system_environment: Any) -> dict[str, str]:
    """
    Extract installed Python package versions from PyTorch environment data.

    Args:
        system_environment: A ``SystemEnv`` instance or equivalent mapping.

    Returns:
        Package names mapped to versions in stable, case-insensitive order.
    """
    packages = {}
    pip_packages = str(_system_value(system_environment, "pip_packages") or "")
    for line in pip_packages.splitlines():
        package, separator, version = line.partition("==")
        if package and separator:
            packages[package] = version

    conda_packages = str(_system_value(system_environment, "conda_packages") or "")
    for line in conda_packages.splitlines():
        package_version = line.split()
        if len(package_version) >= 2:
            packages[package_version[0]] = package_version[1]

    installed_packages = _system_value(system_environment, "installed_python_packages")
    if isinstance(installed_packages, dict):
        packages.update(
            {
                str(package): str(version)
                for package, version in installed_packages.items()
            }
        )

    return dict(sorted(packages.items(), key=lambda package: package[0].casefold()))


def _system_value(system_environment: Any, name: str) -> Any:
    if system_environment is None:
        return None
    if isinstance(system_environment, dict):
        return system_environment.get(name)
    return getattr(system_environment, name, None)


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(nested_value) for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(nested_value) for nested_value in value]
    return str(value)


@dataclass
class EnvironmentChange:
    """Store one environment attribute change between two reports."""

    attribute: str
    value: Any
    baseline_value: Any

    def as_dict(self) -> dict[str, Any]:
        """Create a dictionary containing this change."""
        return asdict(self)


@dataclass
class EnvironmentChanges:
    """Store environment changes grouped by change type."""

    added: list[EnvironmentChange]
    removed: list[EnvironmentChange]
    changed: list[EnvironmentChange]
    unchanged: list[EnvironmentChange]

    def __post_init__(self) -> None:
        for changes in (self.added, self.removed, self.changed, self.unchanged):
            changes.sort(key=lambda change: change.attribute)

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Create a dictionary containing all grouped changes."""
        return {
            item.name: [change.as_dict() for change in getattr(self, item.name)]
            for item in fields(self)
        }


# Match the corresponding fields in the multiline output from ``lscpu``.
_LSCPU_CPU_COUNT_PATTERN = re.compile(r"(?:^|\n)\s*CPU\(s\):\s+([0-9]+)\s*(?:\r?\n|$)")
_LSCPU_CPU_MODEL_PATTERN = re.compile(r"(?:^|\n)\s*Model name:\s+(.*?)\s*(?:\r?\n|$)")
_COLLECT_CURRENT_ENVIRONMENT = object()


@dataclass
class Environment:
    """Store comparable information about a software and hardware environment.

    The field names preserve the performance-report schema. Additional launcher
    information is represented as fields in the same schema so reports written by
    FairChem can be compared by the existing performance-report tooling.
    """

    _system_environment: InitVar[Any] = _COLLECT_CURRENT_ENVIRONMENT

    git_commit_hash: str = field(init=False)

    pytorch_version: Any = field(init=False)
    pytorch_is_debug_build: Any = field(init=False)
    cuda_version_to_build_pytorch: Any = field(init=False)
    rocm_version_to_build_pytorch: Any = field(init=False)
    pytorch_caching_allocator_config: Any = field(init=False)

    os: Any = field(init=False)
    gcc_version: Any = field(init=False)
    clang_version: Any = field(init=False)
    cmake_version: Any = field(init=False)
    libc_version: Any = field(init=False)

    python_version: Any = field(init=False)
    python_platform: Any = field(init=False)
    cuda_runtime_version: Any = field(init=False)
    cuda_module_loading: Any = field(init=False)
    nvidia_driver_version: Any = field(init=False)
    cudnn_version: Any = field(init=False)
    hip_runtime_version: Any = field(init=False)
    miopen_runtime_version: Any = field(init=False)
    xnnpack_available: Any = field(init=False)

    libraries: dict[str, str] = field(init=False)

    num_gpus: str = field(init=False)
    gpu_model: str = field(init=False)
    nvidia_gpu_models: Any = field(init=False)
    num_cpus: str = field(init=False)
    cpu_model: str = field(init=False)
    cuda_available: Any = field(init=False)
    xpu_available: Any = field(init=False)
    environment_variables: dict[str, str] = field(init=False)

    @classmethod
    def from_system_environment(
        cls,
        system_environment: Any,
    ) -> Environment:
        """Build an environment from already collected PyTorch data."""
        return cls(_system_environment=system_environment)

    def __post_init__(self, _system_environment: Any) -> None:
        system_environment = (
            get_torch_env_info()
            if _system_environment is _COLLECT_CURRENT_ENVIRONMENT
            else _system_environment
        )

        self.git_commit_hash = self._get_git_commit_hash()

        self.pytorch_version = _system_value(system_environment, "torch_version")
        self.pytorch_is_debug_build = _system_value(
            system_environment, "is_debug_build"
        )
        self.cuda_version_to_build_pytorch = _system_value(
            system_environment, "cuda_compiled_version"
        )
        self.rocm_version_to_build_pytorch = _system_value(
            system_environment, "hip_compiled_version"
        ) or _system_value(system_environment, "rocm_compiled_version")
        self.pytorch_caching_allocator_config = _system_value(
            system_environment, "caching_allocator_config"
        )

        self.os = _system_value(system_environment, "os")
        self.gcc_version = _system_value(system_environment, "gcc_version")
        self.clang_version = _system_value(system_environment, "clang_version")
        self.cmake_version = _system_value(system_environment, "cmake_version")
        self.libc_version = _system_value(system_environment, "libc_version")

        self.python_version = _system_value(system_environment, "python_version")
        self.python_platform = _system_value(system_environment, "python_platform")
        self.cuda_runtime_version = _system_value(
            system_environment, "cuda_runtime_version"
        )
        self.cuda_module_loading = _system_value(
            system_environment, "cuda_module_loading"
        )
        self.nvidia_driver_version = _system_value(
            system_environment, "nvidia_driver_version"
        )
        self.cudnn_version = _system_value(system_environment, "cudnn_version")
        self.hip_runtime_version = _system_value(
            system_environment, "hip_runtime_version"
        )
        self.miopen_runtime_version = _system_value(
            system_environment, "miopen_runtime_version"
        )
        self.xnnpack_available = _system_value(
            system_environment, "is_xnnpack_available"
        )

        self.libraries = get_python_package_versions(system_environment)

        gpu_models_text = str(
            _system_value(system_environment, "nvidia_gpu_models") or ""
        )
        self.nvidia_gpu_models = _system_value(system_environment, "nvidia_gpu_models")
        gpu_model_lines = gpu_models_text.splitlines()
        self.num_gpus = str(len(gpu_model_lines))
        gpu_models = {
            parts[1].strip()
            for line in gpu_model_lines
            if len(parts := line.split(":", maxsplit=1)) > 1
        }
        self.gpu_model = next(iter(gpu_models)) if len(gpu_models) == 1 else "Unknown"

        cpu_info = str(_system_value(system_environment, "cpu_info") or "")
        self.num_cpus = (
            match.group(1)
            if (match := _LSCPU_CPU_COUNT_PATTERN.search(cpu_info))
            else "Unknown"
        )
        self.cpu_model = (
            match.group(1)
            if (match := _LSCPU_CPU_MODEL_PATTERN.search(cpu_info))
            else "Unknown"
        )

        self.cuda_available = _system_value(system_environment, "is_cuda_available")
        self.xpu_available = _system_value(system_environment, "is_xpu_available")
        self.environment_variables = {
            variable: os.environ[variable]
            for variable in ENVIRONMENT_VARIABLE_ALLOWLIST
            if variable in os.environ
        }

    @staticmethod
    def _get_git_commit_hash() -> str:
        """Return the current FairChem source revision."""
        return get_commit_hash() or "Unknown"

    def as_dict(self) -> dict[str, Any]:
        """Create a dictionary containing the comparable environment."""
        return _to_json_safe(asdict(self))

    @staticmethod
    def compare(
        target: dict[str, Any],
        baseline: dict[str, Any],
    ) -> EnvironmentChanges:
        """Compare dictionaries produced by :meth:`as_dict`."""
        all_attributes: set[str] = set()
        for name, value in itertools.chain(target.items(), baseline.items()):
            if isinstance(value, dict):
                all_attributes.update(f"{name}.{nested_name}" for nested_name in value)
            else:
                all_attributes.add(name)

        def get_value(environment: dict[str, Any], attribute: str) -> Any:
            path = attribute.split(".", maxsplit=1)
            value = environment.get(path[0])
            if len(path) == 2:
                return value.get(path[1]) if isinstance(value, dict) else None
            return value

        added: list[EnvironmentChange] = []
        removed: list[EnvironmentChange] = []
        changed: list[EnvironmentChange] = []
        unchanged: list[EnvironmentChange] = []
        for attribute in all_attributes:
            target_value = get_value(target, attribute)
            baseline_value = get_value(baseline, attribute)
            change = EnvironmentChange(attribute, target_value, baseline_value)
            if baseline_value is None and target_value is None:
                continue
            if baseline_value is None:
                added.append(change)
            elif target_value is None:
                removed.append(change)
            elif target_value != baseline_value:
                changed.append(change)
            else:
                unchanged.append(change)

        return EnvironmentChanges(
            added=added,
            removed=removed,
            changed=changed,
            unchanged=unchanged,
        )


@cache
def get_torch_env_info() -> SystemEnv:
    """Return and cache the environment information reported by PyTorch."""
    return get_env_info()


def collect_environment_report(
    run_type: str,
    timestamp_id: str,
    submission_commit: str,
) -> dict[str, Any]:
    """
    Collect diagnostic information for a FairChem process.

    Environment variables are restricted to a fixed allow-list so credentials and
    other secrets are not copied into the report.

    Args:
        run_type: FairChem operation being executed, such as run or reduce.
        timestamp_id: Logical run ID shared across nodes and restarts.
        submission_commit: FairChem revision captured when the job was configured.
    """
    system_environment, system_errors = _collect_system_environment()
    comparable_environment = Environment.from_system_environment(
        system_environment=system_environment,
    )
    slurm_environment = _get_slurm_env()
    return _to_json_safe(
        {
            "schema_version": 1,
            "environment": comparable_environment.as_dict(),
            "measurements": {},
            "job": {
                "run_type": run_type,
                "timestamp_id": timestamp_id,
                # This revision is captured when the job configuration is created.
                # It can differ from environment.git_commit_hash if a queued or
                # requeued job runs from a changed checkout, or if its worker imports
                # FairChem from a different installation.
                "commit": submission_commit,
                "job_id": slurm_environment.job_id,
                "array_job_id": slurm_environment.array_job_id,
                "array_task_id": slurm_environment.array_task_id,
                "restart_count": slurm_environment.restart_count,
            },
            "rank": {
                "global_rank": _get_global_rank(),
                "local_rank": _get_local_rank(),
                "world_size": _get_world_size(),
                "node_id": _get_node_id(),
                "hostname": socket.gethostname(),
            },
            "collection_errors": (
                {"system_environment": system_errors} if system_errors else {}
            ),
        }
    )


def write_environment_report(
    log_dir: str,
    run_type: str,
    timestamp_id: str,
    submission_commit: str,
) -> Path | None:
    """
    Write an environment report from local rank zero on each node.

    Args:
        log_dir: Configured directory in which to create the environment directory.
        run_type: FairChem operation being executed, such as run or reduce.
        timestamp_id: Logical run ID shared across nodes and restarts.
        submission_commit: FairChem revision captured when the job was configured.

    Returns:
        The report path on node leaders, otherwise ``None``.
    """
    try:
        local_rank = _get_local_rank()
    except ValueError as error:
        logging.warning(
            "Failed to determine local rank for environment report (%s)",
            type(error).__name__,
        )
        return None

    if local_rank != 0:
        return None

    try:
        report = collect_environment_report(
            run_type=run_type,
            timestamp_id=timestamp_id,
            submission_commit=submission_commit,
        )
        report_dir = Path(log_dir) / "environment"
        report_dir.mkdir(parents=True, exist_ok=True)

        job_metadata = report["job"]
        execution_id = (
            f"{job_metadata['array_job_id']}_{job_metadata['array_task_id']}"
            if job_metadata["array_job_id"] is not None
            and job_metadata["array_task_id"] is not None
            else job_metadata["job_id"] or timestamp_id
        )
        filename = "_".join(
            (
                _safe_filename_component(run_type),
                _safe_filename_component(execution_id),
                f"node_{_safe_filename_component(_get_node_id())}",
                f"restart_{_safe_filename_component(job_metadata['restart_count'] or '0')}",
            )
        )
        report_path = report_dir / f"{filename}.json"

        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=report_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=4)
                report_file.write("\n")
            os.replace(temporary_path, report_path)
        finally:
            Path(temporary_path).unlink(missing_ok=True)

        logging.info("Wrote environment report to %s", report_path)
        return report_path
    except (OSError, TypeError, ValueError) as error:
        logging.warning("Failed to write environment report (%s)", type(error).__name__)
        return None
