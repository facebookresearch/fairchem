"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib.resources
import os
import shutil
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, OmegaConf

from fairchem.core.launchers.api import SlurmConfig
from fairchem.core.regression_suite.spec.profiles import (
    CONFIGS,
    unsupported_suites_for_profile,
)
from fairchem.core.regression_suite.spec.schema import CommandError
from fairchem.core.regression_suite.suites import SUITES

if TYPE_CHECKING:
    from collections.abc import Sequence


def load_settings() -> DictConfig:
    """
    Load packaged command-line defaults with optional user overrides.

    ``FAIRCHEM_REGRESSION_CONFIG`` names a YAML file merged over the defaults.
    """
    settings = OmegaConf.load(importlib.resources.files(CONFIGS) / "settings.yaml")
    override = os.environ.get("FAIRCHEM_REGRESSION_CONFIG")
    if override:
        settings = OmegaConf.merge(settings, OmegaConf.load(override))
    return settings


def slurm_resources(settings: DictConfig, *, gpu: bool) -> SlurmConfig:
    """
    Build the SLURM resources used for GPU tasks or CPU-only follow-ups.
    """
    slurm = settings.slurm
    return SlurmConfig(
        partition=str(slurm.partition),
        qos=str(slurm.qos),
        account=str(slurm.account),
        mem_gb=int(slurm.gpu_mem_gb if gpu else slurm.cpu_mem_gb),
        timeout_hr=int(slurm.timeout_hr) if gpu else 1,
        cpus_per_task=int(slurm.cpus_per_task) if gpu else 2,
    )


def resolve_run_on(value: str) -> str:
    """
    Resolve ``auto`` to local inside a SLURM allocation or without sbatch.
    """
    if value != "auto":
        return value
    if os.environ.get("SLURM_JOB_ID") or shutil.which("sbatch") is None:
        return "local"
    return "slurm"


class DeviceUnavailableError(RuntimeError):
    """
    Raised before anything starts when a local run needs a GPU this host lacks.
    """


def resolve_device(device: str, run_on: str, *, source: str = "--device cuda") -> str:
    """
    Resolve a launch's device on the submitting host.

    ``auto`` is CUDA for SLURM jobs and, for local runs, CUDA when this host
    has a visible GPU and the CPU otherwise. ``source`` names what asked for
    the device in the error raised when a local CUDA run has no GPU.
    """
    local_gpu = run_on == "local" and torch.cuda.is_available()
    if device == "auto":
        return "cuda" if run_on == "slurm" or local_gpu else "cpu"
    if device == "cuda" and run_on == "local" and not local_gpu:
        raise DeviceUnavailableError(
            f"{source} needs a visible GPU to run locally, and this host has none"
        )
    return device


def selected_suites(
    configured: Sequence[str] | None,
    profile: str,
    available: Sequence[str] = SUITES,
) -> tuple[str, ...]:
    """
    Return the configured suites, or every available suite ``profile``
    supports, in suite order.

    ``available`` narrows the choice to the suites of a baseline.
    """
    unsupported = unsupported_suites_for_profile(profile)
    if configured is None:
        return tuple(suite for suite in available if suite not in unsupported)
    selected = {str(suite).replace("_", "-") for suite in configured}
    unknown = selected - set(SUITES)
    if unknown:
        raise CommandError(f"unknown suites: {sorted(unknown)}")
    missing = selected - set(available)
    if missing:
        raise CommandError(f"the baseline does not contain suites: {sorted(missing)}")
    if selected & unsupported:
        raise CommandError(
            f"profile {profile!r} does not support suites: "
            f"{sorted(selected & unsupported)}"
        )
    return tuple(suite for suite in available if suite in selected)
