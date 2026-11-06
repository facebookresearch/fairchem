"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import platform
import shutil
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

import torch

_TRITON_SEED_CACHE = "FAIRCHEM_TRITON_SEED_CACHE_DIR"


def execution_environment(device: str) -> dict[str, Any]:
    """
    Describe the execution architecture without node-specific identity.
    """
    parsed = torch.device(device)
    result: dict[str, Any] = {
        "device_type": parsed.type,
        "cpu_architecture": platform.machine(),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "accelerator_name": None,
        "compute_capability": None,
    }
    if parsed.type == "cuda":
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        properties = torch.cuda.get_device_properties(index)
        result.update(
            accelerator_name=properties.name,
            compute_capability=[properties.major, properties.minor],
        )
    return result


def prepare_compilation_cache() -> None:
    """
    Put Triton and TorchInductor caches on node-local storage.

    Shared home-directory caches can mix architectures and race when many
    array tasks compile concurrently. A compatible prebuilt Triton CUDA helper
    is copied first because some compute images lack Python development
    headers. Explicit user settings are preserved.
    """
    root = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    # Local tasks share their allocation's job ID, so they use their process.
    name = (
        f"{os.environ.get('SLURM_JOB_ID')}-{task_id}"
        if task_id
        else f"local-{os.getpid()}"
    )
    cache_root = root / "fairchem-regression" / name
    inductor = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", cache_root / "torchinductor")
    ).expanduser()
    inductor.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)

    if "TRITON_CACHE_DIR" in os.environ:
        triton = Path(os.environ["TRITON_CACHE_DIR"]).expanduser()
        triton.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton)
        return

    extension = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(extension, str) or not extension:
        return
    seed_root = Path(
        os.environ.get(_TRITON_SEED_CACHE, Path.home() / ".triton" / "cache")
    ).expanduser()
    seed_helpers = sorted(seed_root.glob(f"*/cuda_utils{extension}"))
    if not seed_helpers:
        return
    triton = cache_root / "triton"
    for source in seed_helpers:
        destination = triton / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
