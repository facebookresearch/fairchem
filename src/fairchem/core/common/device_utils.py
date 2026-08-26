"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Accelerator-agnostic device helpers.

fairchem historically hard-coded ``cuda`` as *the* accelerator: ``torch.cuda.*``
calls, ``nccl`` as the only non-CPU collective backend, and asserts of the form
``device in ["cpu", "cuda"]``. That is portable only to NVIDIA hardware.

This module centralises the device-type question so the rest of the codebase can
say "the accelerator" instead of "cuda". It supports:

  * ``cuda`` -- NVIDIA, via ``torch.cuda`` + NCCL.
  * ``xpu``  -- Intel GPUs (Data Center GPU Max), via ``torch.xpu`` + XCCL
    (oneCCL upstreamed into PyTorch).

Everything here is built on PyTorch's own generic device APIs
(``torch.get_device_module``, ``torch.accelerator``, ``torch.amp.autocast``), so
adding a further backend is a matter of extending the tables below rather than
touching call sites.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "ACCELERATOR_DEVICE_TYPES",
    "SUPPORTED_DEVICE_TYPES",
    "accelerator_is_available",
    "current_device_str",
    "device_count",
    "device_module",
    "distributed_backend",
    "empty_cache",
    "get_available_accelerator",
    "is_accelerator",
    "manual_seed_all",
    "max_memory_allocated",
    "memory_allocated",
    "reset_peak_memory_stats",
    "resolve_device_type",
    "set_device",
    "supports_native_all_to_all",
    "synchronize",
    "triton_accelerator_enabled",
    "visible_devices_env",
]

# Non-CPU device types this codebase knows how to drive, in autodetection order.
ACCELERATOR_DEVICE_TYPES: tuple[str, ...] = ("cuda", "xpu")
SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cpu", *ACCELERATOR_DEVICE_TYPES)

# Collective backend per device type.
#
# NCCL is NVIDIA-only. Intel GPUs use **oneCCL**, which is reachable two ways:
#
#   "xccl" -- oneCCL upstreamed into PyTorch as a native backend. This is the
#             modern path and the one PyTorch itself selects for xpu (see
#             torch.distributed.Backend.default_device_backend_map). Despite the
#             name it *is* oneCCL: libtorch_xpu.so links libccl.so directly.
#   "ccl"  -- the legacy out-of-tree bindings (torch-ccl /
#             oneccl_bindings_for_pytorch), which must be imported before
#             init_process_group to register themselves. Recent PyTorch
#             distributions no longer ship them, so this is only a fallback.
_DISTRIBUTED_BACKENDS: dict[str, str] = {
    "cpu": "gloo",
    "cuda": "nccl",
    "xpu": "xccl",
}

# Legacy backend names tried when the preferred one is unavailable.
_FALLBACK_BACKENDS: dict[str, tuple[str, ...]] = {
    "xpu": ("ccl",),
}


def _legacy_ccl_registered() -> bool:
    """Whether the out-of-tree oneCCL bindings are importable."""
    import importlib.util

    return any(
        importlib.util.find_spec(m) is not None
        for m in ("oneccl_bindings_for_pytorch", "torch_ccl")
    )


# Vendor env vars that mask which physical devices a process may see. Logged on
# startup because "wrong rank-to-device binding" and "device not visible at all"
# are otherwise indistinguishable from a hang.
_VISIBLE_DEVICE_ENV_VARS: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDA_VISIBLE_DEVICES",),
    "xpu": ("ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR", "ZE_FLAT_DEVICE_HIERARCHY"),
}

# Env var allowing a site to force a device type (useful on machines that expose
# more than one accelerator, or to pin CPU for debugging).
FAIRCHEM_DEVICE_TYPE_ENV = "FAIRCHEM_DEVICE_TYPE"


def device_type_of(device: str | torch.device) -> str:
    """Return the bare device type ('cuda', 'xpu', 'cpu') for a device spec.

    Accepts 'cuda', 'cuda:0', torch.device('xpu:1'), etc. A string torch cannot
    parse at all raises ValueError rather than torch's RuntimeError, so callers
    can treat "bad device name" and "unsupported device name" the same way.
    """
    try:
        return torch.device(device).type
    except RuntimeError as exc:
        raise ValueError(f"unrecognised device specification: {device!r}") from exc


def is_accelerator(device: str | torch.device) -> bool:
    """Whether ``device`` names a non-CPU accelerator this module supports."""
    return device_type_of(device) in ACCELERATOR_DEVICE_TYPES


def device_module(device: str | torch.device) -> ModuleType:
    """Return the ``torch.<device_type>`` submodule (``torch.cuda``/``torch.xpu``).

    Uses ``torch.get_device_module`` so no per-backend branching is needed.
    """
    return torch.get_device_module(device_type_of(device))


def accelerator_is_available(device: str | torch.device) -> bool:
    """Whether the given accelerator type is present AND usable right now.

    Note this is deliberately stricter than ``torch.xpu`` merely importing:
    ``torch.xpu`` exists in any XPU-enabled build regardless of whether the node
    actually has a device. A silent CPU fall-through is a much worse failure
    mode than a loud error, because it looks like a working but inexplicably
    slow run.
    """
    device_type = device_type_of(device)
    if device_type == "cpu":
        return True
    if device_type not in ACCELERATOR_DEVICE_TYPES:
        return False
    if not hasattr(torch, device_type):
        return False
    try:
        return bool(device_module(device_type).is_available())
    except Exception:  # - a broken backend must read as "absent"
        return False


def get_available_accelerator() -> str | None:
    """Autodetect the accelerator to use, or None if this is a CPU-only node.

    Honours ``$FAIRCHEM_DEVICE_TYPE`` first, then prefers PyTorch's own
    ``torch.accelerator.current_accelerator()``, then falls back to probing
    the known types in order.
    """
    forced = os.environ.get(FAIRCHEM_DEVICE_TYPE_ENV)
    if forced:
        forced = forced.strip().lower()
        if forced == "cpu":
            return None
        if forced not in ACCELERATOR_DEVICE_TYPES:
            raise ValueError(
                f"{FAIRCHEM_DEVICE_TYPE_ENV}={forced!r} is not one of "
                f"{list(SUPPORTED_DEVICE_TYPES)}"
            )
        if not accelerator_is_available(forced):
            raise RuntimeError(
                f"{FAIRCHEM_DEVICE_TYPE_ENV}={forced!r} was requested but "
                f"torch.{forced}.is_available() is False on this node."
            )
        return forced

    if hasattr(torch, "accelerator"):
        try:
            current = torch.accelerator.current_accelerator()
            if (
                current is not None
                and current.type in ACCELERATOR_DEVICE_TYPES
                and accelerator_is_available(current.type)
            ):
                return current.type
        except Exception:  # - fall through to explicit probing
            pass

    return next(
        (dt for dt in ACCELERATOR_DEVICE_TYPES if accelerator_is_available(dt)),
        None,
    )


def resolve_device_type(device: str | torch.device | None) -> str:
    """Normalise a user-supplied device request into a concrete device type.

    ``None`` or ``"auto"`` autodetects. An explicit accelerator request is
    validated against the hardware so a typo or a wrong-node submission fails
    here rather than silently running on CPU.
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return get_available_accelerator() or "cpu"

    device_type = device_type_of(device)
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"device must be one of {list(SUPPORTED_DEVICE_TYPES)}, got {device!r}"
        )
    if device_type != "cpu" and not accelerator_is_available(device_type):
        raise RuntimeError(
            f"device={device!r} was requested but torch.{device_type}.is_available() "
            f"is False. Available accelerator: {get_available_accelerator() or 'none'}."
        )
    return device_type


def set_device(device: str | torch.device, local_rank: int) -> None:
    """Bind this process to ``local_rank`` on the given accelerator."""
    device_type = device_type_of(device)
    if device_type == "cpu":
        return
    device_module(device_type).set_device(local_rank)


def current_device_str(device: str | torch.device) -> str:
    """Return e.g. 'cuda:0' / 'xpu:3' for the currently-selected device."""
    device_type = device_type_of(device)
    if device_type == "cpu":
        return "cpu"
    return f"{device_type}:{device_module(device_type).current_device()}"


def device_count(device: str | torch.device | None = None) -> int:
    """Number of visible devices of the given (or autodetected) type."""
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return 0
    return device_module(device_type).device_count()


def empty_cache(device: str | torch.device) -> None:
    """Release cached device memory, if the backend supports it."""
    device_type = device_type_of(device)
    if device_type == "cpu":
        return
    module = device_module(device_type)
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def synchronize(device: str | torch.device) -> None:
    """Block until all queued work on the device has completed."""
    device_type = device_type_of(device)
    if device_type == "cpu":
        return
    device_module(device_type).synchronize()


def manual_seed_all(seed: int, device: str | torch.device | None = None) -> None:
    """Seed all devices of the given (or autodetected) accelerator type."""
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return
    module = device_module(device_type)
    if hasattr(module, "manual_seed_all"):
        module.manual_seed_all(seed)


def distributed_backend(device: str | torch.device) -> str:
    """Collective backend matching the device type ('nccl'/'xccl'/'gloo')."""
    device_type = device_type_of(device)
    backend = _DISTRIBUTED_BACKENDS.get(device_type)
    if backend is None:
        raise ValueError(
            f"no distributed backend known for device type {device_type!r}"
        )
    if backend == "gloo":
        return backend

    available = getattr(torch.distributed, f"is_{backend}_available", None)
    if available is None or available():
        return backend

    # Preferred backend missing -- try the legacy name before demoting to gloo.
    for legacy in _FALLBACK_BACKENDS.get(device_type, ()):
        if legacy == "ccl" and _legacy_ccl_registered():
            logging.warning(
                "%s is unavailable in this PyTorch build; using the out-of-tree "
                "oneCCL bindings (%r) instead. Import oneccl_bindings_for_pytorch "
                "before init_process_group so the backend registers.",
                backend,
                legacy,
            )
            return legacy

    logging.warning(
        "No %s collective backend is available in this PyTorch build; falling back "
        "to gloo. Collectives will be staged through host memory, which is slow "
        "for %s tensors.",
        device_type,
        device_type,
    )
    return "gloo"


def visible_devices_env(device: str | torch.device | None = None) -> str:
    """Human-readable summary of the vendor device-masking env vars.

    CUDA uses ``CUDA_VISIBLE_DEVICES``; Intel GPUs use ``ZE_AFFINITY_MASK`` and
    ``ONEAPI_DEVICE_SELECTOR``. Logging the wrong vendor's variable produces a
    confident "None" that hides a real misconfiguration, so pick by device type.
    """
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    names = _VISIBLE_DEVICE_ENV_VARS.get(device_type or "", ())
    if not names:
        return "None"
    present = [f"{n}={os.environ[n]}" for n in names if n in os.environ]
    return ", ".join(present) if present else "None"


def triton_accelerator_enabled(device: str | torch.device | None = None) -> bool:
    """Whether Triton kernels are opted in for a non-CUDA accelerator.

    fairchem's fused kernels are written and autotuned for NVIDIA. Triton itself
    is portable (Intel ships triton-xpu), but a kernel that compiles on another
    backend is not thereby correct or fast: it may rely on backend-specific
    intrinsics, and its ``num_warps``/``num_stages`` configs encode NVIDIA
    occupancy assumptions. So these stay opt-in per backend, via
    ``FAIRCHEM_ENABLE_TRITON_<DEVICE>=1`` (e.g. ``FAIRCHEM_ENABLE_TRITON_XPU``).

    CUDA is always considered enabled -- it is the path the kernels target.
    """
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return False
    if device_type == "cuda":
        return True
    flag = os.environ.get(f"FAIRCHEM_ENABLE_TRITON_{device_type.upper()}", "")
    return flag.strip().lower() in ("1", "true", "yes", "on")


def reset_peak_memory_stats(device: str | torch.device | None = None) -> None:
    """Reset the peak-memory counter, if the backend tracks one."""
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return
    module = device_module(device_type)
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats()


def max_memory_allocated(device: str | torch.device | None = None) -> int:
    """Peak bytes allocated on the device since the last reset (0 on CPU)."""
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return 0
    module = device_module(device_type)
    return (
        module.max_memory_allocated() if hasattr(module, "max_memory_allocated") else 0
    )


def memory_allocated(device: str | torch.device | None = None) -> int:
    """Bytes currently allocated on the device (0 on CPU)."""
    device_type = (
        device_type_of(device) if device is not None else get_available_accelerator()
    )
    if device_type is None or device_type == "cpu":
        return 0
    module = device_module(device_type)
    return module.memory_allocated() if hasattr(module, "memory_allocated") else 0


# Collective backends implementing all_to_all / all_to_all_single natively.
# Gloo does not, so callers must fall back to pairwise send/recv there.
_NATIVE_ALL_TO_ALL_BACKENDS = frozenset({"nccl", "xccl", "ccl", "mpi"})


def supports_native_all_to_all(backend: str) -> bool:
    """Whether ``backend`` implements all_to_all without a send/recv fallback.

    NCCL and oneCCL (``xccl``/``ccl``) do; gloo does not. Checking the backend
    by capability rather than by name keeps the fast path available on Intel
    GPUs instead of silently demoting them to pairwise send/recv.
    """
    return str(backend).lower() in _NATIVE_ALL_TO_ALL_BACKENDS
