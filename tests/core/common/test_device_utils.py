"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the accelerator-agnostic device layer.

These run everywhere: the CPU-only assertions execute on any machine, and the
accelerator assertions skip cleanly when no GPU is present.
"""

from __future__ import annotations

import types

import pytest
import torch

from fairchem.core.common import device_utils as du

ACCEL = du.get_available_accelerator()
requires_accelerator = pytest.mark.skipif(
    ACCEL is None, reason="no cuda/xpu accelerator on this node"
)


# --------------------------------------------------------------------------
# device-type plumbing (CPU-safe)
# --------------------------------------------------------------------------


def test_supported_device_types_include_xpu():
    assert "xpu" in du.ACCELERATOR_DEVICE_TYPES
    assert "cuda" in du.ACCELERATOR_DEVICE_TYPES
    assert set(du.SUPPORTED_DEVICE_TYPES) == {"cpu", *du.ACCELERATOR_DEVICE_TYPES}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("cuda:1", "cuda"),
        ("xpu", "xpu"),
        ("xpu:3", "xpu"),
    ],
)
def test_device_type_of_strips_index(spec, expected):
    assert du.device_type_of(spec) == expected


def test_is_accelerator_excludes_cpu():
    assert not du.is_accelerator("cpu")
    assert du.is_accelerator("xpu:0")
    assert du.is_accelerator("cuda:0")


def test_resolve_device_type_rejects_unknown():
    with pytest.raises(ValueError):
        du.resolve_device_type("tpu")


def test_resolve_device_type_cpu_always_allowed():
    assert du.resolve_device_type("cpu") == "cpu"


def test_cpu_helpers_are_noops():
    # These must not raise on a CPU-only machine.
    du.set_device("cpu", 0)
    du.empty_cache("cpu")
    du.synchronize("cpu")
    du.reset_peak_memory_stats("cpu")
    assert du.current_device_str("cpu") == "cpu"
    assert du.device_count("cpu") == 0
    assert du.max_memory_allocated("cpu") == 0
    assert du.memory_allocated("cpu") == 0


# --------------------------------------------------------------------------
# collective backend selection
# --------------------------------------------------------------------------


def test_cpu_backend_is_gloo():
    assert du.distributed_backend("cpu") == "gloo"


def test_xpu_prefers_oneccl_when_available():
    """xpu must map to oneCCL ("xccl"), never to NCCL.

    "xccl" is oneCCL upstreamed into PyTorch -- libtorch_xpu.so links libccl.so
    directly. The legacy out-of-tree name is "ccl". Either is acceptable; gloo
    is the only allowed fallback, and NCCL never is.
    """
    backend = du.distributed_backend("xpu")
    assert backend in ("xccl", "ccl", "gloo")
    assert backend != "nccl"
    if torch.distributed.is_xccl_available():
        assert backend == "xccl"


def test_unknown_device_has_no_backend():
    with pytest.raises(ValueError):
        du.distributed_backend("tpu")


@pytest.mark.parametrize(
    ("backend", "native"),
    [("nccl", True), ("xccl", True), ("ccl", True), ("gloo", False)],
)
def test_all_to_all_capability(backend, native):
    assert du.supports_native_all_to_all(backend) is native


# --------------------------------------------------------------------------
# real-hardware checks
# --------------------------------------------------------------------------


@requires_accelerator
def test_detected_accelerator_is_usable():
    assert du.accelerator_is_available(ACCEL)
    assert du.device_count(ACCEL) >= 1
    assert du.current_device_str(ACCEL).startswith(f"{ACCEL}:")


@requires_accelerator
def test_auto_resolves_to_detected_accelerator():
    assert du.resolve_device_type("auto") == ACCEL
    assert du.resolve_device_type(None) == ACCEL


@requires_accelerator
def test_absent_accelerator_is_refused_not_silently_downgraded():
    """Requesting a device this node lacks must raise, not fall back to CPU.

    Stock fairchem resolved anything that was not "cuda" to CPU, which turns a
    wrong-node submission into a silently slow run rather than an error.
    """
    absent = [
        d for d in du.ACCELERATOR_DEVICE_TYPES if not du.accelerator_is_available(d)
    ]
    if not absent:
        pytest.skip("this node has every supported accelerator")
    with pytest.raises(RuntimeError):
        du.resolve_device_type(absent[0])


@requires_accelerator
def test_roundtrip_tensor_and_memory_accounting():
    du.reset_peak_memory_stats(ACCEL)
    x = torch.randn(1024, 1024, device=ACCEL)
    y = (x @ x.T).sum()
    du.synchronize(ACCEL)
    assert y.device.type == ACCEL
    assert du.max_memory_allocated(ACCEL) > 0
    del x, y
    du.empty_cache(ACCEL)


@requires_accelerator
def test_autograd_runs_on_accelerator():
    """A missing autograd kernel yields a model that infers but cannot train."""
    x = torch.randn(256, 256, device=ACCEL, requires_grad=True)
    (x * x).sum().backward()
    du.synchronize(ACCEL)
    assert x.grad is not None
    assert torch.allclose(x.grad, 2 * x.detach(), atol=1e-5)


# --------------------------------------------------------------------------
# device index handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("cuda", None), ("cuda:0", 0), ("xpu", None), ("xpu:3", 3), ("cpu", None)],
)
def test_device_index_of(spec, expected):
    assert du.device_index_of(spec) == expected


def test_device_index_of_none():
    assert du.device_index_of(None) is None


def test_device_index_of_rejects_garbage():
    with pytest.raises(ValueError):
        du.device_index_of("not-a-device")


def test_missing_device_api_raises_loudly(monkeypatch):
    """An incomplete backend must fail, not silently report zero.

    The previous hasattr() guards returned 0 bytes / skipped seeding, which
    reads as success while the run is actually unmeasured or unseeded.
    """
    monkeypatch.setattr(du, "device_module", lambda d: types.SimpleNamespace())
    with pytest.raises(AttributeError, match="memory_allocated"):
        du._device_api("xpu", "memory_allocated")


def test_required_apis_present_on_both_backends():
    """Every call the device layer needs exists on cuda and xpu alike."""
    required = (
        "empty_cache",
        "manual_seed_all",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "memory_allocated",
        "synchronize",
        "set_device",
        "current_device",
        "device_count",
        "is_available",
    )
    for device_type in du.ACCELERATOR_DEVICE_TYPES:
        module = torch.get_device_module(device_type)
        missing = [name for name in required if not hasattr(module, name)]
        assert not missing, f"torch.{device_type} lacks {missing}"


@requires_accelerator
def test_memory_queries_are_per_device():
    """An indexed spec must report THAT device, not the current one.

    Regression test: the wrappers used to call torch's per-device memory APIs
    with no argument, so ``memory_allocated("xpu:1")`` silently reported
    device 0 -- reading as 0 bytes while memory was in fact allocated.
    """
    if du.device_count(ACCEL) < 2:
        pytest.skip("needs at least 2 devices to tell them apart")

    big = torch.empty(4096, 4096, dtype=torch.float32, device=f"{ACCEL}:1")
    du.synchronize(ACCEL)
    try:
        on_one = du.memory_allocated(f"{ACCEL}:1")
        assert on_one >= big.numel() * big.element_size()
        # Bare type follows the current device, which is not device 1.
        assert du.memory_allocated(ACCEL) == du.memory_allocated(f"{ACCEL}:0")
        assert du.max_memory_allocated(f"{ACCEL}:1") >= on_one
    finally:
        del big
        du.empty_cache(ACCEL)
