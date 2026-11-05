"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unit tests for UMA Nitro kernel orchestration: capacity recovery, CUDA graph
caching, architecture kernel policy, and portable fallback operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")


@pytest.mark.parametrize("force_fp32", [True, False])
def test_uma_nitro_set_force_precision_toggles_flags(monkeypatch, force_fp32):
    """Set the force-path precision flags."""
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    for flag in inference._FORCE_PATH_FP16_FLAGS:
        monkeypatch.setattr(inference, flag, getattr(inference, flag))
    inference.set_force_precision(force_fp32)
    assert all(
        getattr(inference, flag) is (not force_fp32)
        for flag in inference._FORCE_PATH_FP16_FLAGS
    )


def test_uma_nitro_graph_flag_classifies_failures():
    """
    Retry only graph-capacity overflow and keep numerical failures fatal.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    cache_key = (4, "omat", "geometry")
    inference._check_graph_flag(torch.zeros(1, dtype=torch.int64), cache_key, 4)

    with pytest.raises(inference._GraphCapacityOverflow) as error:
        inference._check_graph_flag(torch.ones(1, dtype=torch.int64), cache_key, 4)
    assert error.value.cache_key == cache_key
    assert error.value.natoms == 4

    with pytest.raises(RuntimeError, match="non-finite model output"):
        inference._check_graph_flag(
            torch.full((1,), 2, dtype=torch.int64), cache_key, 4
        )


def test_uma_nitro_capacity_overflow_invalidates_and_retries(monkeypatch):
    """
    An overflow remeasures capacity and replaces the captured graph.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    cache_key = (4, "omat", "geometry")
    live_edges = {cache_key: (100, 24, 1)}
    result = {"energy": torch.tensor([1.0])}
    attempts = []
    flushes = []

    def run_once(_model, _data, _core):
        attempts.append(None)
        if len(attempts) == 1:
            raise inference._GraphCapacityOverflow(cache_key, 4)
        assert cache_key not in live_edges
        return result

    monkeypatch.setattr(inference, "_LIVE_EDGES", live_edges)
    monkeypatch.setattr(inference, "_GRAPH_STATE", {"old": object()})
    monkeypatch.setattr(inference, "_forward_once", run_once)
    monkeypatch.setattr(inference, "_flush_graphs", lambda: flushes.append(None))

    assert inference._nitro_forward(object(), object()) is result
    assert len(attempts) == 2
    assert len(flushes) == 1


def test_uma_nitro_capacity_high_water_reuses_and_grows():
    """
    Keep a fitting capacity across density changes and grow after overflow.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    backbone = SimpleNamespace()
    composition = b"same-composition"
    dense = (-10, composition)
    expanded = (-20, composition)
    initial = {"edge_capacity": 200, "cell_capacity": 24, "rep_max": 1}
    smaller = {"edge_capacity": 120, "cell_capacity": 16, "rep_max": 1}
    larger = {"edge_capacity": 240, "cell_capacity": 32, "rep_max": 2}

    assert (
        inference._capacity_high_water(backbone, 4, "omol", dense, initial, 130)
        == initial
    )
    assert (
        inference._capacity_high_water(backbone, 4, "omol", expanded, smaller, 80)
        == initial
    )
    assert (
        inference._capacity_high_water(backbone, 4, "omol", dense, larger, 201)
        == larger
    )


def test_uma_nitro_skips_empty_charge_channels(monkeypatch):
    """Models without charge channels leave the node embedding unchanged."""
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    inputs = torch.randn(4, 9, 8, requires_grad=True)
    scratch = torch.empty((2, 1, 0))

    def fail_if_launched(*_args):
        raise AssertionError("zero-width charge balance launched a custom kernel")

    monkeypatch.setattr(inference._BalanceCharge, "apply", fail_if_launched)
    output = inference._balance_charge(
        SimpleNamespace(charge_channel_start=0),
        inputs,
        torch.zeros(1),
        torch.tensor([4]),
        torch.zeros(4, dtype=torch.long),
        scratch,
        0,
    )

    assert output is inputs
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))


def test_uma_nitro_replaces_graph_when_key_changes(monkeypatch):
    """A new graph key releases the previous graph before capture."""
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    events = []

    class FakeGraph:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr(inference, "_GRAPH_STATE", {"old": (FakeGraph(), {})})
    monkeypatch.setattr(inference, "_GRAPH_POOL", object())
    monkeypatch.setattr(
        inference.torch.cuda, "synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(
        inference.torch.cuda, "empty_cache", lambda: events.append("empty")
    )
    new_state = (FakeGraph(), {})

    def capture(_build):
        assert not inference._GRAPH_STATE
        assert inference._GRAPH_POOL is None
        events.append("capture")
        return new_state

    monkeypatch.setattr(inference, "_capture", capture)

    assert inference._get_or_capture_graph("new", lambda: None) is new_state
    assert list(inference._GRAPH_STATE) == ["new"]
    assert events == ["sync", "reset", "empty", "capture"]


def test_uma_nitro_synchronizes_before_graph_reset(monkeypatch):
    """Graph teardown waits for asynchronous replay and output copies."""
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    events = []

    class FakeGraph:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr(inference, "_GRAPH_STATE", {"key": (FakeGraph(), {})})
    monkeypatch.setattr(inference, "_GRAPH_POOL", object())
    monkeypatch.setattr(
        inference.torch.cuda, "synchronize", lambda: events.append("sync")
    )
    monkeypatch.setattr(
        inference.torch.cuda, "empty_cache", lambda: events.append("empty")
    )

    inference._flush_graphs()

    assert events == ["sync", "reset", "empty"]
    assert not inference._GRAPH_STATE
    assert inference._GRAPH_POOL is None


def test_uma_nitro_reuses_graph_for_same_key(monkeypatch):
    """An unchanged graph key neither flushes nor captures."""
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    state = (object(), {})
    monkeypatch.setattr(inference, "_GRAPH_STATE", {"same": state})

    def fail(*_args):
        raise AssertionError("same-key lookup changed the graph")

    monkeypatch.setattr(inference, "_capture", fail)
    monkeypatch.setattr(inference, "_flush_graphs", fail)

    assert inference._get_or_capture_graph("same", lambda: None) is state


@pytest.mark.parametrize(
    (
        "capability",
        "expected_name",
        "expected_cute",
        "expected_hopper_gemm",
        "expected_tf32",
    ),
    [
        ((7, 0), "portable_pytorch_ieee", False, False, False),
        ((8, 0), "portable_pytorch_ieee", False, False, False),
        ((8, 6), "portable_pytorch_ieee", False, False, False),
        ((8, 9), "portable_pytorch_ieee", False, False, False),
        ((9, 0), "native_sm90_plus", True, True, True),
        ((10, 0), "blackwell_cute_cublas", True, False, True),
        ((10, 3), "blackwell_cute_cublas", True, False, True),
        ((11, 0), "blackwell_cute_cublas", True, False, True),
        ((12, 0), "blackwell_cute_cublas", True, False, True),
    ],
)
def test_uma_nitro_kernel_policy(
    capability,
    expected_name,
    expected_cute,
    expected_hopper_gemm,
    expected_tf32,
):
    """
    Use only architecture-compatible CuTe kernels.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    policy = inference._kernel_policy(capability)

    assert policy.name == expected_name
    assert policy.use_cute_kernels is expected_cute
    assert policy.use_hopper_gemm is expected_hopper_gemm
    assert policy.use_tf32 is expected_tf32
    assert policy.supports_tf32_round is expected_cute
    assert policy.supports_vector_atomics is expected_cute


def test_uma_nitro_effective_kernel_policy():
    """
    Keep the module-level primitive flags synchronized with capability policy.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    policy = inference._kernel_policy(inference._CUDA_CAPABILITY)

    assert policy == inference._KERNEL_POLICY
    assert inference._USE_CUTEDSL is policy.use_cute_kernels
    assert inference._USE_HOPPER_GEMM is policy.use_hopper_gemm
    assert inference._USE_TF32 is policy.use_tf32
    assert inference._SUPPORTS_TF32_ROUND is policy.supports_tf32_round
    assert inference._SUPPORTS_VECTOR_ATOMICS is policy.supports_vector_atomics
    assert policy.name in inference.kernel_backend_summary()


def test_uma_nitro_device_guard_accepts_sm80_without_cute(monkeypatch):
    """
    Ampere can run Nitro without selecting unsupported Hopper CuTe kernels.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    monkeypatch.setattr(inference, "_CUDA_CAPABILITY", (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    inference._validate_device("cuda")
    assert not inference._kernel_policy((8, 0)).use_cute_kernels
    assert not inference._kernel_policy((8, 0)).use_tf32


@pytest.mark.parametrize(
    ("use_tf32", "expected_precision"),
    [(False, "highest"), (True, "high")],
)
def test_uma_nitro_tf32_context_enforces_policy_and_restores(
    monkeypatch, use_tf32, expected_precision
):
    """
    Scope the architecture precision policy without leaking global state.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    monkeypatch.setattr(inference, "_USE_TF32", use_tf32)
    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )

    with inference._tf32_context():
        assert torch.backends.cuda.matmul.allow_tf32 is use_tf32
        assert torch.backends.cudnn.allow_tf32 is use_tf32
        assert torch.get_float32_matmul_precision() == expected_precision

    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    ) == original


@pytest.mark.gpu()
def test_uma_nitro_pytorch_fallback_ops():
    """
    Preserve native operation contracts with the portable implementations.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    torch.manual_seed(7)
    activations = torch.randn(17, 64, device="cuda", dtype=torch.float16)
    weights = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(32, device="cuda")
    row_scale = torch.randn(17, device="cuda")
    expected = (
        torch.mm(activations, weights.t(), out_dtype=torch.float32) * row_scale[:, None]
        + bias
    )
    storage = torch.empty(17, 48, device="cuda")
    output = storage[:, 8:40]

    inference._torch_mm_op(activations, weights, output, bias=bias, rscale=row_scale)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    source = torch.randn(6, 9, 8, device="cuda")
    index = torch.tensor([4, 1, 4, 0], device="cuda", dtype=torch.int32)
    gathered = torch.empty(4, 9, 8, device="cuda")
    inference._gather_node_v4_torch(source, index, gathered)
    torch.testing.assert_close(gathered, source[index.long()], rtol=0, atol=0)

    grad_edge = torch.randn(4, 9, 8, device="cuda", dtype=torch.float16)
    edge_scale = torch.randn(4, device="cuda")
    scattered = torch.zeros(6, 9, 8, device="cuda")
    expected_scatter = scattered.clone().index_add_(
        0, index, grad_edge.float() * edge_scale[:, None, None]
    )
    inference._scatter_src_v4_fp16_torch(grad_edge, index, edge_scale, scattered)
    torch.testing.assert_close(scattered, expected_scatter, rtol=0, atol=0)
