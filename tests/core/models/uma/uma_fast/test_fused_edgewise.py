"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Correctness of the edgewise fusions used by the
        UMA-S fast-GPU backend at lmax=mmax=2:
          - producer: wigner_conv1_fused (gather + Wigner + L→M + radial
            scale/pack into the three conv1 GEMM buffers)
          - consumer: wigner_inv_conv2_fused (M→L unpack of the conv2 GEMM
            buffers + inverse-Wigner rotate)
          - packed gate: SiLU/sigmoid activation between conv1 and conv2
        Kernel-vs-PyTorch-reference forward tests, Wigner autograd gradcheck,
        and packed-gate first- and second-order derivative checks.
CI:     test_gpu_sweep (units shard).
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.activation import GateActivation
from fairchem.core.models.uma.triton import (
    packed_gate_op,
    wigner_conv1_fused_op,
    wigner_inv_conv2_fused_op,
    wigner_inv_conv2_scatter_op,
)
from fairchem.core.models.uma.triton.constants import M_TO_L_GATHER_IDX
from tests.core.models.uma.uma_fast.triton_test_utils import (
    wigner_conv1_fused_fwd_launcher,
    wigner_inv_conv2_fused_fwd_launcher,
)

# L_TO_M_GATHER_IDX is the inverse of M_TO_L_GATHER_IDX (test refs only).
L_TO_M_GATHER_IDX = [0] * 9
for _i, _val in enumerate(M_TO_L_GATHER_IDX):
    L_TO_M_GATHER_IDX[_val] = _i

# conv1 packs 9 M-rows into three buffers as [m0={M0,M1,M2}, m1={M3..M6}, m2={M7,M8}].
_M_SPLIT_SIZES = [3, 4, 2]


def _create_block_diagonal_wigner(num_edges: int, device: str, dtype=torch.float32):
    """
    Create block-diagonal Wigner matrix [E, 9, 9].

    Structure: L=0 (1x1), L=1 (3x3), L=2 (5x5).
    """
    wigner = torch.zeros(num_edges, 9, 9, device=device, dtype=dtype)
    wigner[:, 0, 0] = torch.randn(num_edges, device=device, dtype=dtype)
    wigner[:, 1:4, 1:4] = torch.randn(num_edges, 3, 3, device=device, dtype=dtype)
    wigner[:, 4:9, 4:9] = torch.randn(num_edges, 5, 5, device=device, dtype=dtype)
    return wigner


# =============================================================================
# Tests: producer conv1 fused kernel vs PyTorch reference
# =============================================================================


def _ref_wigner_conv1_pack(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    radial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference for the producer conv1 fusion.

    Mirrors node_to_edge_wigner_permute (gather + block-diagonal Wigner rotate +
    L→M permute) followed by SO2_Conv1_WithRadialBlock scale/pack up to (not
    including) the GEMMs.

    Args:
        x: Node features [N, 9, C] (L-major).
        edge_index: [2, E].
        wigner: [E, 9, 9].
        radial: conv1 radial embedding [E, 6*2C].

    Returns:
        m0 [E, 3*2C], m1 [E, 4*2C], m2 [E, 2*2C].
    """
    num_edges = edge_index.shape[1]
    C = x.shape[2]
    C2 = 2 * C

    # Gather + Wigner rotate (L-order) + concat src||tgt on channels.
    rot_src = torch.bmm(wigner, x[edge_index[0]])
    rot_tgt = torch.bmm(wigner, x[edge_index[1]])
    rot_src_m = rot_src[:, L_TO_M_GATHER_IDX, :]
    rot_tgt_m = rot_tgt[:, L_TO_M_GATHER_IDX, :]
    x_message = torch.cat([rot_src_m, rot_tgt_m], dim=-1)  # [E, 9, 2C]

    # Radial split: 6 blocks of 2C each -> m0 uses 3, m1 uses 2, m2 uses 1.
    edge_split_sizes = [3 * C2, 2 * C2, C2]
    x_edge_by_m = radial.split(edge_split_sizes, dim=1)
    x_by_m = x_message.split(_M_SPLIT_SIZES, dim=1)

    m0 = x_by_m[0].reshape(num_edges, -1) * x_edge_by_m[0]
    x1 = x_by_m[1].view(num_edges, 2, -1) * x_edge_by_m[1].unsqueeze(1)
    m1 = x1.flatten(1)
    x2 = x_by_m[2].view(num_edges, 2, -1) * x_edge_by_m[2].unsqueeze(1)
    m2 = x2.flatten(1)
    return m0, m1, m2


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256])
def test_wigner_conv1_fused_matches_pytorch(sphere_channels):
    """
    Verify the producer conv1 fused kernel matches the PyTorch reference.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_nodes = 16
    num_edges = 32
    C2 = 2 * sphere_channels

    x = torch.randn(num_nodes, 9, sphere_channels, device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_tgt = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)
    wigner = _create_block_diagonal_wigner(num_edges, device)
    radial = torch.randn(num_edges, 6 * C2, device=device)

    ref_m0, ref_m1, ref_m2 = _ref_wigner_conv1_pack(x, edge_index, wigner, radial)
    m0, m1, m2 = wigner_conv1_fused_fwd_launcher(x, edge_index, wigner, radial)

    assert torch.allclose(
        ref_m0, m0, rtol=1e-4, atol=1e-4
    ), f"m0 max diff: {(ref_m0 - m0).abs().max()}"
    assert torch.allclose(
        ref_m1, m1, rtol=1e-4, atol=1e-4
    ), f"m1 max diff: {(ref_m1 - m1).abs().max()}"
    assert torch.allclose(
        ref_m2, m2, rtol=1e-4, atol=1e-4
    ), f"m2 max diff: {(ref_m2 - m2).abs().max()}"


# =============================================================================
# Tests: consumer conv2 inv fused kernel vs PyTorch reference
# =============================================================================


def _ref_wigner_inv_conv2(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    wigner_inv: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference for the consumer conv2 inv fusion.

    Rebuilds the M-major x_message [E, 9, C] from the three conv2 GEMM buffers
    (view/unbind/cat, mirroring SO2_Conv2_InternalBlock), then applies M→L
    permute + block-diagonal inverse-Wigner rotate.

    Args:
        g0: conv2 fc_m0 output [E, 3C].
        g1: conv2 m=1 block-GEMM output [E, 4C].
        g2: conv2 m=2 block-GEMM output [E, 2C].
        wigner_inv: [E, 9, 9].

    Returns:
        x_rotated [E, 9, C] (L-major).
    """
    E = g0.shape[0]
    C = g0.shape[1] // 3
    out = [g0.view(E, 3, C)]
    r1, i1 = g1.view(E, 2, 2, C).unbind(1)
    out.append(r1)
    out.append(i1)
    r2, i2 = g2.view(E, 2, 1, C).unbind(1)
    out.append(r2)
    out.append(i2)
    x_message = torch.cat(out, dim=1)  # [E, 9, C] M-major

    x_l = x_message[:, M_TO_L_GATHER_IDX, :]
    return torch.bmm(wigner_inv, x_l)


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256])
def test_wigner_inv_conv2_fused_matches_pytorch(sphere_channels):
    """
    Verify the consumer conv2 inv fused kernel matches the PyTorch reference.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_edges = 32
    C = sphere_channels

    g0 = torch.randn(num_edges, 3 * C, device=device)
    g1 = torch.randn(num_edges, 4 * C, device=device)
    g2 = torch.randn(num_edges, 2 * C, device=device)
    wigner_inv = _create_block_diagonal_wigner(num_edges, device)

    ref_out = _ref_wigner_inv_conv2(g0, g1, g2, wigner_inv)
    triton_out = wigner_inv_conv2_fused_fwd_launcher(g0, g1, g2, wigner_inv)

    assert torch.allclose(
        ref_out, triton_out, rtol=1e-4, atol=1e-4
    ), f"Max diff: {(ref_out - triton_out).abs().max()}"


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256])
def test_wigner_inv_conv2_scatter_matches_materialized(sphere_channels):
    torch.manual_seed(42)
    num_nodes, num_edges, channels = 8, 32, sphere_channels
    inputs = (
        torch.randn(num_edges, 3 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 4 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 2 * channels, device="cuda", requires_grad=True),
        _create_block_diagonal_wigner(num_edges, "cuda").requires_grad_(),
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs
    )
    scatter_target = torch.randint(0, num_nodes, (num_edges,), device="cuda")

    output = wigner_inv_conv2_scatter_op(*inputs, scatter_target, num_nodes, channels)
    edge_output = wigner_inv_conv2_fused_op(*reference_inputs, channels)
    reference = torch.zeros_like(output).index_add_(0, scatter_target, edge_output)
    grad_output = torch.randn_like(output)
    grads = torch.autograd.grad(output, inputs, grad_output)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu()
def test_fused_edgewise_empty_graph():
    channels = 128
    x = torch.randn(2, 9, channels, device="cuda", requires_grad=True)
    edge_index = torch.empty(2, 0, dtype=torch.long, device="cuda")
    wigner = torch.empty(0, 9, 9, device="cuda", requires_grad=True)
    radial = torch.empty(0, 12 * channels, device="cuda", requires_grad=True)

    conv1_outputs = wigner_conv1_fused_op(x, edge_index, wigner, radial, channels)
    sum(output.sum() for output in conv1_outputs).backward()
    assert [output.shape for output in conv1_outputs] == [
        (0, 6 * channels),
        (0, 8 * channels),
        (0, 4 * channels),
    ]
    assert torch.count_nonzero(x.grad) == 0
    assert wigner.grad.shape == wigner.shape
    assert radial.grad.shape == radial.shape

    g0 = torch.empty(0, 3 * channels, device="cuda", requires_grad=True)
    g1 = torch.empty(0, 4 * channels, device="cuda", requires_grad=True)
    g2 = torch.empty(0, 2 * channels, device="cuda", requires_grad=True)
    wigner_inv = torch.empty(0, 9, 9, device="cuda", requires_grad=True)
    conv2_output = wigner_inv_conv2_fused_op(g0, g1, g2, wigner_inv, channels)
    conv2_output.sum().backward()
    assert conv2_output.shape == (0, 9, channels)
    assert g0.grad.shape == g0.shape
    assert g1.grad.shape == g1.shape
    assert g2.grad.shape == g2.shape
    assert wigner_inv.grad.shape == wigner_inv.shape

    scatter_target = torch.empty(0, dtype=torch.long, device="cuda")
    scattered = wigner_inv_conv2_scatter_op(
        g0, g1, g2, wigner_inv, scatter_target, 2, channels
    )
    scattered.sum().backward()
    assert scattered.shape == (2, 9, channels)
    assert torch.count_nonzero(scattered) == 0


# =============================================================================
# Tests: autograd gradcheck for both fused custom ops
# =============================================================================


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256])
def test_wigner_conv1_fused_gradcheck(sphere_channels):
    """
    Verify the producer conv1 fused op backward via gradcheck.

    Checks grads wrt node features, Wigner (block-diagonal), and radial. Uses
    fast_mode=True for statistical gradient validation to avoid full-Jacobian
    OOM. sphere_channels must be a multiple of BLOCK_C=128.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_nodes = 8
    num_edges = 16
    C = sphere_channels
    C2 = 2 * C

    x = torch.randn(num_nodes, 9, C, device=device, dtype=torch.float64).requires_grad_(
        True
    )
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_tgt = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)
    radial = torch.randn(
        num_edges, 6 * C2, device=device, dtype=torch.float64
    ).requires_grad_(True)

    def fn(x_in, w_in, r_in):
        return wigner_conv1_fused_op(x_in, edge_index, w_in, r_in, C)

    assert torch.autograd.gradcheck(
        fn,
        (x, wigner, radial),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-12,  # Node accumulation uses atomic adds.
    )


@pytest.mark.gpu()
def test_wigner_conv1_fused_deterministic_backward(
    torch_deterministic, compile_reset_state
):
    torch.manual_seed(42)
    num_nodes, num_edges, channels = 8, 32, 128
    x = torch.randn(num_nodes, 9, channels, device="cuda", requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device="cuda")
    wigner = _create_block_diagonal_wigner(num_edges, "cuda").requires_grad_()
    radial = torch.randn(num_edges, 12 * channels, device="cuda", requires_grad=True)
    compiled = torch.compile(wigner_conv1_fused_op, fullgraph=True, dynamic=True)
    outputs = compiled(x, edge_index, wigner, radial, channels)
    grad_outputs = tuple(torch.randn_like(output) for output in outputs)

    first = torch.autograd.grad(
        outputs, (x, wigner, radial), grad_outputs, retain_graph=True
    )
    second = torch.autograd.grad(outputs, (x, wigner, radial), grad_outputs)

    for first_grad, second_grad in zip(first, second):
        assert torch.equal(first_grad, second_grad)


@pytest.mark.gpu()
def test_wigner_conv1_fused_dynamic_compile(compile_reset_state):
    torch.manual_seed(42)
    num_nodes, channels = 8, 128
    compiled = torch.compile(wigner_conv1_fused_op, fullgraph=True, dynamic=True)

    for num_edges in (17, 31):
        x = torch.randn(num_nodes, 9, channels, device="cuda", requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device="cuda")
        wigner = _create_block_diagonal_wigner(num_edges, "cuda").requires_grad_()
        radial = torch.randn(
            num_edges, 12 * channels, device="cuda", requires_grad=True
        )
        outputs = compiled(x, edge_index, wigner, radial, channels)
        grad_outputs = tuple(torch.randn_like(output) for output in outputs)
        grads = torch.autograd.grad(outputs, (x, wigner, radial), grad_outputs)

        assert tuple(output.shape[0] for output in outputs) == (num_edges,) * 3
        assert tuple(grad.shape for grad in grads) == (
            x.shape,
            wigner.shape,
            radial.shape,
        )


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256])
def test_wigner_inv_conv2_fused_gradcheck(sphere_channels):
    """
    Verify the consumer conv2 inv fused op backward via gradcheck.

    Checks grads wrt the three conv2 GEMM buffers and Wigner (block-diagonal).
    Uses fast_mode=True. sphere_channels must be a multiple of BLOCK_C=128.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_edges = 16
    C = sphere_channels

    g0 = torch.randn(
        num_edges, 3 * C, device=device, dtype=torch.float64
    ).requires_grad_(True)
    g1 = torch.randn(
        num_edges, 4 * C, device=device, dtype=torch.float64
    ).requires_grad_(True)
    g2 = torch.randn(
        num_edges, 2 * C, device=device, dtype=torch.float64
    ).requires_grad_(True)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)

    def fn(a, b, c, w_in):
        return wigner_inv_conv2_fused_op(a, b, c, w_in, C)

    assert torch.autograd.gradcheck(
        fn,
        (g0, g1, g2, wigner),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )


@pytest.mark.gpu()
def test_wigner_inv_conv2_scatter_dynamic_compile(compile_reset_state):
    torch.manual_seed(42)
    num_nodes, channels = 8, 128
    compiled = torch.compile(wigner_inv_conv2_scatter_op, fullgraph=True, dynamic=True)

    for num_edges in (17, 31):
        inputs = (
            torch.randn(num_edges, 3 * channels, device="cuda", requires_grad=True),
            torch.randn(num_edges, 4 * channels, device="cuda", requires_grad=True),
            torch.randn(num_edges, 2 * channels, device="cuda", requires_grad=True),
            _create_block_diagonal_wigner(num_edges, "cuda").requires_grad_(),
        )
        scatter_target = torch.randint(0, num_nodes, (num_edges,), device="cuda")
        output = compiled(*inputs, scatter_target, num_nodes, channels)
        grads = torch.autograd.grad(output, inputs, torch.randn_like(output))

        assert output.shape == (num_nodes, 9, channels)
        assert tuple(grad.shape for grad in grads) == tuple(
            value.shape for value in inputs
        )


@pytest.mark.gpu()
def test_wigner_inv_conv2_scatter_deterministic(
    torch_deterministic, compile_reset_state
):
    torch.manual_seed(42)
    num_nodes, num_edges, channels = 8, 32, 128
    inputs = (
        torch.randn(num_edges, 3 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 4 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 2 * channels, device="cuda", requires_grad=True),
        _create_block_diagonal_wigner(num_edges, "cuda").requires_grad_(),
    )
    scatter_target = torch.randint(0, num_nodes, (num_edges,), device="cuda")
    compiled = torch.compile(wigner_inv_conv2_scatter_op, fullgraph=True, dynamic=True)
    first = compiled(*inputs, scatter_target, num_nodes, channels)
    second = compiled(*inputs, scatter_target, num_nodes, channels)
    grad_output = torch.randn_like(first)
    first_grads = torch.autograd.grad(first, inputs, grad_output)
    second_grads = torch.autograd.grad(second, inputs, grad_output)

    assert torch.equal(first, second)
    for first_grad, second_grad in zip(first_grads, second_grads, strict=True):
        assert torch.equal(first_grad, second_grad)


def _ref_packed_gate(x0_full, x1, x2, channels):
    num_edges = x0_full.shape[0]
    activation = GateActivation(2, 2, channels, m_prime=True).cuda()
    ref_message = torch.cat(
        (
            x0_full[:, 2 * channels :].view(num_edges, 3, channels),
            x1.view(num_edges, 4, channels),
            x2.view(num_edges, 2, channels),
        ),
        dim=1,
    )
    return tuple(
        value.flatten(1)
        for value in activation(x0_full[:, : 2 * channels], ref_message).split(
            (3, 4, 2), dim=1
        )
    )


@pytest.mark.gpu()
@pytest.mark.parametrize("num_edges", [0, 32])
def test_packed_gate_matches_materialized_activation(num_edges):
    torch.manual_seed(42)
    channels = 128
    x0_backing = torch.randn(
        num_edges, 12 * channels, device="cuda", requires_grad=True
    )
    x1_backing = torch.randn(num_edges, 8 * channels, device="cuda", requires_grad=True)
    x2_backing = torch.randn(num_edges, 4 * channels, device="cuda", requires_grad=True)
    x0_full = x0_backing[:, : 5 * channels]
    x1 = x1_backing[:, : 4 * channels]
    x2 = x2_backing[:, : 2 * channels]

    ref_x0 = x0_full.detach().clone().requires_grad_()
    ref_x1 = x1.detach().clone().requires_grad_()
    ref_x2 = x2.detach().clone().requires_grad_()
    ref_outputs = _ref_packed_gate(ref_x0, ref_x1, ref_x2, channels)
    grad_outputs = tuple(torch.randn_like(value) for value in ref_outputs)
    ref_grads = torch.autograd.grad(ref_outputs, (ref_x0, ref_x1, ref_x2), grad_outputs)

    outputs = packed_gate_op(x0_full, x1, x2, channels)
    grads = torch.autograd.grad(outputs, (x0_full, x1, x2), grad_outputs)
    for actual, expected in zip(outputs, ref_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu()
def test_packed_gate_second_order_matches_materialized_activation():
    torch.manual_seed(42)
    num_edges, channels = 7, 128
    inputs = (
        torch.randn(num_edges, 5 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 4 * channels, device="cuda", requires_grad=True),
        torch.randn(num_edges, 2 * channels, device="cuda", requires_grad=True),
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs
    )
    outputs = packed_gate_op(*inputs, channels)
    reference_outputs = _ref_packed_gate(*reference_inputs, channels)
    grad_outputs = tuple(torch.randn_like(output) for output in outputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True)
    reference_grads = torch.autograd.grad(
        reference_outputs,
        reference_inputs,
        grad_outputs,
        create_graph=True,
    )
    vectors = tuple(torch.randn_like(value) for value in inputs)
    second_grads = torch.autograd.grad(grads, inputs, vectors)
    reference_second_grads = torch.autograd.grad(
        reference_grads, reference_inputs, vectors
    )

    for actual, expected in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(second_grads, reference_second_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.gpu()
def test_packed_gate_dynamic_compile(compile_reset_state):
    channels = 128
    compiled = torch.compile(packed_gate_op, fullgraph=True, dynamic=True)
    for num_edges in (17, 31):
        inputs = (
            torch.randn(
                num_edges,
                5 * channels,
                device="cuda",
                requires_grad=True,
            ),
            torch.randn(
                num_edges,
                4 * channels,
                device="cuda",
                requires_grad=True,
            ),
            torch.randn(
                num_edges,
                2 * channels,
                device="cuda",
                requires_grad=True,
            ),
        )
        reference_inputs = tuple(
            value.detach().clone().requires_grad_() for value in inputs
        )
        grad_outputs = (
            torch.randn(num_edges, 3 * channels, device="cuda"),
            torch.randn_like(inputs[1]),
            torch.randn_like(inputs[2]),
        )

        outputs = compiled(*inputs, channels)
        reference_outputs = packed_gate_op(*reference_inputs, channels)
        grads = torch.autograd.grad(outputs, inputs, grad_outputs)
        reference_grads = torch.autograd.grad(
            reference_outputs, reference_inputs, grad_outputs
        )
        for actual, expected in zip(outputs, reference_outputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(grads, reference_grads, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
