"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Correctness of the two producer/consumer edgewise fusions used by the
        UMA-S fast-GPU backend at lmax=mmax=2:
          - producer: wigner_conv1_fused (gather + Wigner + L→M + radial
            scale/pack into the three conv1 GEMM buffers)
          - consumer: wigner_inv_conv2_fused (M→L unpack of the conv2 GEMM
            buffers + inverse-Wigner rotate)
        Kernel-vs-PyTorch-reference forward tests plus autograd gradcheck for
        both custom ops (grads wrt node features / GEMM buffers, Wigner, radial).
CI:     test_gpu_sweep (units shard).
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.triton import (
    wigner_conv1_fused_op,
    wigner_inv_conv2_fused_op,
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
