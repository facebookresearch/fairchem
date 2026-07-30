"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test-only Triton kernel launchers.

These launchers directly invoke Triton kernels (bypassing @triton_op wrappers)
and use num_edges as GRID_E_STRIDE for simpler testing. Production code uses
the autograd Functions in the triton/ package which use fixed GRID_E_STRIDE=2048
for torch.compile compatibility.

These should NOT be used in production - only for unit testing kernel correctness.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.triton.constants import BLOCK_C
from fairchem.core.models.uma.triton.kernels import (
    node_to_edge_wigner_permute_kernel,
    permute_wigner_inv_edge_to_node_kernel,
    wigner_conv1_fused_fwd_kernel,
    wigner_inv_conv2_fused_fwd_kernel,
)


def node_to_edge_wigner_permute_launcher(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test-only launcher: gather + Wigner + L→M permute.

    Uses num_edges as GRID_E_STRIDE (not the production constant 2048).
    For testing kernel correctness, not for production use.

    Args:
        x: Node features [N, 9, C] in L-major order
        edge_index: Edge indices [2, E]
        wigner: Wigner matrices [E, 9, 9] (block-diagonal structure)

    Returns:
        out: Rotated edge features [E, 9, 2C] in M-major order (src||tgt)
        x_edge: Pre-Wigner gathered features [E, 9, 2C] for backward dW
    """
    # x: [N, 9, C] - node features with 9 coefficients (lmax=2)
    assert x.ndim == 3, "x must be 3D [N, 9, C]"
    assert x.shape[1] == 9, "x must have 9 coefficients (lmax=2)"
    # wigner: [E, 9, 9] - block-diagonal Wigner matrices
    assert wigner.ndim == 3, "wigner must be 3D [E, 9, 9]"
    assert wigner.shape[1] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.shape[2] == 9, "wigner must have shape [E, 9, 9]"
    # Wigner must be contiguous for flattening
    assert wigner.is_contiguous(), "wigner must be contiguous"

    num_edges = edge_index.shape[1]
    sphere_channels = x.shape[2]

    # Flatten wigner [E, 9, 9] -> [E, 81]
    wigner_flat = wigner.reshape(num_edges, -1)

    # Allocate outputs
    out = torch.empty(
        (num_edges, 9, sphere_channels * 2),
        dtype=x.dtype,
        device=x.device,
    )
    x_edge = torch.empty(
        (num_edges, 9, sphere_channels * 2),
        dtype=x.dtype,
        device=x.device,
    )

    # Grid: (edges, channel_blocks)
    num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C
    grid = (num_edges, num_c_blocks)

    # Use num_edges as GRID_E_STRIDE so each program handles exactly one edge
    node_to_edge_wigner_permute_kernel[grid](
        x,
        edge_index,
        wigner_flat,
        out,
        x_edge,
        num_edges,
        sphere_channels,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        edge_index.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        x_edge.stride(0),
        x_edge.stride(1),
        x_edge.stride(2),
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=num_edges,
    )

    return out, x_edge


def permute_wigner_inv_edge_to_node_launcher(
    x: torch.Tensor,
    wigner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test-only launcher: M→L permute + Wigner inverse.

    Uses E as GRID_E_STRIDE (not the production constant 2048).
    For testing kernel correctness, not for production use.

    Args:
        x: Edge features [E, 9, C] in M-major order
        wigner: Wigner inverse matrices [E, 9, 9]

    Returns:
        out: Rotated features [E, 9, C] in L-major order
        x_l: Permuted input [E, 9, C] (saved for backward dW computation)
    """
    # x: [E, 9, C] - edge features with 9 coefficients (lmax=2)
    assert x.ndim == 3, "x must be 3D [E, 9, C]"
    assert x.shape[1] == 9, "x must have 9 coefficients (lmax=2)"
    # wigner: [E, 9, 9] - block-diagonal Wigner inverse matrices
    assert wigner.ndim == 3, "wigner must be 3D [E, 9, 9]"
    assert wigner.shape[1] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.shape[2] == 9, "wigner must have shape [E, 9, 9]"
    # Contiguity required for memory access pattern
    assert x.is_contiguous(), "x must be contiguous"
    assert wigner.is_contiguous(), "wigner must be contiguous"

    E, num_coeffs, C = x.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    out = torch.empty_like(x)
    x_l = torch.empty_like(x)

    # Use E as GRID_E_STRIDE so each program handles exactly one edge
    permute_wigner_inv_edge_to_node_kernel[(E, num_c_blocks)](
        x,
        wigner,
        out,
        x_l,
        E,
        C,
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=E,
    )
    return out, x_l


def wigner_conv1_fused_fwd_launcher(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    radial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test-only launcher: producer-side fused wigner->conv1 forward.

    Gather + block-diagonal Wigner + L→M permute + per-m radial scale + pack
    into three GEMM-ready buffers. Uses num_edges as GRID_E_STRIDE (not the
    production constant 2048). For testing kernel correctness only.

    Args:
        x: Node features [N, 9, C] in L-major order.
        edge_index: Edge indices [2, E].
        wigner: Wigner matrices [E, 9, 9] (block-diagonal structure).
        radial: conv1 radial embedding [E, 6*2C].

    Returns:
        m0 [E, 3*2C], m1 [E, 4*2C], m2 [E, 2*2C].
    """
    assert x.ndim == 3, "x must be 3D [N, 9, C]"
    assert x.shape[1] == 9, "x must have 9 coefficients (lmax=2)"
    assert wigner.shape[1:] == (9, 9), "wigner must be [E, 9, 9]"

    num_edges = edge_index.shape[1]
    C = x.shape[2]
    C2 = 2 * C
    wigner_flat = wigner.reshape(num_edges, -1).contiguous()

    m0 = torch.empty((num_edges, 3 * C2), dtype=x.dtype, device=x.device)
    m1 = torch.empty((num_edges, 4 * C2), dtype=x.dtype, device=x.device)
    m2 = torch.empty((num_edges, 2 * C2), dtype=x.dtype, device=x.device)

    # Use num_edges as GRID_E_STRIDE so each program handles exactly one edge
    wigner_conv1_fused_fwd_kernel[(num_edges,)](
        x.contiguous(),
        edge_index,
        wigner_flat,
        radial.contiguous(),
        m0,
        m1,
        m2,
        num_edges,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        edge_index.stride(0),
        BLOCK_C=C,
        GRID_E_STRIDE=num_edges,
    )
    return m0, m1, m2


def wigner_inv_conv2_fused_fwd_launcher(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    Test-only launcher: consumer-side fused conv2 GEMM buffers -> inv-Wigner.

    Reads the three conv2 GEMM buffers directly (absorbing the M→L unpack) and
    applies the block-diagonal inverse-Wigner rotation. Uses num_edges as
    GRID_E_STRIDE. For testing kernel correctness only.

    Args:
        g0: conv2 fc_m0 output [E, 3C].
        g1: conv2 m=1 block-GEMM output [E, 4C].
        g2: conv2 m=2 block-GEMM output [E, 2C].
        wigner: inverse Wigner matrices [E, 9, 9].

    Returns:
        x_rotated [E, 9, C] in L-major order.
    """
    assert wigner.shape[1:] == (9, 9), "wigner must be [E, 9, 9]"
    E = g0.shape[0]
    C = g0.shape[1] // 3
    wigner_flat = wigner.reshape(E, -1).contiguous()

    out = torch.empty((E, 9, C), dtype=g0.dtype, device=g0.device)

    wigner_inv_conv2_fused_fwd_kernel[(E, 1)](
        g0.contiguous(),
        g1.contiguous(),
        g2.contiguous(),
        wigner_flat,
        out,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=E,
    )
    return out
