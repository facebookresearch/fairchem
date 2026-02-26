"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Public Triton operations for UMA model.

This module provides torch.autograd.Function classes for GPU-accelerated
operations in the UMA backbone. All operations have custom backward passes
optimized for force computation.

Main Operations:
    - UMASFastGPUNodeToEdgeWignerPermute: Forward gather + rotate
    - UMASFastGPUPermuteWignerInvEdgeToNode: Backward rotate (M->L + Wigner)
"""

from __future__ import annotations

import torch
import triton

from ._kernels.gather_wigner_bwd import wigner_transform_bwd_kernel
from ._kernels.gather_wigner_fwd import fused_node_to_edge_wigner_permute_kernel
from ._kernels.wigner_transform import (
    fused_m_to_l_wigner_lmax2_kernel,
    fused_wigner_bwd_dx_l_to_m_kernel,
    wigner_lmax2_bwd_dw_kernel,
)
from .constants import BLOCK_C, M_TO_L_GATHER_IDX

__all__ = [
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]


class UMASFastGPUPermuteWignerInvEdgeToNode(torch.autograd.Function):
    """
    Autograd function for fused M->L + Wigner (lmax=2).

    Forward: Single kernel fuses M->L permutation + W @ x_l.
    Backward:
        dx_m = fused kernel (W^T @ dy + L->M permutation)
        dW = dy_l @ x_l^T (reuses existing kernel)

    Eliminates separate permutation kernel launches in both directions.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor:
        # Fused forward: M->L permutation + Wigner multiply in one kernel
        E, _, C = x.shape
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
        out = torch.empty_like(x)
        x_l = torch.empty_like(x)  # Save for backward
        fused_m_to_l_wigner_lmax2_kernel[(E, num_c_blocks)](
            x, wigner, out, x_l, E, C, BLOCK_C=BLOCK_C
        )
        ctx.save_for_backward(x_l, wigner)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_output = (
            grad_output.contiguous()
        )  # Grad tensors from autograd may not be contiguous
        x_l, wigner = ctx.saved_tensors
        E, _, C = grad_output.shape

        # Compute grad_x_m: Wigner^T @ grad_output + L->M permutation in one kernel
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
        grad_x_m = torch.empty_like(grad_output)
        fused_wigner_bwd_dx_l_to_m_kernel[(E, num_c_blocks)](
            grad_output, wigner, grad_x_m, E, C, BLOCK_C=BLOCK_C
        )

        # Compute dW = grad_output @ x_l^T (block-diagonal via Triton kernel)
        assert C == BLOCK_C, f"Only C={BLOCK_C} supported, got {C}"
        grad_wigner = torch.zeros(
            E, 9, 9, device=grad_output.device, dtype=grad_output.dtype
        )
        wigner_lmax2_bwd_dw_kernel[(E,)](grad_output, x_l, grad_wigner, C)
        return grad_x_m, grad_wigner


# =============================================================================
# Edge Gather + Wigner Transform Operations
# =============================================================================


def _fused_node_to_edge_wigner_permute_dx(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    V2 Triton backward: two-phase approach avoiding atomic contention.

    Phase 1: Triton kernel computes M→L + W^T @ grad, writes to [E, 9, 2C]
    Phase 2: PyTorch index_add_ scatters to nodes (uses segment reduction)

    This is ~2x faster than the atomic-based approach for high edge counts.

    Args:
        grad_output: Gradient from downstream [E, 9, 2C] in M-major order
        edge_index: Edge indices [2, E]
        wigner: Per-edge Wigner matrices [E, 81] or [E, 9, 9]
        num_nodes: Number of nodes N

    Returns:
        grad_x: Gradient w.r.t. input x [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    wigner_flat = wigner_flat.contiguous()

    # Phase 1: Compute per-edge gradients (no scatter)
    grad_edge = torch.empty(
        num_edges,
        9,
        2 * sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    BLOCK_C_BWD = triton.next_power_of_2(sphere_channels)

    wigner_transform_bwd_kernel[(num_edges,)](
        grad_output,
        wigner_flat,
        grad_edge,
        num_edges,
        sphere_channels,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_edge.stride(0),
        grad_edge.stride(1),
        grad_edge.stride(2),
        BLOCK_C=BLOCK_C_BWD,
    )

    # Phase 2: Scatter using PyTorch's optimized index_add_
    # This uses segment reduction instead of atomics
    grad_x = torch.zeros(
        num_nodes,
        9,
        sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    # Flatten for index_add_: [E, 9, C] -> [E, 9*C]
    grad_src = grad_edge[:, :, :sphere_channels].reshape(num_edges, -1)  # [E, 9*C]
    grad_tgt = grad_edge[:, :, sphere_channels:].reshape(num_edges, -1)  # [E, 9*C]
    grad_x_flat = grad_x.view(num_nodes, -1)  # [N, 9*C]

    # Scatter add: much faster than atomics due to segment reduction
    grad_x_flat.index_add_(0, edge_index[0], grad_src)
    grad_x_flat.index_add_(0, edge_index[1], grad_tgt)

    return grad_x


class UMASFastGPUNodeToEdgeWignerPermute(torch.autograd.Function):
    """
    Autograd function using the emit kernel that produces side outputs.

    The forward kernel emits x_edge [E, 9, 2C] (src at [:C], tgt at [C:2C])
    as a side output alongside the main rotated output [E, 9, 2C].
    This eliminates the redundant edge gather and torch.cat that the
    V2 forward does explicitly.

    Backward uses two bmm calls per L-block (K=2C) instead of four
    with K=C. Same total FLOPs, fewer kernel launches.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using emit Triton kernel.

        Single kernel call does gather + Wigner + L→M + side output writes.
        No explicit x[edge_index[0]], x[edge_index[1]], or torch.cat.
        """
        num_edges = edge_index.shape[1]
        num_nodes, num_coeffs, sphere_channels = x.shape

        # Flatten wigner if needed
        wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

        out = torch.empty(
            num_edges, num_coeffs, 2 * sphere_channels, device=x.device, dtype=x.dtype
        )
        x_edge = torch.empty(
            num_edges, num_coeffs, 2 * sphere_channels, device=x.device, dtype=x.dtype
        )

        # Use 2D grid: (edges, channel_blocks) to handle channels > BLOCK_C
        num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C

        fused_node_to_edge_wigner_permute_kernel[(num_edges, num_c_blocks)](
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
        )

        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = num_nodes
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward using concatenated x_edge [E, 9, 2C].

        Uses two bmm calls per L-block with K=2C instead of four with K=C.
        """
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: grad_x via two-phase scatter_add (unchanged)
        grad_x = _fused_node_to_edge_wigner_permute_dx(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: grad_wigner using concatenated x_edge
        grad_l = grad_output[:, M_TO_L_GATHER_IDX, :]  # [E, 9, 2C]

        E, _, _ = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1)
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3)
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :], x_edge[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block (5x5)
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :], x_edge[:, 4:9, :].transpose(1, 2)
        )

        return grad_x, None, grad_wigner
