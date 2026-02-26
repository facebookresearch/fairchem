"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Public Triton operations for UMA model.

This module provides torch.autograd.Function classes for GPU-accelerated
Wigner D-matrix operations in the UMA backbone. All operations have custom
backward passes optimized for force computation (gradients w.r.t. positions).

Classes:
    UMASFastGPUNodeToEdgeWignerPermute
        Forward: node features → gather → Wigner rotate → L→M permute → edge
        Backward: edge grad → M→L permute → W^T → scatter → node grad
        Used for: Edge message passing in equivariant layers

    UMASFastGPUPermuteWignerInvEdgeToNode
        Forward: edge features → M→L permute → Wigner rotate → edge (inverse)
        Backward: edge grad → W^T → L→M permute → edge grad
        Used for: Inverse rotation back to node-aligned frame
"""

from __future__ import annotations

import torch
import triton

from ._kernels.node_to_edge_wigner_l2m import node_to_edge_wigner_l2m_kernel
from ._kernels.wigner_l2m_bwd import wigner_l2m_bwd_kernel
from ._kernels.wigner_m2l import wigner_m2l_kernel
from ._kernels.wigner_m2l_bwd import wigner_m2l_bwd_kernel
from ._kernels.wigner_weight_bwd import wigner_weight_bwd_kernel
from .constants import BLOCK_C, M_TO_L_GATHER_IDX

__all__ = [
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]


class UMASFastGPUPermuteWignerInvEdgeToNode(torch.autograd.Function):
    """
    Autograd function for M→L permutation + Wigner rotation.

    This is the "inverse" rotation operation that transforms edge features
    back from the edge-aligned frame to the node-aligned frame.

    Forward:
        x_m[E, 9, C] → permute M→L → W @ x_l → y_l[E, 9, C]
        Single kernel fuses permutation and rotation.

    Backward:
        dy_l → W^T @ dy → permute L→M → dx_m  (via wigner_l2m_bwd_kernel)
        dW = dy_l @ x_l^T  (via wigner_weight_bwd_kernel)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor:
        E, _, C = x.shape
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
        out = torch.empty_like(x)
        x_l = torch.empty_like(x)  # Save for backward dW computation
        wigner_m2l_kernel[(E, num_c_blocks)](x, wigner, out, x_l, E, C, BLOCK_C=BLOCK_C)
        ctx.save_for_backward(x_l, wigner)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Grad tensors from autograd may not be contiguous
        grad_output = grad_output.contiguous()
        x_l, wigner = ctx.saved_tensors
        E, _, C = grad_output.shape

        # Compute grad_x_m: W^T @ dy + L→M permutation in one kernel
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
        grad_x_m = torch.empty_like(grad_output)
        wigner_l2m_bwd_kernel[(E, num_c_blocks)](
            grad_output, wigner, grad_x_m, E, C, BLOCK_C=BLOCK_C
        )

        # Compute dW = grad_output @ x_l^T (block-diagonal via Triton kernel)
        assert C == BLOCK_C, f"Only C={BLOCK_C} supported, got {C}"
        grad_wigner = torch.zeros(
            E, 9, 9, device=grad_output.device, dtype=grad_output.dtype
        )
        wigner_weight_bwd_kernel[(E,)](grad_output, x_l, grad_wigner, C)
        return grad_x_m, grad_wigner


# =============================================================================
# Node-to-Edge Wigner Transform Operation
# =============================================================================


def _node_to_edge_backward_dx(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Backward pass for node_to_edge_wigner_l2m: compute dx via two-phase scatter.

    Phase 1: Triton kernel computes M→L permutation + W^T @ grad per edge
    Phase 2: PyTorch index_add_ scatters edge gradients to nodes

    The two-phase approach avoids atomic contention in Triton, which is the
    main bottleneck for high edge counts. PyTorch's index_add_ uses segment
    reduction which is ~2x faster than Triton atomics.

    Args:
        grad_output: Upstream gradient [E, 9, 2C] in M-major order
        edge_index: Edge indices [2, E] (row 0 = src, row 1 = tgt)
        wigner: Per-edge Wigner matrices [E, 81] or [E, 9, 9]
        num_nodes: Number of nodes N

    Returns:
        grad_x: Gradient w.r.t. input node features [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    wigner_flat = wigner_flat.contiguous()

    # Phase 1: Compute per-edge gradients via M→L + W^T
    grad_edge = torch.empty(
        num_edges,
        9,
        2 * sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    BLOCK_C_BWD = triton.next_power_of_2(sphere_channels)

    wigner_m2l_bwd_kernel[(num_edges,)](
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
    # Uses segment reduction instead of atomics (much faster)
    grad_x = torch.zeros(
        num_nodes,
        9,
        sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    # Flatten for index_add_: [E, 9, C] → [E, 9*C]
    grad_src = grad_edge[:, :, :sphere_channels].reshape(num_edges, -1)  # [E, 9*C]
    grad_tgt = grad_edge[:, :, sphere_channels:].reshape(num_edges, -1)  # [E, 9*C]
    grad_x_flat = grad_x.view(num_nodes, -1)  # [N, 9*C]

    # Scatter add: accumulate gradients from all edges to their source/target nodes
    grad_x_flat.index_add_(0, edge_index[0], grad_src)
    grad_x_flat.index_add_(0, edge_index[1], grad_tgt)

    return grad_x


class UMASFastGPUNodeToEdgeWignerPermute(torch.autograd.Function):
    """
    Autograd function for node-to-edge gather + Wigner rotation + L→M permutation.

    This is the main forward operation in the message passing layer. It gathers
    node features onto edges, applies the per-edge Wigner rotation to align
    features with the edge direction, and permutes to M-major ordering.

    Forward:
        x[N, 9, C] → gather[edge_index] → W @ x → permute L→M → out[E, 9, 2C]
        Side output: x_edge[E, 9, 2C] (pre-rotation, saved for backward dW)

    Backward:
        grad_x: via _node_to_edge_backward_dx (two-phase scatter)
        grad_wigner: via bmm on saved x_edge
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward: single kernel does gather + Wigner + L→M + side output.
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

        # 2D grid: (edges, channel_blocks) to handle channels > BLOCK_C
        num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C

        node_to_edge_wigner_l2m_kernel[(num_edges, num_c_blocks)](
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
        Backward: compute grad_x and grad_wigner.

        grad_x: Two-phase scatter via wigner_m2l_bwd_kernel + index_add_
        grad_wigner: bmm on saved x_edge (concatenated src/tgt, K=2C)
        """
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # grad_x via two-phase scatter
        grad_x = _node_to_edge_backward_dx(grad_output, edge_index, wigner, num_nodes)

        # grad_wigner: dW = grad_l @ x_edge^T (block-diagonal)
        grad_l = grad_output[:, M_TO_L_GATHER_IDX, :]  # [E, 9, 2C]

        E, _, _ = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1): dW[0,0] = sum_c grad[0,c] * x[0,c]
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3): dW[1:4, 1:4] = grad[1:4, :] @ x[1:4, :]^T
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :], x_edge[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block (5x5): dW[4:9, 4:9] = grad[4:9, :] @ x[4:9, :]^T
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :], x_edge[:, 4:9, :].transpose(1, 2)
        )

        return grad_x, None, grad_wigner
