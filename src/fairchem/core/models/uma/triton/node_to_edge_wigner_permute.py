"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Node-to-edge gather + Wigner transform + L→M permutation operation.

This operation is the first step in the edge message passing pipeline:
1. Gather node features for source and target (via edge_index)
2. Apply block-diagonal Wigner rotation
3. Permute from L-major to M-major ordering

Public API:
- NodeToEdgeWignerPermuteFunction: torch.autograd.Function for the full operation
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.triton.kernels import (
    node_to_edge_wigner_permute_bwd_dx_kernel,
    node_to_edge_wigner_permute_kernel,
)

# Block size for channel vectorization
BLOCK_C = 128

# M→L permutation index (used in backward)
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]


def node_to_edge_wigner_permute_launcher(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward launcher: gather + Wigner + L→M permute.

    Args:
        x: Node features [N, 9, C] in L-major order
        edge_index: Edge indices [2, E]
        wigner: Wigner matrices [E, 9, 9] (block-diagonal structure)

    Returns:
        out: Rotated edge features [E, 9, 2C] in M-major order (src||tgt)
        x_edge: Pre-Wigner gathered features [E, 9, 2C] for backward dW
    """
    num_edges = edge_index.shape[1]
    sphere_channels = x.shape[2]

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

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
    )

    return out, x_edge


def node_to_edge_wigner_permute_bwd_dx_launcher(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Backward launcher w.r.t. input x: M→L + W^T @ grad + scatter.

    Args:
        grad_out: Gradient from downstream [E, 9, 2C] in M-major order
        wigner: Wigner matrices [E, 9, 9]
        edge_index: Edge indices [2, E]
        num_nodes: Number of nodes N

    Returns:
        grad_x: Gradient w.r.t. input [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_out.shape[2] // 2

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner
    wigner_flat = wigner_flat.contiguous()

    # Per-edge gradient buffer (no atomic contention)
    grad_edge = torch.empty(
        (num_edges, 9, sphere_channels * 2),
        dtype=grad_out.dtype,
        device=grad_out.device,
    )

    # Single block per edge (all channels in one pass)
    grid = (num_edges,)

    node_to_edge_wigner_permute_bwd_dx_kernel[grid](
        grad_out,
        wigner_flat,
        grad_edge,
        num_edges,
        sphere_channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_edge.stride(0),
        grad_edge.stride(1),
        grad_edge.stride(2),
        BLOCK_C=sphere_channels,  # Process all channels
    )

    # Efficient scatter using index_add_
    grad_x = torch.zeros(
        (num_nodes, 9, sphere_channels),
        dtype=grad_out.dtype,
        device=grad_out.device,
    )

    # Reshape for scatter: [E, 9*C] for src, [E, 9*C] for tgt
    grad_edge_flat = grad_edge.view(num_edges, 9 * sphere_channels * 2)
    grad_src = grad_edge_flat[:, : 9 * sphere_channels].reshape(
        num_edges, 9 * sphere_channels
    )
    grad_tgt = grad_edge_flat[:, 9 * sphere_channels :].reshape(
        num_edges, 9 * sphere_channels
    )

    src_idx = edge_index[0]  # [E]
    tgt_idx = edge_index[1]  # [E]

    # Scatter add to node gradients
    grad_x_flat = grad_x.view(num_nodes, 9 * sphere_channels)
    grad_x_flat.index_add_(0, src_idx, grad_src)
    grad_x_flat.index_add_(0, tgt_idx, grad_tgt)

    return grad_x


def _permute_m_to_l(x: torch.Tensor) -> torch.Tensor:
    """
    Permute from M-major to L-major ordering.

    Used in backward dW computation where we need L-major gradients.

    Args:
        x: Tensor with dim=1 of size 9 in M-major order

    Returns:
        Tensor in L-major order
    """
    return x[:, M_TO_L_GATHER_IDX, :]


class NodeToEdgeWignerPermuteFunction(torch.autograd.Function):
    """
    Autograd function for node-to-edge gather + Wigner + L→M permute.

    Forward: x[N,9,C] -> out[E,9,2C]
    Backward: Computes grad_x, grad_wigner
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, 9, C]
            edge_index: [2, E]
            wigner: [E, 9, 9]

        Returns:
            out: [E, 9, 2C] rotated edge features
        """
        out, x_edge = node_to_edge_wigner_permute_launcher(x, edge_index, wigner)
        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = x.shape[0]
        return out

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """
        Backward pass.

        Args:
            grad_out: [E, 9, 2C] gradient from downstream

        Returns:
            grad_x: [N, 9, C]
            None: edge_index has no gradient
            grad_wigner: [E, 9, 9]
        """
        x_edge, edge_index, wigner = ctx.saved_tensors

        # grad_x via Triton kernel + PyTorch scatter
        grad_x = node_to_edge_wigner_permute_bwd_dx_launcher(
            grad_out, wigner, edge_index, ctx.num_nodes
        )

        # grad_wigner = dy @ x^T using block-sparse structure
        # Convert grad to L-major for outer product
        grad_l = _permute_m_to_l(grad_out)  # [E, 9, 2C]

        E = x_edge.shape[0]
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
