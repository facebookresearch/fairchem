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

from fairchem.core.models.uma.triton.constants import BLOCK_C, M_TO_L_GATHER_IDX
from fairchem.core.models.uma.triton.kernels import (
    node_to_edge_wigner_permute_kernel,
)


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
        # Import here to avoid circular dependency
        import fairchem.core.models.uma.triton.custom_ops  # noqa: F401 - registers ops

        num_edges = edge_index.shape[1]
        sphere_channels = x.shape[2]

        # Allocation VISIBLE to torch.compile (can be optimized)
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

        # Ensure inputs are contiguous for Triton
        x = x.contiguous()

        # ONLY kernel launch is opaque (via custom_op with mutates_args)
        torch.ops.fairchem._kernel_node_to_edge_wigner_permute(
            x, edge_index, wigner, out, x_edge
        )

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
        num_edges = edge_index.shape[1]
        sphere_channels = grad_out.shape[2] // 2

        # Ensure grad_out is contiguous
        grad_out = grad_out.contiguous()

        # Allocation VISIBLE to torch.compile
        grad_edge = torch.empty(
            (num_edges, 9, sphere_channels * 2),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )

        # ONLY kernel launch is opaque
        torch.ops.fairchem._kernel_node_to_edge_wigner_permute_bwd_dx(
            grad_out, wigner, grad_edge
        )

        # Scatter (VISIBLE to torch.compile)
        grad_x = torch.zeros(
            (ctx.num_nodes, 9, sphere_channels),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )

        # Slice: grad_edge [E, 9, 2C] -> src at [:C], tgt at [C:]
        grad_src = grad_edge[:, :, :sphere_channels].reshape(
            num_edges, 9 * sphere_channels
        )
        grad_tgt = grad_edge[:, :, sphere_channels:].reshape(
            num_edges, 9 * sphere_channels
        )

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        grad_x_flat = grad_x.view(ctx.num_nodes, 9 * sphere_channels)
        grad_x_flat.index_add_(0, src_idx, grad_src)
        grad_x_flat.index_add_(0, tgt_idx, grad_tgt)

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
