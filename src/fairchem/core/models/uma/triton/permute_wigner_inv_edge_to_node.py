"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

M→L permutation + Wigner inverse transform operation.

This operation is the final step in the edge message passing pipeline:
1. Permute from M-major to L-major ordering
2. Apply block-diagonal Wigner inverse rotation

Public API:
- PermuteWignerInvEdgeToNodeFunction: torch.autograd.Function for the full operation
"""

from __future__ import annotations

import torch


class PermuteWignerInvEdgeToNodeFunction(torch.autograd.Function):
    """
    Autograd function for M→L permute + Wigner inverse.

    Forward: x[E,9,C] -> out[E,9,C]
    Backward: Computes grad_x, grad_wigner
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Edge features [E, 9, C] in M-major order
            wigner: Wigner inverse matrices [E, 9, 9]

        Returns:
            out: [E, 9, C] rotated features in L-major order
        """

        # Allocation VISIBLE to torch.compile (can be optimized)
        out = torch.empty_like(x)

        # ONLY kernel launch is opaque (via custom_op with mutates_args)
        torch.ops.fairchem._kernel_permute_wigner_inv_edge_to_node(x, wigner, out)

        # Save x (M-major) instead of x_l — bwd_dw kernel permutes internally
        ctx.save_for_backward(x, wigner)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass.

        Args:
            grad_out: [E, 9, C] gradient from downstream in L-major order

        Returns:
            grad_x: [E, 9, C] in M-major order
            grad_wigner: [E, 9, 9]
        """
        x_m, wigner = ctx.saved_tensors
        num_edges = grad_out.shape[0]

        # Allocation VISIBLE to torch.compile
        grad_x = torch.empty_like(grad_out)

        # ONLY kernel launch is opaque
        torch.ops.fairchem._kernel_permute_wigner_inv_edge_to_node_bwd_dx(
            grad_out, wigner, grad_x
        )

        # Allocation VISIBLE to torch.compile (zeros for off-diagonal blocks)
        grad_wigner_flat = torch.zeros(
            (num_edges, 81),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )

        # Pass x_m (M-major) — kernel handles M→L permutation internally
        torch.ops.fairchem._kernel_permute_wigner_inv_edge_to_node_bwd_dw(
            grad_out, x_m, grad_wigner_flat
        )

        grad_wigner = grad_wigner_flat.reshape(num_edges, 9, 9)

        return grad_x, grad_wigner
