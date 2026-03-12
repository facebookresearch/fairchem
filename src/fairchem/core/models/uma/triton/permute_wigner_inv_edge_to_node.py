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

from fairchem.core.models.uma.triton.constants import BLOCK_C
from fairchem.core.models.uma.triton.kernels import (
    permute_wigner_inv_edge_to_node_bwd_dw_kernel,
    permute_wigner_inv_edge_to_node_bwd_dx_kernel,
    permute_wigner_inv_edge_to_node_kernel,
)


def permute_wigner_inv_edge_to_node_launcher(
    x: torch.Tensor,
    wigner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward launcher: M→L permute + Wigner inverse.

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

    permute_wigner_inv_edge_to_node_kernel[(E, num_c_blocks)](
        x,
        wigner,
        out,
        x_l,
        E,
        C,
        BLOCK_C=BLOCK_C,
        SAVE_XL=True,
    )
    return out, x_l


def permute_wigner_inv_edge_to_node_bwd_dx_launcher(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    Backward launcher w.r.t. input x: W^T @ dy + L→M permute.

    Args:
        grad_out: Gradient from downstream [E, 9, C] in L-major order
        wigner: Wigner inverse matrices [E, 9, 9]

    Returns:
        grad_x: Gradient w.r.t. input [E, 9, C] in M-major order
    """
    # grad_out: [E, 9, C] - gradient with 9 coefficients (lmax=2)
    assert grad_out.ndim == 3, "grad_out must be 3D [E, 9, C]"
    assert grad_out.shape[1] == 9, "grad_out must have 9 coefficients (lmax=2)"
    # wigner: [E, 9, 9] - block-diagonal Wigner inverse matrices
    assert wigner.ndim == 3, "wigner must be 3D [E, 9, 9]"
    assert wigner.shape[1] == 9, "wigner must have shape [E, 9, 9]"
    assert wigner.shape[2] == 9, "wigner must have shape [E, 9, 9]"
    # Wigner must be contiguous for memory access pattern
    assert wigner.is_contiguous(), "wigner must be contiguous"

    grad_out = grad_out.contiguous()
    E, num_coeffs, C = grad_out.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    grad_x = torch.empty_like(grad_out)
    permute_wigner_inv_edge_to_node_bwd_dx_kernel[(E, num_c_blocks)](
        grad_out,
        wigner,
        grad_x,
        E,
        C,
        BLOCK_C=BLOCK_C,
    )
    return grad_x


def permute_wigner_inv_edge_to_node_bwd_dw_launcher(
    grad_out: torch.Tensor,
    x_l: torch.Tensor,
) -> torch.Tensor:
    """
    Backward launcher w.r.t. Wigner: dW = dy @ x_l^T.

    Args:
        grad_out: Gradient from downstream [E, 9, C] in L-major order
        x_l: Saved permuted input [E, 9, C] in L-major order

    Returns:
        grad_wigner: Gradient w.r.t. Wigner [E, 9, 9]
    """
    # grad_out: [E, 9, C] - gradient with 9 coefficients (lmax=2)
    assert grad_out.ndim == 3, "grad_out must be 3D [E, 9, C]"
    assert grad_out.shape[1] == 9, "grad_out must have 9 coefficients (lmax=2)"
    # x_l: [E, 9, C] - saved permuted input
    assert x_l.ndim == 3, "x_l must be 3D [E, 9, C]"
    assert x_l.shape[1] == 9, "x_l must have 9 coefficients (lmax=2)"
    # Shapes must match (kernel accesses both identically)
    assert grad_out.shape == x_l.shape, "grad_out and x_l must have same shape"
    # C must be <= 128 (kernel loads all channels in single pass via tl.arange(0, 128))
    assert grad_out.shape[2] <= 128, "sphere_channels must be <= 128"
    # Contiguity required for memory access pattern
    assert grad_out.is_contiguous(), "grad_out must be contiguous"
    assert x_l.is_contiguous(), "x_l must be contiguous"

    num_edges = grad_out.shape[0]
    sphere_channels = grad_out.shape[2]

    # Allocate output (full 9x9 per edge, zeros in off-diagonal blocks)
    grad_wigner = torch.zeros(
        (num_edges, 81),
        dtype=grad_out.dtype,
        device=grad_out.device,
    )

    # Grid: one thread block per edge
    grid = (num_edges,)

    permute_wigner_inv_edge_to_node_bwd_dw_kernel[grid](
        grad_out,
        x_l,
        grad_wigner,
        num_edges,
        sphere_channels,
    )

    return grad_wigner.reshape(num_edges, 9, 9)


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
        out, x_l = permute_wigner_inv_edge_to_node_launcher(x, wigner)
        ctx.save_for_backward(x_l, wigner)
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
        x_l, wigner = ctx.saved_tensors
        grad_x = permute_wigner_inv_edge_to_node_bwd_dx_launcher(grad_out, wigner)
        grad_wigner = permute_wigner_inv_edge_to_node_bwd_dw_launcher(grad_out, x_l)
        return grad_x, grad_wigner
