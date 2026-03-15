"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CPU-optimized autograd functions for UMA inference.

These mirror the Triton GPU autograd wrappers but use PyTorch CPU ops
(with the same Wigner passthrough / L↔M permutation logic as the GPU backend).
"""

from __future__ import annotations

import torch

# L-major to M-major permutation for lmax=2
# L-major: [l0, l1m-1, l1m0, l1m1, l2m-2, l2m-1, l2m0, l2m1, l2m2]
# M-major: reordered by m value
L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]


def _block_diag_bmm_forward(wigner_flat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Block-diagonal Wigner rotation: [E, 81] x [E, 9, C] -> [E, 9, C].

    Exploits lmax=2 sparsity: 1x1 + 3x3 + 5x5 blocks.
    Input x is in L-major order, output is in L-major order.
    """
    E, _, C = x.shape
    W = wigner_flat.reshape(E, 9, 9)

    out = torch.empty_like(x)

    # L=0 block (1x1): just scale
    out[:, 0:1, :] = W[:, 0:1, 0:1] * x[:, 0:1, :]

    # L=1 block (3x3): bmm on slice
    out[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], x[:, 1:4, :])

    # L=2 block (5x5): bmm on slice
    out[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], x[:, 4:9, :])

    return out


class CPUNodeToEdgeWignerPermuteFunction(torch.autograd.Function):
    """
    CPU autograd function for node-to-edge gather + Wigner + L→M permute.

    Forward: x[N,9,C] -> out[E,9,2C]
    Uses L-ordered Wigner matrices directly (passthrough, same as GPU backend).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        E = edge_index.shape[1]
        C = x.shape[2]

        # Gather source and target node features
        x_source = x[edge_index[0]]  # [E, 9, C]
        x_target = x[edge_index[1]]  # [E, 9, C]

        # Concatenate: [E, 9, 2C]
        x_cat = torch.cat((x_source, x_target), dim=2)

        # Block-diagonal Wigner rotation in L-major order
        wigner_flat = wigner.reshape(E, -1)
        rotated = _block_diag_bmm_forward(wigner_flat, x_cat)

        # L→M permutation
        out = rotated[:, L_TO_M_GATHER_IDX, :]

        ctx.save_for_backward(x, edge_index, wigner)
        ctx.num_nodes = x.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, edge_index, wigner = ctx.saved_tensors
        E = edge_index.shape[1]
        C = x.shape[2]

        # M→L permutation on grad
        grad_l = grad_out[:, M_TO_L_GATHER_IDX, :]

        # Inverse Wigner rotation (W^T for block-diagonal)
        W = wigner.reshape(E, 9, 9)
        grad_cat = torch.empty_like(grad_l)
        # L=0
        grad_cat[:, 0:1, :] = W[:, 0:1, 0:1] * grad_l[:, 0:1, :]
        # L=1: W^T @ grad
        grad_cat[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4].transpose(1, 2), grad_l[:, 1:4, :])
        # L=2: W^T @ grad
        grad_cat[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9].transpose(1, 2), grad_l[:, 4:9, :])

        # Split src/tgt gradients
        grad_src = grad_cat[:, :, :C]
        grad_tgt = grad_cat[:, :, C:]

        # Scatter back to nodes
        grad_x = torch.zeros(
            (ctx.num_nodes, 9, C),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )
        grad_x.index_add_(0, edge_index[0], grad_src)
        grad_x.index_add_(0, edge_index[1], grad_tgt)

        # Wigner gradient: dW = grad_l @ x_cat^T (block-diagonal only)
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_cat = torch.cat((x_source, x_target), dim=2)

        grad_wigner = torch.zeros(E, 9, 9, dtype=wigner.dtype, device=wigner.device)
        # L=0
        grad_wigner[:, 0:1, 0:1] = torch.bmm(grad_l[:, 0:1, :], x_cat[:, 0:1, :].transpose(1, 2))
        # L=1
        grad_wigner[:, 1:4, 1:4] = torch.bmm(grad_l[:, 1:4, :], x_cat[:, 1:4, :].transpose(1, 2))
        # L=2
        grad_wigner[:, 4:9, 4:9] = torch.bmm(grad_l[:, 4:9, :], x_cat[:, 4:9, :].transpose(1, 2))

        return grad_x, None, grad_wigner


class CPUPermuteWignerInvEdgeToNodeFunction(torch.autograd.Function):
    """
    CPU autograd function for M→L permute + Wigner inverse rotation.

    Forward: x[E,9,C] (M-major) -> out[E,9,C] (L-major)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        E = x.shape[0]

        # M→L permutation
        x_l = x[:, M_TO_L_GATHER_IDX, :]

        # Block-diagonal Wigner inverse rotation
        wigner_flat = wigner_inv.reshape(E, -1)
        out = _block_diag_bmm_forward(wigner_flat, x_l)

        ctx.save_for_backward(x, wigner_inv)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_m, wigner_inv = ctx.saved_tensors
        E = grad_out.shape[0]

        W = wigner_inv.reshape(E, 9, 9)

        # grad_x_l = W^T @ grad_out
        grad_x_l = torch.empty_like(grad_out)
        grad_x_l[:, 0:1, :] = W[:, 0:1, 0:1] * grad_out[:, 0:1, :]
        grad_x_l[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4].transpose(1, 2), grad_out[:, 1:4, :])
        grad_x_l[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9].transpose(1, 2), grad_out[:, 4:9, :])

        # L→M permutation for grad_x
        grad_x = grad_x_l[:, L_TO_M_GATHER_IDX, :]

        # Wigner gradient: dW = grad_out @ x_l^T
        x_l = x_m[:, M_TO_L_GATHER_IDX, :]
        grad_wigner = torch.zeros(E, 9, 9, dtype=wigner_inv.dtype, device=wigner_inv.device)
        grad_wigner[:, 0:1, 0:1] = torch.bmm(grad_out[:, 0:1, :], x_l[:, 0:1, :].transpose(1, 2))
        grad_wigner[:, 1:4, 1:4] = torch.bmm(grad_out[:, 1:4, :], x_l[:, 1:4, :].transpose(1, 2))
        grad_wigner[:, 4:9, 4:9] = torch.bmm(grad_out[:, 4:9, :], x_l[:, 4:9, :].transpose(1, 2))

        return grad_x, grad_wigner
