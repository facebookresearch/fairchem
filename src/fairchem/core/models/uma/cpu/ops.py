"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CPU-optimized autograd functions for UMA inference.

Uses JIT-compiled C++ kernels for fused gather+Wigner+permute operations.
Falls back to pure PyTorch if C++ compilation fails.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

# JIT-compile C++ kernels
_cpp_kernels = None


def _get_cpp_kernels():
    global _cpp_kernels
    if _cpp_kernels is not None:
        return _cpp_kernels

    try:
        from torch.utils.cpp_extension import load

        kernel_src = os.path.join(os.path.dirname(__file__), "kernels.cpp")
        _cpp_kernels = load(
            name="uma_cpu_kernels",
            sources=[kernel_src],
            extra_cflags=[
                "-O3",
                "-fopenmp",
                "-march=native",
                "-funroll-loops",
                "-ffast-math",
            ],
            extra_ldflags=["-lgomp"],
            verbose=False,
        )
        log.info("C++ CPU kernels compiled successfully")
    except Exception as e:
        log.warning("Failed to compile C++ kernels, using PyTorch fallback: %s", e)
        _cpp_kernels = False  # sentinel to avoid retrying

    return _cpp_kernels


class CPUNodeToEdgeWignerPermuteFunction(torch.autograd.Function):
    """
    CPU autograd function for node-to-edge gather + Wigner + L->M permute.

    Forward: x[N,9,C] -> out[E,9,2C]
    Uses C++ fused kernel when available, PyTorch fallback otherwise.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        kernels = _get_cpp_kernels()

        if kernels:
            ei_c = edge_index.to(torch.int64)

            out = kernels.node_to_edge_wigner_permute_fwd(x, ei_c, wigner)

            ctx.save_for_backward(x, ei_c, wigner)
            ctx.num_nodes = x.shape[0]
            ctx.use_cpp = True
            return out
        else:
            return _pytorch_node_to_edge_fwd(ctx, x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.use_cpp:
            x, edge_index, wigner = ctx.saved_tensors
            kernels = _get_cpp_kernels()

            grad_out = grad_out.contiguous()
            grad_x = kernels.node_to_edge_wigner_permute_bwd_dx(
                grad_out, wigner, edge_index, ctx.num_nodes
            )
            grad_wigner = kernels.node_to_edge_wigner_permute_bwd_dw(
                grad_out, x, edge_index
            )
            return grad_x, None, grad_wigner
        else:
            return _pytorch_node_to_edge_bwd(ctx, grad_out)


class CPUPermuteWignerInvEdgeToNodeFunction(torch.autograd.Function):
    """
    CPU autograd function for M->L permute + Wigner inverse rotation.

    Forward: x[E,9,C] (M-major) -> out[E,9,C] (L-major)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        kernels = _get_cpp_kernels()

        if kernels:
            out = kernels.permute_wigner_inv_fwd(x, wigner_inv)

            ctx.save_for_backward(x, wigner_inv)
            ctx.use_cpp = True
            return out
        else:
            return _pytorch_permute_wigner_inv_fwd(ctx, x, wigner_inv)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.use_cpp:
            x_m, wigner_inv = ctx.saved_tensors
            kernels = _get_cpp_kernels()

            grad_out = grad_out.contiguous()
            grad_x = kernels.permute_wigner_inv_bwd_dx(grad_out, wigner_inv)
            grad_wigner = kernels.permute_wigner_inv_bwd_dw(grad_out, x_m)

            return grad_x, grad_wigner
        else:
            return _pytorch_permute_wigner_inv_bwd(ctx, grad_out)


# ============================================================================
# PyTorch fallback implementations
# ============================================================================

L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]


def _pytorch_node_to_edge_fwd(ctx, x, edge_index, wigner):
    E = edge_index.shape[1]

    x_source = x[edge_index[0]]
    x_target = x[edge_index[1]]
    x_cat = torch.cat((x_source, x_target), dim=2)

    W = wigner.reshape(E, 9, 9)
    rotated = torch.empty_like(x_cat)
    rotated[:, 0:1, :] = W[:, 0:1, 0:1] * x_cat[:, 0:1, :]
    rotated[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], x_cat[:, 1:4, :])
    rotated[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], x_cat[:, 4:9, :])

    out = rotated[:, L_TO_M_GATHER_IDX, :]

    ctx.save_for_backward(x, edge_index, wigner)
    ctx.num_nodes = x.shape[0]
    ctx.use_cpp = False
    return out


def _pytorch_node_to_edge_bwd(ctx, grad_out):
    x, edge_index, wigner = ctx.saved_tensors
    E = edge_index.shape[1]
    C = x.shape[2]

    grad_l = grad_out[:, M_TO_L_GATHER_IDX, :]
    W = wigner.reshape(E, 9, 9)

    grad_cat = torch.empty_like(grad_l)
    grad_cat[:, 0:1, :] = W[:, 0:1, 0:1] * grad_l[:, 0:1, :]
    grad_cat[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4].transpose(1, 2), grad_l[:, 1:4, :])
    grad_cat[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9].transpose(1, 2), grad_l[:, 4:9, :])

    grad_src = grad_cat[:, :, :C]
    grad_tgt = grad_cat[:, :, C:]

    grad_x = torch.zeros(
        (ctx.num_nodes, 9, C),
        dtype=grad_out.dtype,
        device=grad_out.device,
    )
    grad_x.index_add_(0, edge_index[0], grad_src)
    grad_x.index_add_(0, edge_index[1], grad_tgt)

    x_source = x[edge_index[0]]
    x_target = x[edge_index[1]]
    x_cat = torch.cat((x_source, x_target), dim=2)

    grad_wigner = torch.zeros(E, 9, 9, dtype=wigner.dtype, device=wigner.device)
    grad_wigner[:, 0:1, 0:1] = torch.bmm(
        grad_l[:, 0:1, :], x_cat[:, 0:1, :].transpose(1, 2)
    )
    grad_wigner[:, 1:4, 1:4] = torch.bmm(
        grad_l[:, 1:4, :], x_cat[:, 1:4, :].transpose(1, 2)
    )
    grad_wigner[:, 4:9, 4:9] = torch.bmm(
        grad_l[:, 4:9, :], x_cat[:, 4:9, :].transpose(1, 2)
    )

    return grad_x, None, grad_wigner


def _pytorch_permute_wigner_inv_fwd(ctx, x, wigner_inv):
    E = x.shape[0]
    x_l = x[:, M_TO_L_GATHER_IDX, :]
    W = wigner_inv.reshape(E, 9, 9)

    out = torch.empty_like(x_l)
    out[:, 0:1, :] = W[:, 0:1, 0:1] * x_l[:, 0:1, :]
    out[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], x_l[:, 1:4, :])
    out[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], x_l[:, 4:9, :])

    ctx.save_for_backward(x, wigner_inv)
    ctx.use_cpp = False
    return out


def _pytorch_permute_wigner_inv_bwd(ctx, grad_out):
    x_m, wigner_inv = ctx.saved_tensors
    E = grad_out.shape[0]
    W = wigner_inv.reshape(E, 9, 9)

    grad_x_l = torch.empty_like(grad_out)
    grad_x_l[:, 0:1, :] = W[:, 0:1, 0:1] * grad_out[:, 0:1, :]
    grad_x_l[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4].transpose(1, 2), grad_out[:, 1:4, :])
    grad_x_l[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9].transpose(1, 2), grad_out[:, 4:9, :])

    grad_x = grad_x_l[:, L_TO_M_GATHER_IDX, :]

    x_l = x_m[:, M_TO_L_GATHER_IDX, :]
    grad_wigner = torch.zeros(E, 9, 9, dtype=wigner_inv.dtype, device=wigner_inv.device)
    grad_wigner[:, 0:1, 0:1] = torch.bmm(
        grad_out[:, 0:1, :], x_l[:, 0:1, :].transpose(1, 2)
    )
    grad_wigner[:, 1:4, 1:4] = torch.bmm(
        grad_out[:, 1:4, :], x_l[:, 1:4, :].transpose(1, 2)
    )
    grad_wigner[:, 4:9, 4:9] = torch.bmm(
        grad_out[:, 4:9, :], x_l[:, 4:9, :].transpose(1, 2)
    )

    return grad_x, grad_wigner
