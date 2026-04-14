"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Register CPU Wigner kernels as torch custom ops for torch.compile
compatibility.

Using torch.library.custom_op makes the kernels visible to
torch.compile/dynamo, avoiding graph breaks. This is the CPU
equivalent of triton_op used for GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import custom_op


def _get_kernels():
    """
    Lazy import to avoid circular dependency.
    """
    from fairchem.core.models.uma.cpu.ops import _get_cpp_kernels

    return _get_cpp_kernels()


@custom_op("fairchem_cpu::node_to_edge_wigner_permute_fwd", mutates_args=())
def _n2e_fwd(x: Tensor, edge_index: Tensor, wigner: Tensor) -> Tensor:
    kernels = _get_kernels()
    return kernels.node_to_edge_wigner_permute_fwd(
        x, edge_index.to(torch.int64), wigner
    )


@_n2e_fwd.register_fake
def _n2e_fwd_fake(x: Tensor, edge_index: Tensor, wigner: Tensor) -> Tensor:
    E = edge_index.shape[1]
    C = x.shape[2]
    return torch.empty(E, 9, C * 2, dtype=x.dtype, device=x.device)


@custom_op("fairchem_cpu::node_to_edge_wigner_permute_bwd_dx", mutates_args=())
def _n2e_bwd_dx(
    grad_out: Tensor,
    wigner: Tensor,
    edge_index: Tensor,
    num_nodes: int,
) -> Tensor:
    kernels = _get_kernels()
    return kernels.node_to_edge_wigner_permute_bwd_dx(
        grad_out, wigner, edge_index, num_nodes
    )


@_n2e_bwd_dx.register_fake
def _n2e_bwd_dx_fake(
    grad_out: Tensor,
    wigner: Tensor,
    edge_index: Tensor,
    num_nodes: int,
) -> Tensor:
    C = grad_out.shape[2] // 2
    return torch.empty(num_nodes, 9, C, dtype=grad_out.dtype, device=grad_out.device)


@custom_op("fairchem_cpu::node_to_edge_wigner_permute_bwd_dw", mutates_args=())
def _n2e_bwd_dw(grad_out: Tensor, x: Tensor, edge_index: Tensor) -> Tensor:
    kernels = _get_kernels()
    return kernels.node_to_edge_wigner_permute_bwd_dw(grad_out, x, edge_index)


@_n2e_bwd_dw.register_fake
def _n2e_bwd_dw_fake(grad_out: Tensor, x: Tensor, edge_index: Tensor) -> Tensor:
    E = edge_index.shape[1]
    return torch.empty(E, 9, 9, dtype=grad_out.dtype, device=grad_out.device)


@custom_op("fairchem_cpu::permute_wigner_inv_fwd", mutates_args=())
def _pwi_fwd(x: Tensor, wigner_inv: Tensor) -> Tensor:
    kernels = _get_kernels()
    return kernels.permute_wigner_inv_fwd(x, wigner_inv)


@_pwi_fwd.register_fake
def _pwi_fwd_fake(x: Tensor, wigner_inv: Tensor) -> Tensor:
    return torch.empty_like(x)


@custom_op("fairchem_cpu::permute_wigner_inv_bwd_dx", mutates_args=())
def _pwi_bwd_dx(grad_out: Tensor, wigner_inv: Tensor) -> Tensor:
    kernels = _get_kernels()
    return kernels.permute_wigner_inv_bwd_dx(grad_out, wigner_inv)


@_pwi_bwd_dx.register_fake
def _pwi_bwd_dx_fake(grad_out: Tensor, wigner_inv: Tensor) -> Tensor:
    return torch.empty_like(grad_out)


@custom_op("fairchem_cpu::permute_wigner_inv_bwd_dw", mutates_args=())
def _pwi_bwd_dw(grad_out: Tensor, x_m: Tensor) -> Tensor:
    kernels = _get_kernels()
    return kernels.permute_wigner_inv_bwd_dw(grad_out, x_m)


@_pwi_bwd_dw.register_fake
def _pwi_bwd_dw_fake(grad_out: Tensor, x_m: Tensor) -> Tensor:
    E = grad_out.shape[0]
    return torch.empty(E, 9, 9, dtype=grad_out.dtype, device=grad_out.device)
