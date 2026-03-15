"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Triton kernel wrappers using torch.library.triton_op.

These wrappers make kernels visible to torch.compile while keeping
tensor allocation visible for optimization. Using mutates_args
allows the kernel to write to pre-allocated outputs.

Why this design:
- torch.compile can't trace raw Triton kernels (data_ptr access)
- triton_op + wrap_triton makes kernels visible to the compiler
- Tensor allocation (torch.empty) CAN be traced and optimized
- By wrapping only the kernel, compile can optimize memory layout

Public API:
- _kernel_node_to_edge_wigner_permute: Forward kernel wrapper
- _kernel_permute_wigner_inv_edge_to_node: Forward kernel wrapper
- _kernel_node_to_edge_wigner_permute_bwd_dx: Backward kernel wrapper
- _kernel_permute_wigner_inv_edge_to_node_bwd_dx: Backward kernel wrapper
- _kernel_permute_wigner_inv_edge_to_node_bwd_dw: Backward kernel wrapper
"""

from __future__ import annotations

from torch import (
    Tensor,  # noqa: TCH002 - needed at runtime for triton_op schema inference
)
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.triton.constants import BLOCK_C, GRID_E_STRIDE
from fairchem.core.models.uma.triton.kernels import (
    node_to_edge_wigner_permute_bwd_dx_kernel,
    node_to_edge_wigner_permute_kernel,
    permute_wigner_inv_edge_to_node_bwd_dw_kernel,
    permute_wigner_inv_edge_to_node_bwd_dx_kernel,
    permute_wigner_inv_edge_to_node_kernel,
)

# =============================================================================
# Forward kernel wrapper for node_to_edge_wigner_permute
# =============================================================================


@triton_op(
    "fairchem::_kernel_node_to_edge_wigner_permute",
    mutates_args=("out", "x_edge"),
)
def _kernel_node_to_edge_wigner_permute(
    x: Tensor,
    edge_index: Tensor,
    wigner: Tensor,
    out: Tensor,
    x_edge: Tensor,
) -> None:
    """
    Kernel-only wrapper: launches Triton kernel, mutates out/x_edge in-place.

    This is opaque to torch.compile but allocation happens outside.
    """
    num_edges = edge_index.shape[1]
    sphere_channels = x.shape[2]

    # Flatten wigner for kernel
    wigner_flat = wigner.reshape(num_edges, -1)

    # Grid: (num_edges, channel_blocks)
    num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C
    grid = (GRID_E_STRIDE, num_c_blocks)

    wrap_triton(node_to_edge_wigner_permute_kernel)[grid](
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
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


# =============================================================================
# Forward kernel wrapper for permute_wigner_inv_edge_to_node
# =============================================================================


@triton_op(
    "fairchem::_kernel_permute_wigner_inv_edge_to_node",
    mutates_args=("out", "x_l"),
)
def _kernel_permute_wigner_inv_edge_to_node(
    x: Tensor,
    wigner: Tensor,
    out: Tensor,
    x_l: Tensor,
) -> None:
    """
    Kernel-only wrapper: launches Triton kernel, mutates out/x_l in-place.

    This is opaque to torch.compile but allocation happens outside.
    """
    E, num_coeffs, C = x.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

    wrap_triton(permute_wigner_inv_edge_to_node_kernel)[(GRID_E_STRIDE, num_c_blocks)](
        x,
        wigner,
        out,
        x_l,
        E,
        C,
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


# =============================================================================
# Backward kernel wrapper for node_to_edge_wigner_permute (bwd_dx)
# =============================================================================


@triton_op(
    "fairchem::_kernel_node_to_edge_wigner_permute_bwd_dx",
    mutates_args=("grad_edge",),
)
def _kernel_node_to_edge_wigner_permute_bwd_dx(
    grad_out: Tensor,
    wigner: Tensor,
    grad_edge: Tensor,
) -> None:
    """
    Backward kernel for dx: M→L permute + W^T @ grad.

    Writes per-edge gradients to grad_edge. Caller does scatter to nodes.
    """
    num_edges = grad_out.shape[0]
    sphere_channels = grad_out.shape[2] // 2

    assert (sphere_channels & (sphere_channels - 1)) == 0
    assert sphere_channels >= 1

    # Flatten wigner for kernel (wigner already contiguous from escn_md source)
    wigner_flat = wigner.reshape(num_edges, -1)

    grid = (GRID_E_STRIDE,)

    wrap_triton(node_to_edge_wigner_permute_bwd_dx_kernel)[grid](
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
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


# =============================================================================
# Backward kernel wrappers for permute_wigner_inv_edge_to_node
# =============================================================================


@triton_op(
    "fairchem::_kernel_permute_wigner_inv_edge_to_node_bwd_dx",
    mutates_args=("grad_x",),
)
def _kernel_permute_wigner_inv_edge_to_node_bwd_dx(
    grad_out: Tensor,
    wigner: Tensor,
    grad_x: Tensor,
) -> None:
    """
    Backward kernel for dx: W^T @ grad + L→M permute.

    Writes to grad_x in-place.
    """
    E, num_coeffs, C = grad_out.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

    wrap_triton(permute_wigner_inv_edge_to_node_bwd_dx_kernel)[
        (GRID_E_STRIDE, num_c_blocks)
    ](
        grad_out,
        wigner,
        grad_x,
        E,
        C,
        BLOCK_C=BLOCK_C,
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_permute_wigner_inv_edge_to_node_bwd_dw",
    mutates_args=("grad_wigner_flat",),
)
def _kernel_permute_wigner_inv_edge_to_node_bwd_dw(
    grad_out: Tensor,
    x_l: Tensor,
    grad_wigner_flat: Tensor,
) -> None:
    """
    Backward kernel for dW: dy @ x_l^T.

    Writes to grad_wigner_flat [E, 81] in-place.
    """
    num_edges = grad_out.shape[0]
    sphere_channels = grad_out.shape[2]

    assert sphere_channels & (sphere_channels - 1) == 0
    assert sphere_channels >= 1

    grid = (GRID_E_STRIDE,)

    wrap_triton(permute_wigner_inv_edge_to_node_bwd_dw_kernel)[grid](
        grad_out,
        x_l,
        grad_wigner_flat,
        num_edges,
        sphere_channels,
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )
