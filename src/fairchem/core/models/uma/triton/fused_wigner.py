"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused wigner<->SO2-conv edgewise ops (lmax=mmax=2).

Two tightly-coupled ops that keep the M-major [E,9,2C]/[E,9,C] x_message
intermediates out of DRAM around the SO2 convolutions on the umas_fast_gpu path:

- Producer (wigner_conv1_fused_op): expands node_to_edge_wigner_permute to emit
  conv1's scaled + GEMM-packed buffers (m0, m1, m2) directly from registers.
- Consumer (wigner_inv_conv2_scatter_op): absorbs the conv2-output M->L unpack,
  inverse-Wigner rotation, and node scatter without materializing [E, 9, C].

torch.compile-safe: kernel launches are wrapped via torch.library.triton_op
(visible to inductor via wrap_triton) while tensor allocation stays in the
autograd.Function so inductor can optimize it.
The backward kernels re-derive their inputs (re-gather node features / reuse the
saved GEMM buffers) instead of stashing the large per-layer intermediates.

Public API:
- wigner_conv1_fused_op / WignerConv1FusedFunction   (producer, conv1)
- wigner_inv_conv2_fused_op / WignerInvConv2FusedFunction   (consumer, conv2 inv)
- wigner_inv_conv2_scatter_op / WignerInvConv2ScatterFunction (consumer + scatter)
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.triton.constants import FUSED_WIGNER_GRID_E_STRIDE
from fairchem.core.models.uma.triton.kernels import (
    wigner_conv1_fused_bwd_kernel,
    wigner_conv1_fused_fwd_kernel,
    wigner_inv_conv2_fused_bwd_kernel,
    wigner_inv_conv2_fused_fwd_kernel,
    wigner_inv_conv2_scatter_fwd_kernel,
)


def _compact_l2_wigner(wigner: Tensor, num_edges: int) -> Tensor:
    if wigner.ndim != 2 or wigner.shape != (num_edges, 35):
        raise ValueError("wigner must have shape [E, 35]")
    return wigner.reshape(num_edges, 35)


def _prepare_l2_wigner(wigner: Tensor, num_edges: int) -> Tensor:
    if wigner.ndim == 2 and wigner.shape == (num_edges, 35):
        return wigner
    if wigner.ndim == 3 and wigner.shape == (num_edges, 9, 9):
        return torch.cat(
            (
                wigner[:, :1, :1].flatten(1),
                wigner[:, 1:4, 1:4].flatten(1),
                wigner[:, 4:9, 4:9].flatten(1),
            ),
            dim=1,
        )
    raise ValueError("wigner must have shape [E, 35] or [E, 9, 9]")


# =============================================================================
# Producer-side fused wigner -> conv1 (emits conv1's GEMM-ready packed buffers)
# =============================================================================


@triton_op(
    "fairchem::_kernel_wigner_conv1_fused_fwd",
    mutates_args=("m0", "m1", "m2"),
)
def _kernel_wigner_conv1_fused_fwd(
    x_full: Tensor,
    edge_index: Tensor,
    wigner_flat: Tensor,
    radial: Tensor,
    m0: Tensor,
    m1: Tensor,
    m2: Tensor,
    C: int,
) -> None:
    """
    Kernel-only wrapper: launches the producer forward kernel, mutates m0/m1/m2.
    """
    E = edge_index.shape[1]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),)

    wrap_triton(wigner_conv1_fused_fwd_kernel)[grid](
        x_full,
        edge_index,
        wigner_flat,
        radial,
        m0,
        m1,
        m2,
        E,
        C,
        x_full.stride(0),
        x_full.stride(1),
        x_full.stride(2),
        edge_index.stride(0),
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_wigner_conv1_fused_bwd",
    mutates_args=("grad_out", "gwig", "grad_rad"),
)
def _kernel_wigner_conv1_fused_bwd(
    gm0: Tensor,
    gm1: Tensor,
    gm2: Tensor,
    wigner_flat: Tensor,
    radial: Tensor,
    x_full: Tensor,
    edge_index: Tensor,
    grad_out: Tensor,
    gwig: Tensor,
    grad_rad: Tensor,
    C: int,
    direct_scatter: bool,
) -> None:
    """
    Kernel-only wrapper: launches the producer backward kernel.

    Mutates grad_out/gwig/grad_rad in-place. grad_out must be zero-initialized
    when direct_scatter is enabled, as must gwig in both modes.
    """
    E = wigner_flat.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),)

    wrap_triton(wigner_conv1_fused_bwd_kernel)[grid](
        gm0,
        gm1,
        gm2,
        wigner_flat,
        radial,
        x_full,
        edge_index,
        grad_out,
        gwig,
        grad_rad,
        E,
        x_full.stride(0),
        x_full.stride(1),
        x_full.stride(2),
        edge_index.stride(0),
        C=C,
        DIRECT_SCATTER=direct_scatter,
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        num_warps=1,
    )


class WignerConv1FusedFunction(torch.autograd.Function):
    """
    Autograd function for the producer-side fused wigner->conv1 emit.

    Forward: (x_full [N,9,C], edge_index, wigner [E,35], radial) -> the three
    GEMM-ready packed buffers (m0, m1, m2).
    Backward: grads wrt node features (via the gather transpose), wigner, radial.
    """

    @staticmethod
    def forward(
        ctx,
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        radial: torch.Tensor,
        C: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_full: Node features [N, 9, C] (L-major).
            edge_index: Edge indices [2, E].
            wigner: Compact Wigner blocks [E, 35].
            radial: Per-layer conv1 radial embedding [E, 6*2C] (rad_func applied).
            C: sphere_channels.

        Returns:
            (m0, m1, m2) GEMM-ready packed buffers.
        """
        x_full = x_full.contiguous()
        radial = radial.contiguous()
        E = edge_index.shape[1]
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()
        C2 = 2 * C
        dev, dt = x_full.device, x_full.dtype

        m0 = torch.empty((E, 3 * C2), device=dev, dtype=dt)
        m1 = torch.empty((E, 4 * C2), device=dev, dtype=dt)
        m2 = torch.empty((E, 2 * C2), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_conv1_fused_fwd(
            x_full, edge_index, wigner_flat, radial, m0, m1, m2, C
        )

        ctx.save_for_backward(edge_index, wigner, radial, x_full)
        ctx.N = x_full.shape[0]
        ctx.C = C
        return m0, m1, m2

    @staticmethod
    def backward(ctx, gm0, gm1, gm2):
        """
        Backward pass.

        Args:
            gm0/gm1/gm2: Grads wrt the packed buffers.

        Returns:
            grad_x [N, 9, C], None (edge_index), grad_wigner [E, 35],
            grad_radial [E, 6*2C], None (C).
        """
        edge_index, wigner, radial, x_full = ctx.saved_tensors
        N, C = ctx.N, ctx.C
        E = edge_index.shape[1]
        C2 = 2 * C
        dev, dt = x_full.device, x_full.dtype
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()

        direct_scatter = not torch.are_deterministic_algorithms_enabled()
        if direct_scatter:
            grad_out = torch.zeros((N, 9, C), device=dev, dtype=dt)
        else:
            grad_out = torch.empty((E, 9, C2), device=dev, dtype=dt)
        gwig = torch.zeros_like(wigner_flat)
        grad_rad = torch.empty((E, 6 * C2), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_conv1_fused_bwd(
            gm0.contiguous(),
            gm1.contiguous(),
            gm2.contiguous(),
            wigner_flat,
            radial,
            x_full,
            edge_index,
            grad_out,
            gwig,
            grad_rad,
            C,
            direct_scatter,
        )

        if direct_scatter:
            grad_x = grad_out
        else:
            grad_x = torch.zeros((N, 9, C), device=dev, dtype=dt)
            grad_x.view(N, 9 * C).index_add_(
                0, edge_index[0], grad_out[:, :, :C].reshape(E, 9 * C)
            )
            grad_x.view(N, 9 * C).index_add_(
                0, edge_index[1], grad_out[:, :, C:].reshape(E, 9 * C)
            )

        return grad_x, None, gwig, grad_rad, None


def wigner_conv1_fused_op(
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    radial: torch.Tensor,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compile-safe producer-side fused emit.

    Args:
        x_full: Node features [N, 9, C] (L-major).
        edge_index: Edge indices [2, E].
        wigner: Compact Wigner blocks [E, 35] or a dense matrix [E, 9, 9].
        radial: Per-layer conv1 radial embedding [E, 6*2C] (rad_func applied).
        C: sphere_channels.

    Returns:
        (m0, m1, m2) GEMM-ready packed buffers.
    """
    wigner = _prepare_l2_wigner(wigner, edge_index.shape[1])
    return WignerConv1FusedFunction.apply(x_full, edge_index, wigner, radial, C)


# =============================================================================
# Consumer-side fused wigner-inv <- conv2 (unpack + inverse-Wigner rotation)
# =============================================================================


@triton_op(
    "fairchem::_kernel_wigner_inv_conv2_fused_fwd",
    mutates_args=("out",),
)
def _kernel_wigner_inv_conv2_fused_fwd(
    g0: Tensor,
    g1: Tensor,
    g2: Tensor,
    wigner_flat: Tensor,
    out: Tensor,
    C: int,
) -> None:
    """
    Kernel-only wrapper: launches the consumer inv forward kernel, mutates out.
    """
    E = g0.shape[0]
    num_c_blocks = (C + C - 1) // C

    def grid(_meta):
        return (
            torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),
            num_c_blocks,
        )

    wrap_triton(wigner_inv_conv2_fused_fwd_kernel)[grid](
        g0,
        g1,
        g2,
        wigner_flat,
        out,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_wigner_inv_conv2_scatter_fwd",
    mutates_args=("out",),
)
def _kernel_wigner_inv_conv2_scatter_fwd(
    g0: Tensor,
    g1: Tensor,
    g2: Tensor,
    wigner_flat: Tensor,
    scatter_target: Tensor,
    out: Tensor,
    C: int,
) -> None:
    """Launch the fused inverse-Wigner rotation and node scatter."""
    E = g0.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),)

    wrap_triton(wigner_inv_conv2_scatter_fwd_kernel)[grid](
        g0,
        g1,
        g2,
        wigner_flat,
        scatter_target,
        out,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_wigner_inv_conv2_fused_bwd",
    mutates_args=("dg0", "dg1", "dg2", "dw"),
)
def _kernel_wigner_inv_conv2_fused_bwd(
    grad_out: Tensor,
    g0: Tensor,
    g1: Tensor,
    g2: Tensor,
    wigner_flat: Tensor,
    dg0: Tensor,
    dg1: Tensor,
    dg2: Tensor,
    dw: Tensor,
    C: int,
) -> None:
    """
    Kernel-only wrapper: launches the consumer inv backward kernel.

    Mutates dg0/dg1/dg2/dw in-place. dw must be zero-initialized (only the
    block-diagonal entries are written).
    """
    E = g0.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),)

    wrap_triton(wigner_inv_conv2_fused_bwd_kernel)[grid](
        grad_out,
        grad_out,  # Ignored when GATHER_NODE_GRAD is false.
        g0,
        g1,
        g2,
        wigner_flat,
        dg0,
        dg1,
        dg2,
        dw,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        GATHER_NODE_GRAD=False,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_wigner_inv_conv2_scatter_bwd",
    mutates_args=("dg0", "dg1", "dg2", "dw"),
)
def _kernel_wigner_inv_conv2_scatter_bwd(
    grad_out: Tensor,
    scatter_target: Tensor,
    g0: Tensor,
    g1: Tensor,
    g2: Tensor,
    wigner_flat: Tensor,
    dg0: Tensor,
    dg1: Tensor,
    dg2: Tensor,
    dw: Tensor,
    C: int,
) -> None:
    E = g0.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(E, FUSED_WIGNER_GRID_E_STRIDE)),)

    wrap_triton(wigner_inv_conv2_fused_bwd_kernel)[grid](
        grad_out,
        scatter_target,
        g0,
        g1,
        g2,
        wigner_flat,
        dg0,
        dg1,
        dg2,
        dw,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=FUSED_WIGNER_GRID_E_STRIDE,
        GATHER_NODE_GRAD=True,
        num_warps=1,
    )


class WignerInvConv2FusedFunction(torch.autograd.Function):
    """
    Autograd function for the consumer-side fused inv-wigner <- conv2 emit.

    Forward: (g0 [E,3C], g1 [E,4C], g2 [E,2C], wigner [E,35]) -> x_rotated
    [E, 9, C] (L-major).
    Backward: grads wrt the three GEMM buffers and wigner.
    """

    @staticmethod
    def forward(
        ctx,
        g0: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        wigner: torch.Tensor,
        C: int,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            g0: conv2 fc_m0 output [E, 3C] (rows M0,M1,M2).
            g1: conv2 m=1 block-GEMM output [E, 4C] (rows M3,M4,M5,M6).
            g2: conv2 m=2 block-GEMM output [E, 2C] (rows M7,M8).
            wigner: Compact inverse Wigner blocks [E, 35].
            C: sphere_channels.

        Returns:
            x_rotated [E, 9, C] (L-major).
        """
        g0 = g0.contiguous()
        g1 = g1.contiguous()
        g2 = g2.contiguous()
        E = g0.shape[0]
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()
        dev, dt = g0.device, g0.dtype

        out = torch.empty((E, 9, C), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_inv_conv2_fused_fwd(
            g0, g1, g2, wigner_flat, out, C
        )

        ctx.save_for_backward(g0, g1, g2, wigner)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.

        Args:
            grad_out: Grad wrt x_rotated [E, 9, C] (L-major).

        Returns:
            dg0 [E, 3C], dg1 [E, 4C], dg2 [E, 2C], dw [E, 35], None (C).
        """
        g0, g1, g2, wigner = ctx.saved_tensors
        C = ctx.C
        E = g0.shape[0]
        dev, dt = g0.device, g0.dtype
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()

        dg0 = torch.empty((E, 3 * C), device=dev, dtype=dt)
        dg1 = torch.empty((E, 4 * C), device=dev, dtype=dt)
        dg2 = torch.empty((E, 2 * C), device=dev, dtype=dt)
        dw = torch.zeros_like(wigner_flat)

        torch.ops.fairchem._kernel_wigner_inv_conv2_fused_bwd(
            grad_out.contiguous(), g0, g1, g2, wigner_flat, dg0, dg1, dg2, dw, C
        )
        return dg0, dg1, dg2, dw, None


def wigner_inv_conv2_fused_op(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    wigner: torch.Tensor,
    C: int,
) -> torch.Tensor:
    """
    Compile-safe consumer-side fused inv emit.

    Args:
        g0: conv2 fc_m0 output [E, 3C] (rows M0,M1,M2).
        g1: conv2 m=1 block-GEMM output [E, 4C] (rows M3,M4,M5,M6).
        g2: conv2 m=2 block-GEMM output [E, 2C] (rows M7,M8).
        wigner: Compact inverse Wigner blocks [E, 35] or a dense matrix [E, 9, 9].
        C: sphere_channels.

    Returns:
        x_rotated [E, 9, C] (L-major).
    """
    wigner = _prepare_l2_wigner(wigner, g0.shape[0])
    return WignerInvConv2FusedFunction.apply(g0, g1, g2, wigner, C)


class WignerInvConv2ScatterFunction(torch.autograd.Function):
    """Inverse-Wigner rotation with direct target-node accumulation."""

    @staticmethod
    def forward(
        ctx,
        g0: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        wigner: torch.Tensor,
        scatter_target: torch.Tensor,
        num_nodes: int,
        C: int,
    ) -> torch.Tensor:
        g0 = g0.contiguous()
        g1 = g1.contiguous()
        g2 = g2.contiguous()
        E = g0.shape[0]
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()
        out = torch.zeros((num_nodes, 9, C), device=g0.device, dtype=g0.dtype)

        if torch.are_deterministic_algorithms_enabled():
            edge_out = torch.empty((E, 9, C), device=g0.device, dtype=g0.dtype)
            torch.ops.fairchem._kernel_wigner_inv_conv2_fused_fwd(
                g0, g1, g2, wigner_flat, edge_out, C
            )
            out.index_add_(0, scatter_target, edge_out)
        else:
            torch.ops.fairchem._kernel_wigner_inv_conv2_scatter_fwd(
                g0, g1, g2, wigner_flat, scatter_target, out, C
            )

        ctx.save_for_backward(scatter_target, g0, g1, g2, wigner)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, grad_out):
        scatter_target, g0, g1, g2, wigner = ctx.saved_tensors
        C = ctx.C
        E = g0.shape[0]
        dev, dt = g0.device, g0.dtype
        wigner_flat = _compact_l2_wigner(wigner, E).contiguous()
        dg0 = torch.empty((E, 3 * C), device=dev, dtype=dt)
        dg1 = torch.empty((E, 4 * C), device=dev, dtype=dt)
        dg2 = torch.empty((E, 2 * C), device=dev, dtype=dt)
        dw = torch.zeros_like(wigner_flat)
        torch.ops.fairchem._kernel_wigner_inv_conv2_scatter_bwd(
            grad_out.contiguous(),
            scatter_target,
            g0,
            g1,
            g2,
            wigner_flat,
            dg0,
            dg1,
            dg2,
            dw,
            C,
        )
        return dg0, dg1, dg2, dw, None, None, None


def wigner_inv_conv2_scatter_op(
    g0: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    wigner: torch.Tensor,
    scatter_target: torch.Tensor,
    num_nodes: int,
    C: int,
) -> torch.Tensor:
    """
    Rotate packed conv2 outputs and accumulate them into target nodes.

    Uses direct atomic accumulation by default and preserves PyTorch's
    deterministic-algorithm behavior through a materialized fallback.
    """
    wigner = _prepare_l2_wigner(wigner, g0.shape[0])
    return WignerInvConv2ScatterFunction.apply(
        g0, g1, g2, wigner, scatter_target, num_nodes, C
    )
