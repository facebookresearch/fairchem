"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused wigner<->SO2-conv edgewise ops (lmax=mmax=2).

Two tightly-coupled ops that keep the M-major [E,9,2C]/[E,9,C] x_message
intermediates out of DRAM around the SO2 convolutions on the umas_fast_gpu path:

- Producer (wigner_conv1_fused_op): expands node_to_edge_wigner_permute to emit
  conv1's scaled + GEMM-packed buffers (m0, m1, m2) directly from registers.
- Consumer (wigner_inv_conv2_fused_op): absorbs the conv2-output M->L unpack and
  the inverse-Wigner rotation, emitting x_rotated [E, 9, C]; the node scatter
  (index_add) stays OUTSIDE the op (visible to torch.compile).

torch.compile-safe: kernel launches are wrapped via torch.library.triton_op
(visible to inductor via wrap_triton) while tensor allocation and the
node-feature scatter stay in the autograd.Function so inductor can optimize them.
Both backwards re-derive their inputs (re-gather node features / reuse the saved
GEMM buffers) instead of stashing the large per-layer intermediates.

Public API:
- wigner_conv1_fused_op / WignerConv1FusedFunction   (producer, conv1)
- wigner_inv_conv2_fused_op / WignerInvConv2FusedFunction   (consumer, conv2 inv)
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.triton.constants import GRID_E_STRIDE
from fairchem.core.models.uma.triton.kernels import (
    wigner_conv1_fused_bwd_kernel,
    wigner_conv1_fused_fwd_kernel,
    wigner_inv_conv2_fused_bwd_kernel,
    wigner_inv_conv2_fused_fwd_kernel,
)

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
    wrap_triton(wigner_conv1_fused_fwd_kernel)[(GRID_E_STRIDE,)](
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
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_kernel_wigner_conv1_fused_bwd",
    mutates_args=("grad_edge", "gwig", "grad_rad"),
)
def _kernel_wigner_conv1_fused_bwd(
    gm0: Tensor,
    gm1: Tensor,
    gm2: Tensor,
    wigner_flat: Tensor,
    radial: Tensor,
    x_full: Tensor,
    edge_index: Tensor,
    grad_edge: Tensor,
    gwig: Tensor,
    grad_rad: Tensor,
    C: int,
) -> None:
    """
    Kernel-only wrapper: launches the producer backward kernel.

    Mutates grad_edge/gwig/grad_rad in-place. gwig must be zero-initialized (only
    the block-diagonal entries are written).
    """
    E = wigner_flat.shape[0]
    wrap_triton(wigner_conv1_fused_bwd_kernel)[(GRID_E_STRIDE,)](
        gm0,
        gm1,
        gm2,
        wigner_flat,
        radial,
        x_full,
        edge_index,
        grad_edge,
        gwig,
        grad_rad,
        E,
        x_full.stride(0),
        x_full.stride(1),
        x_full.stride(2),
        edge_index.stride(0),
        C=C,
        BLOCK_C=C,
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


class WignerConv1FusedFunction(torch.autograd.Function):
    """
    Autograd function for the producer-side fused wigner->conv1 emit.

    Forward: (x_full [N,9,C], edge_index, wigner [E,9,9], radial) -> the three
    GEMM-ready packed buffers (m0, m1, m2).
    Backward: grads wrt node features (via the gather transpose), wigner, radial.
    """

    @staticmethod
    def forward(
        ctx,
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_flat: torch.Tensor,
        radial: torch.Tensor,
        C: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_full: Node features [N, 9, C] (L-major).
            edge_index: Edge indices [2, E].
            wigner_flat: Flattened Wigner matrices [E, 81].
            radial: Per-layer conv1 radial embedding [E, 6*2C] (rad_func applied).
            C: sphere_channels.

        Returns:
            (m0, m1, m2) GEMM-ready packed buffers.
        """
        x_full = x_full.contiguous()
        radial = radial.contiguous()
        wigner_flat = wigner_flat.contiguous()
        E = edge_index.shape[1]
        C2 = 2 * C
        dev, dt = x_full.device, x_full.dtype

        m0 = torch.empty((E, 3 * C2), device=dev, dtype=dt)
        m1 = torch.empty((E, 4 * C2), device=dev, dtype=dt)
        m2 = torch.empty((E, 2 * C2), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_conv1_fused_fwd(
            x_full, edge_index, wigner_flat, radial, m0, m1, m2, C
        )

        ctx.save_for_backward(edge_index, wigner_flat, radial, x_full)
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
            grad_x [N, 9, C], None (edge_index), grad_wigner [E, 81],
            grad_radial [E, 6*2C], None (C).
        """
        edge_index, wigner_flat, radial, x_full = ctx.saved_tensors
        N, C = ctx.N, ctx.C
        E = edge_index.shape[1]
        C2 = 2 * C
        dev, dt = x_full.device, x_full.dtype

        # gwig is zeroed: only the block-diagonal entries are written by the kernel.
        grad_edge = torch.empty((E, 9, C2), device=dev, dtype=dt)
        gwig = torch.zeros((E, 81), device=dev, dtype=dt)
        grad_rad = torch.empty((E, 6 * C2), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_conv1_fused_bwd(
            gm0.contiguous(),
            gm1.contiguous(),
            gm2.contiguous(),
            wigner_flat,
            radial,
            x_full,
            edge_index,
            grad_edge,
            gwig,
            grad_rad,
            C,
        )

        # Scatter per-edge grad wrt x (L-major src|tgt) to node gradients.
        grad_x = torch.zeros((N, 9, C), device=dev, dtype=dt)
        gsrc = grad_edge[:, :, :C].reshape(E, 9 * C)
        gtgt = grad_edge[:, :, C:].reshape(E, 9 * C)
        gxf = grad_x.view(N, 9 * C)
        gxf.index_add_(0, edge_index[0], gsrc)
        gxf.index_add_(0, edge_index[1], gtgt)
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
        wigner: Wigner rotation matrices [E, 9, 9].
        radial: Per-layer conv1 radial embedding [E, 6*2C] (rad_func applied).
        C: sphere_channels.

    Returns:
        (m0, m1, m2) GEMM-ready packed buffers.
    """
    wigner_flat = wigner.reshape(edge_index.shape[1], -1)
    return WignerConv1FusedFunction.apply(x_full, edge_index, wigner_flat, radial, C)


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
    wrap_triton(wigner_inv_conv2_fused_fwd_kernel)[(GRID_E_STRIDE, num_c_blocks)](
        g0,
        g1,
        g2,
        wigner_flat,
        out,
        E,
        C,
        BLOCK_C=C,
        GRID_E_STRIDE=GRID_E_STRIDE,
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
    wrap_triton(wigner_inv_conv2_fused_bwd_kernel)[(GRID_E_STRIDE,)](
        grad_out,
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
        GRID_E_STRIDE=GRID_E_STRIDE,
        num_warps=1,
    )


class WignerInvConv2FusedFunction(torch.autograd.Function):
    """
    Autograd function for the consumer-side fused inv-wigner <- conv2 emit.

    Forward: (g0 [E,3C], g1 [E,4C], g2 [E,2C], wigner [E,9,9]) -> x_rotated
    [E, 9, C] (L-major).
    Backward: grads wrt the three GEMM buffers and wigner.
    """

    @staticmethod
    def forward(
        ctx,
        g0: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        wigner_flat: torch.Tensor,
        C: int,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            g0: conv2 fc_m0 output [E, 3C] (rows M0,M1,M2).
            g1: conv2 m=1 block-GEMM output [E, 4C] (rows M3,M4,M5,M6).
            g2: conv2 m=2 block-GEMM output [E, 2C] (rows M7,M8).
            wigner_flat: Flattened inverse Wigner [E, 81] (envelope pre-fused).
            C: sphere_channels.

        Returns:
            x_rotated [E, 9, C] (L-major).
        """
        g0 = g0.contiguous()
        g1 = g1.contiguous()
        g2 = g2.contiguous()
        wigner_flat = wigner_flat.contiguous()
        E = g0.shape[0]
        dev, dt = g0.device, g0.dtype

        out = torch.empty((E, 9, C), device=dev, dtype=dt)

        torch.ops.fairchem._kernel_wigner_inv_conv2_fused_fwd(
            g0, g1, g2, wigner_flat, out, C
        )

        ctx.save_for_backward(g0, g1, g2, wigner_flat)
        ctx.C = C
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass.

        Args:
            grad_out: Grad wrt x_rotated [E, 9, C] (L-major).

        Returns:
            dg0 [E, 3C], dg1 [E, 4C], dg2 [E, 2C], dw [E, 81], None (C).
        """
        g0, g1, g2, wigner_flat = ctx.saved_tensors
        C = ctx.C
        E = g0.shape[0]
        dev, dt = g0.device, g0.dtype

        # dw is zeroed: only the block-diagonal entries are written by the kernel.
        dg0 = torch.empty((E, 3 * C), device=dev, dtype=dt)
        dg1 = torch.empty((E, 4 * C), device=dev, dtype=dt)
        dg2 = torch.empty((E, 2 * C), device=dev, dtype=dt)
        dw = torch.zeros((E, 81), device=dev, dtype=dt)

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
        wigner: inverse Wigner (envelope pre-fused) [E, 9, 9].
        C: sphere_channels.

    Returns:
        x_rotated [E, 9, C] (L-major).
    """
    E = g0.shape[0]
    wigner_flat = wigner.reshape(E, -1)
    return WignerInvConv2FusedFunction.apply(g0, g1, g2, wigner_flat, C)
