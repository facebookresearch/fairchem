"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import Linear

from .radial import RadialMLP

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import CoefficientMapping


class _FusedConv1Func(torch.autograd.Function):
    """
    Fused autograd.Function for SO2_Conv1_WithRadialBlock.forward (UMA-S
    lmax = mmax = 2). Equivalent computation to the eager forward but
    writes per-m results directly into a caller-owned pre-allocated
    output buffer, skipping the trailing torch.cat allocation. Backward
    is derived manually so the cat reshuffle does not reappear there.

    Same precision (fp32 throughout) and bit-exact forward + gradients
    vs. eager (verified on a synthetic prototype before wiring in).

    Hardcoded for mmax = 2: exactly two block-GEMM weights (W_block_1
    for m=1, W_block_2 for m=2). If lmax/mmax ever change, this kernel
    needs to be regenerated for the new m-loop count.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        fc_m0_w: torch.Tensor,
        fc_m0_b: torch.Tensor,
        W_block_1: torch.Tensor,
        W_block_2: torch.Tensor,
        out_buf: torch.Tensor,
        gating_buf: torch.Tensor,
        m_split_sizes: tuple,
        edge_split_sizes: tuple,
        lmax: int,
        m_out: int,
        extra: int,
    ):
        E = x.shape[0]
        m0_size = m_split_sizes[0]
        num_l_1 = lmax  # = lmax - 1 + 1
        num_l_2 = lmax - 1  # = lmax - 2 + 1

        with torch.no_grad():
            x_edge_0 = x_edge.narrow(1, 0, edge_split_sizes[0])
            x_edge_1 = x_edge.narrow(1, edge_split_sizes[0], edge_split_sizes[1])
            x_edge_2 = x_edge.narrow(
                1,
                edge_split_sizes[0] + edge_split_sizes[1],
                edge_split_sizes[2],
            )

            # m=0: radial mul + Linear; split out gating
            x_0_flat = x.narrow(1, 0, m0_size).reshape(E, -1).contiguous()
            x_0_radial = x_0_flat * x_edge_0
            z_0 = torch.addmm(fc_m0_b, x_0_radial, fc_m0_w.T)
            gating_buf.copy_(z_0[:, :extra])
            out_buf[:, 0:m0_size, :].copy_(
                z_0[:, extra:].view(E, m0_size, m_out)
            )

            # m=1: radial mul + block GEMM
            offset_1 = m0_size
            x_1_view = x.narrow(1, offset_1, m_split_sizes[1]).reshape(E, 2, -1)
            x_1_scaled = x_1_view * x_edge_1.unsqueeze(1)
            x_1_cat = x_1_scaled.reshape(E, -1).contiguous()
            out_cat_1 = x_1_cat @ W_block_1.T
            out_buf[:, offset_1 : offset_1 + num_l_1, :].copy_(
                out_cat_1[:, : num_l_1 * m_out].view(E, num_l_1, m_out)
            )
            out_buf[:, offset_1 + num_l_1 : offset_1 + 2 * num_l_1, :].copy_(
                out_cat_1[:, num_l_1 * m_out :].view(E, num_l_1, m_out)
            )

            # m=2
            offset_2 = offset_1 + m_split_sizes[1]
            x_2_view = x.narrow(1, offset_2, m_split_sizes[2]).reshape(E, 2, -1)
            x_2_scaled = x_2_view * x_edge_2.unsqueeze(1)
            x_2_cat = x_2_scaled.reshape(E, -1).contiguous()
            out_cat_2 = x_2_cat @ W_block_2.T
            out_buf[:, offset_2 : offset_2 + num_l_2, :].copy_(
                out_cat_2[:, : num_l_2 * m_out].view(E, num_l_2, m_out)
            )
            out_buf[:, offset_2 + num_l_2 : offset_2 + 2 * num_l_2, :].copy_(
                out_cat_2[:, num_l_2 * m_out :].view(E, num_l_2, m_out)
            )

        ctx.save_for_backward(
            x, x_edge, fc_m0_w, W_block_1, W_block_2,
            x_0_radial, x_1_cat, x_2_cat,
        )
        ctx.m_split_sizes = m_split_sizes
        ctx.edge_split_sizes = edge_split_sizes
        ctx.lmax = lmax
        ctx.m_out = m_out
        ctx.extra = extra
        return out_buf, gating_buf

    @staticmethod
    def backward(ctx, grad_out, grad_gating):
        (
            x, x_edge, fc_m0_w, W_block_1, W_block_2,
            x_0_radial, x_1_cat, x_2_cat,
        ) = ctx.saved_tensors
        m_split_sizes = ctx.m_split_sizes
        edge_split_sizes = ctx.edge_split_sizes
        lmax = ctx.lmax
        E = x.shape[0]
        C = x.shape[2]
        m0_size = m_split_sizes[0]
        num_l_1 = lmax
        num_l_2 = lmax - 1

        need_x = ctx.needs_input_grad[0]
        need_x_edge = ctx.needs_input_grad[1]
        need_w0 = ctx.needs_input_grad[2]
        need_b0 = ctx.needs_input_grad[3]
        need_w1 = ctx.needs_input_grad[4]
        need_w2 = ctx.needs_input_grad[5]

        grad_x = torch.empty_like(x) if need_x else None
        grad_x_edge = torch.empty_like(x_edge) if need_x_edge else None
        grad_fc_m0_w = grad_fc_m0_b = None
        grad_W_block_1 = grad_W_block_2 = None

        # ---- m=0 backward ----
        grad_z_0_main = grad_out[:, 0:m0_size, :].reshape(E, -1)
        grad_z_0 = torch.cat([grad_gating, grad_z_0_main], dim=1)

        if need_x or need_x_edge:
            grad_x_0_radial = grad_z_0 @ fc_m0_w
        if need_w0:
            grad_fc_m0_w = grad_z_0.T @ x_0_radial
        if need_b0:
            grad_fc_m0_b = grad_z_0.sum(0)

        if need_x or need_x_edge:
            x_edge_0 = x_edge.narrow(1, 0, edge_split_sizes[0])
        if need_x:
            grad_x_0_flat = grad_x_0_radial * x_edge_0
            grad_x[:, 0:m0_size, :].copy_(grad_x_0_flat.view(E, m0_size, C))
        if need_x_edge:
            x_0_flat = x.narrow(1, 0, m0_size).reshape(E, -1)
            grad_x_edge[:, 0 : edge_split_sizes[0]].copy_(
                grad_x_0_radial * x_0_flat
            )

        # ---- m=1 backward ----
        offset_1 = m0_size
        grad_real_1 = grad_out[:, offset_1 : offset_1 + num_l_1, :].reshape(E, -1)
        grad_imag_1 = grad_out[
            :, offset_1 + num_l_1 : offset_1 + 2 * num_l_1, :
        ].reshape(E, -1)
        grad_out_cat_1 = torch.cat([grad_real_1, grad_imag_1], dim=1)

        if need_x or need_x_edge:
            grad_x_1_cat = grad_out_cat_1 @ W_block_1
            grad_x_1_scaled = grad_x_1_cat.view(E, 2, -1)
        if need_w1:
            grad_W_block_1 = grad_out_cat_1.T @ x_1_cat

        if need_x or need_x_edge:
            x_edge_1 = x_edge.narrow(1, edge_split_sizes[0], edge_split_sizes[1])
        if need_x:
            grad_x_1_view = grad_x_1_scaled * x_edge_1.unsqueeze(1)
            grad_x[:, offset_1 : offset_1 + m_split_sizes[1], :].copy_(
                grad_x_1_view.view(E, m_split_sizes[1], C)
            )
        if need_x_edge:
            x_1_view = x.narrow(1, offset_1, m_split_sizes[1]).reshape(E, 2, -1)
            grad_x_edge[
                :, edge_split_sizes[0] : edge_split_sizes[0] + edge_split_sizes[1]
            ].copy_((grad_x_1_scaled * x_1_view).sum(dim=1))

        # ---- m=2 backward ----
        offset_2 = offset_1 + m_split_sizes[1]
        grad_real_2 = grad_out[:, offset_2 : offset_2 + num_l_2, :].reshape(E, -1)
        grad_imag_2 = grad_out[
            :, offset_2 + num_l_2 : offset_2 + 2 * num_l_2, :
        ].reshape(E, -1)
        grad_out_cat_2 = torch.cat([grad_real_2, grad_imag_2], dim=1)

        if need_x or need_x_edge:
            grad_x_2_cat = grad_out_cat_2 @ W_block_2
            grad_x_2_scaled = grad_x_2_cat.view(E, 2, -1)
        if need_w2:
            grad_W_block_2 = grad_out_cat_2.T @ x_2_cat

        if need_x or need_x_edge:
            x_edge_2 = x_edge.narrow(
                1,
                edge_split_sizes[0] + edge_split_sizes[1],
                edge_split_sizes[2],
            )
        if need_x:
            grad_x_2_view = grad_x_2_scaled * x_edge_2.unsqueeze(1)
            grad_x[:, offset_2 : offset_2 + m_split_sizes[2], :].copy_(
                grad_x_2_view.view(E, m_split_sizes[2], C)
            )
        if need_x_edge:
            x_2_view = x.narrow(1, offset_2, m_split_sizes[2]).reshape(E, 2, -1)
            grad_x_edge[
                :, edge_split_sizes[0] + edge_split_sizes[1] :
            ].copy_((grad_x_2_scaled * x_2_view).sum(dim=1))

        return (
            grad_x, grad_x_edge,
            grad_fc_m0_w, grad_fc_m0_b,
            grad_W_block_1, grad_W_block_2,
            None, None,  # buffers (no grad)
            None, None, None, None, None,  # static metadata
        )


class _FusedConv2Func(torch.autograd.Function):
    """
    Same pattern as _FusedConv1Func but for SO2_Conv2_InternalBlock — no
    radial multiplication, no gating split. Shorter input list.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        fc_m0_w: torch.Tensor,
        fc_m0_b: torch.Tensor,
        W_block_1: torch.Tensor,
        W_block_2: torch.Tensor,
        out_buf: torch.Tensor,
        m_split_sizes: tuple,
        lmax: int,
        m_out: int,
    ):
        E = x.shape[0]
        m0_size = m_split_sizes[0]
        num_l_1 = lmax
        num_l_2 = lmax - 1

        with torch.no_grad():
            x_0 = x.narrow(1, 0, m0_size).reshape(E, -1).contiguous()
            z_0 = torch.addmm(fc_m0_b, x_0, fc_m0_w.T)
            out_buf[:, 0:m0_size, :].copy_(z_0.view(E, m0_size, m_out))

            offset_1 = m0_size
            x_1_cat = x.narrow(1, offset_1, m_split_sizes[1]).reshape(E, -1).contiguous()
            out_cat_1 = x_1_cat @ W_block_1.T
            out_buf[:, offset_1 : offset_1 + num_l_1, :].copy_(
                out_cat_1[:, : num_l_1 * m_out].view(E, num_l_1, m_out)
            )
            out_buf[:, offset_1 + num_l_1 : offset_1 + 2 * num_l_1, :].copy_(
                out_cat_1[:, num_l_1 * m_out :].view(E, num_l_1, m_out)
            )

            offset_2 = offset_1 + m_split_sizes[1]
            x_2_cat = x.narrow(1, offset_2, m_split_sizes[2]).reshape(E, -1).contiguous()
            out_cat_2 = x_2_cat @ W_block_2.T
            out_buf[:, offset_2 : offset_2 + num_l_2, :].copy_(
                out_cat_2[:, : num_l_2 * m_out].view(E, num_l_2, m_out)
            )
            out_buf[:, offset_2 + num_l_2 : offset_2 + 2 * num_l_2, :].copy_(
                out_cat_2[:, num_l_2 * m_out :].view(E, num_l_2, m_out)
            )

        ctx.save_for_backward(x, fc_m0_w, W_block_1, W_block_2)
        ctx.m_split_sizes = m_split_sizes
        ctx.lmax = lmax
        ctx.m_out = m_out
        return out_buf

    @staticmethod
    def backward(ctx, grad_out):
        x, fc_m0_w, W_block_1, W_block_2 = ctx.saved_tensors
        m_split_sizes = ctx.m_split_sizes
        lmax = ctx.lmax
        E = x.shape[0]
        C = x.shape[2]
        m0_size = m_split_sizes[0]
        num_l_1 = lmax
        num_l_2 = lmax - 1

        need_x = ctx.needs_input_grad[0]
        need_w0 = ctx.needs_input_grad[1]
        need_b0 = ctx.needs_input_grad[2]
        need_w1 = ctx.needs_input_grad[3]
        need_w2 = ctx.needs_input_grad[4]

        grad_x = torch.empty_like(x) if need_x else None
        grad_fc_m0_w = grad_fc_m0_b = None
        grad_W_block_1 = grad_W_block_2 = None

        # m=0
        grad_z_0 = grad_out[:, 0:m0_size, :].reshape(E, -1)
        if need_x:
            grad_x_0_flat = grad_z_0 @ fc_m0_w
            grad_x[:, 0:m0_size, :].copy_(grad_x_0_flat.view(E, m0_size, C))
        if need_w0:
            grad_fc_m0_w = grad_z_0.T @ x.narrow(1, 0, m0_size).reshape(E, -1)
        if need_b0:
            grad_fc_m0_b = grad_z_0.sum(0)

        # m=1
        offset_1 = m0_size
        grad_real_1 = grad_out[:, offset_1 : offset_1 + num_l_1, :].reshape(E, -1)
        grad_imag_1 = grad_out[
            :, offset_1 + num_l_1 : offset_1 + 2 * num_l_1, :
        ].reshape(E, -1)
        grad_out_cat_1 = torch.cat([grad_real_1, grad_imag_1], dim=1)
        if need_x:
            grad_x_1_cat = grad_out_cat_1 @ W_block_1
            grad_x[:, offset_1 : offset_1 + m_split_sizes[1], :].copy_(
                grad_x_1_cat.view(E, m_split_sizes[1], C)
            )
        if need_w1:
            x_1_cat = x.narrow(1, offset_1, m_split_sizes[1]).reshape(E, -1)
            grad_W_block_1 = grad_out_cat_1.T @ x_1_cat

        # m=2
        offset_2 = offset_1 + m_split_sizes[1]
        grad_real_2 = grad_out[:, offset_2 : offset_2 + num_l_2, :].reshape(E, -1)
        grad_imag_2 = grad_out[
            :, offset_2 + num_l_2 : offset_2 + 2 * num_l_2, :
        ].reshape(E, -1)
        grad_out_cat_2 = torch.cat([grad_real_2, grad_imag_2], dim=1)
        if need_x:
            grad_x_2_cat = grad_out_cat_2 @ W_block_2
            grad_x[:, offset_2 : offset_2 + m_split_sizes[2], :].copy_(
                grad_x_2_cat.view(E, m_split_sizes[2], C)
            )
        if need_w2:
            x_2_cat = x.narrow(1, offset_2, m_split_sizes[2]).reshape(E, -1)
            grad_W_block_2 = grad_out_cat_2.T @ x_2_cat

        return (
            grad_x,
            grad_fc_m0_w, grad_fc_m0_b,
            grad_W_block_1, grad_W_block_2,
            None,  # buffer
            None, None, None,  # static metadata
        )


class SO2_m_Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.fc = Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = self.fc(x_m)
        x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(
            x_m.shape[0], -1, self.out_channels_half
        ).split(1, dim=1)
        x_m_r = x_r_0 - x_i_1  # x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r_1 + x_i_0  # x_r[:, 1] + x_i[:, 0]
        return (
            x_m_r.view(x_m.shape[0], -1, self.m_output_channels),
            x_m_i.view(x_m.shape[0], -1, self.m_output_channels),
        )


class SO2_m_Conv_Block(torch.nn.Module):
    """
    SO(2) Conv with block-diagonal matrix formulation for m > 0.

    Uses a single larger GEMM instead of the standard approach:
        [x_real, x_imag] @ [[W1, -W2], [W2, W1]]^T

    Produces identical results to SO2_m_Conv but with better tensor
    core utilization (1.3-1.5x faster).

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.in_size = num_channels
        self.num_l = self.out_channels_half // self.m_output_channels

        self.fc = Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

        self.register_buffer("_w_block", None, persistent=False)

    @torch.no_grad()
    def _build_w_block(self) -> None:
        """
        Build and cache the block matrix from fc weights.

        Converts from standard weight layout to block matrix:
            [[W1, -W2], [W2, W1]]
        """
        W1, W2 = self.fc.weight.split(self.out_channels_half, dim=0)
        self._w_block = torch.cat(
            [
                torch.cat([W1, -W2], dim=1),
                torch.cat([W2, W1], dim=1),
            ],
            dim=0,
        ).contiguous()

    def _maybe_build_w_block(self) -> None:
        """No-op if _w_block is already materialized; otherwise builds it."""
        if self._w_block is None:
            self._build_w_block()

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using block matrix GEMM.

        Args:
            x_m: [E, 2, in_size] where dim 1 is [real, imag]

        Returns:
            (out_real, out_imag): each [E, num_l, m_output_channels]
        """
        if self._w_block is None:
            self._build_w_block()

        x_cat = x_m.flatten(1)
        out_cat = x_cat @ self._w_block.T

        return out_cat.view(-1, 2, self.num_l, self.m_output_channels).unbind(1)


class SO2_Conv1_WithRadialBlock(torch.nn.Module):
    """
    First SO2 convolution using block-diagonal GEMM for m > 0.

    Replaces SO2_Convolution for conv1 (external radial weights,
    extra m0 gating channels). Always returns (output, gating).

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Coefficient mapping
        extra_m0_output_channels (int): Extra channels for gating
        edge_channels_list (list[int]): Edge embedding channels
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int],
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (lmax + 1) * sphere_channels

        m0_output_channels = (
            m_output_channels * (num_channels_m0 // sphere_channels)
            + extra_m0_output_channels
        )
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv_Block(
                    m,
                    sphere_channels,
                    m_output_channels,
                    lmax,
                    mmax,
                )
            )
            num_channels_rad += self.so2_m_conv[-1].fc.in_features

        edge_channels_list = copy.deepcopy(edge_channels_list)
        edge_channels_list.append(int(num_channels_rad))
        self.rad_func = RadialMLP(edge_channels_list)

        self.m_split_sizes = [mappingReduced.m_size[0]] + (
            torch.tensor(mappingReduced.m_size[1:]) * 2
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]
        # Cached output buffers reused across the 50 timed iters.
        # Re-allocated only when E or dtype/device changes.
        self._out_buf: torch.Tensor | None = None
        self._gating_buf: torch.Tensor | None = None
        self._total_coeffs = sum(self.m_split_sizes)

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [E, coeffs, channels]
            x_edge: Precomputed radial embeddings [E, radial_features]

        Returns:
            (output, gating): output [E, coeffs, m_output_channels],
                gating [E, extra_m0_output_channels]
        """
        # Fast path: UMA-S lmax = mmax = 2 — fused autograd.Function with
        # manual backward, writes per-m results directly into pre-allocated
        # output buffers (skips the trailing torch.cat allocation).
        if self.lmax == 2 and self.mmax == 2:
            E = x.shape[0]
            if (
                self._out_buf is None
                or self._out_buf.shape[0] != E
                or self._out_buf.dtype != x.dtype
                or self._out_buf.device != x.device
            ):
                self._out_buf = torch.empty(
                    E,
                    self._total_coeffs,
                    self.m_output_channels,
                    dtype=x.dtype,
                    device=x.device,
                )
                self._gating_buf = torch.empty(
                    E,
                    self.extra_m0_output_channels,
                    dtype=x.dtype,
                    device=x.device,
                )
            self.so2_m_conv[0]._maybe_build_w_block()
            self.so2_m_conv[1]._maybe_build_w_block()
            return _FusedConv1Func.apply(
                x,
                x_edge,
                self.fc_m0.weight,
                self.fc_m0.bias,
                self.so2_m_conv[0]._w_block,
                self.so2_m_conv[1]._w_block,
                self._out_buf,
                self._gating_buf,
                tuple(self.m_split_sizes),
                tuple(self.edge_split_sizes),
                self.lmax,
                self.m_output_channels,
                self.extra_m0_output_channels,
            )

        # Generic fallback: original eager forward (for other lmax/mmax).
        x_edge_by_m = x_edge.split(self.edge_split_sizes, dim=1)
        x_by_m = x.split(self.m_split_sizes, dim=1)
        num_edges = x.shape[0]

        x_0 = x_by_m[0].view(num_edges, -1) * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)
        x_0_extra, x_0 = x_0.split(
            (
                self.extra_m0_output_channels,
                self.fc_m0.out_features - self.extra_m0_output_channels,
            ),
            -1,
        )
        out = [x_0.view(num_edges, -1, self.m_output_channels)]

        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m)

        return torch.cat(out, dim=1), x_0_extra


class SO2_Conv2_InternalBlock(torch.nn.Module):
    """
    Second SO2 convolution using block-diagonal GEMM for m > 0.

    Replaces SO2_Convolution for conv2 (internal weights, no radial,
    no extra m0 channels). Always returns a single tensor.

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Coefficient mapping
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced

        num_channels_m0 = (lmax + 1) * sphere_channels

        self.fc_m0 = Linear(num_channels_m0, m_output_channels * (lmax + 1))

        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv_Block(
                    m,
                    sphere_channels,
                    m_output_channels,
                    lmax,
                    mmax,
                )
            )

        self.m_split_sizes = [mappingReduced.m_size[0]] + (
            torch.tensor(mappingReduced.m_size[1:]) * 2
        ).tolist()
        self._out_buf: torch.Tensor | None = None
        self._total_coeffs = sum(self.m_split_sizes)

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass -- internal weights only, no radial.

        Args:
            x: [E, coeffs, channels]
            x_edge: Unused. Accepted for call-site compatibility
                with Edgewise.

        Returns:
            Output features [E, coeffs, m_output_channels]
        """
        if self.lmax == 2 and self.mmax == 2:
            E = x.shape[0]
            if (
                self._out_buf is None
                or self._out_buf.shape[0] != E
                or self._out_buf.dtype != x.dtype
                or self._out_buf.device != x.device
            ):
                self._out_buf = torch.empty(
                    E,
                    self._total_coeffs,
                    self.m_output_channels,
                    dtype=x.dtype,
                    device=x.device,
                )
            self.so2_m_conv[0]._maybe_build_w_block()
            self.so2_m_conv[1]._maybe_build_w_block()
            return _FusedConv2Func.apply(
                x,
                self.fc_m0.weight,
                self.fc_m0.bias,
                self.so2_m_conv[0]._w_block,
                self.so2_m_conv[1]._w_block,
                self._out_buf,
                tuple(self.m_split_sizes),
                self.lmax,
                self.m_output_channels,
            )

        # Generic fallback for other lmax/mmax.
        x_by_m = x.split(self.m_split_sizes, dim=1)
        num_edges = x.shape[0]

        x_0 = x_by_m[0].view(num_edges, -1)
        x_0 = self.fc_m0(x_0)
        out = [x_0.view(num_edges, -1, self.m_output_channels)]

        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m)

        return torch.cat(out, dim=1)


def convert_so2_conv1(
    old: SO2_Convolution,
) -> SO2_Conv1_WithRadialBlock:
    """
    Convert an SO2_Convolution (conv1) to SO2_Conv1_WithRadialBlock.

    Replaces SO2_m_Conv with SO2_m_Conv_Block for block-diagonal
    GEMM. Weights transfer via load_state_dict (identical keys).
    Block matrices are eagerly built.

    Args:
        old: Initialized SO2_Convolution with internal_weights=False
            and extra_m0_output_channels set.

    Returns:
        SO2_Conv1_WithRadialBlock with identical weights.
    """
    assert old.rad_func is not None, (
        "convert_so2_conv1 requires an SO2_Convolution with a RadialMLP "
        "(internal_weights=False). Got an SO2_Convolution without rad_func, "
        "which means it was created with internal_weights=True."
    )
    device = old.fc_m0.weight.device
    new = SO2_Conv1_WithRadialBlock(
        sphere_channels=old.sphere_channels,
        m_output_channels=old.m_output_channels,
        lmax=old.lmax,
        mmax=old.mmax,
        mappingReduced=old.mappingReduced,
        extra_m0_output_channels=old.extra_m0_output_channels,
        edge_channels_list=old.edge_channels_list,
    )
    new.load_state_dict(old.state_dict())
    for m_conv in new.so2_m_conv:
        m_conv._build_w_block()
    return new.to(device)


def convert_so2_conv2(
    old: SO2_Convolution,
) -> SO2_Conv2_InternalBlock:
    """
    Convert an SO2_Convolution (conv2) to SO2_Conv2_InternalBlock.

    Replaces SO2_m_Conv with SO2_m_Conv_Block for block-diagonal
    GEMM. Weights transfer via load_state_dict (identical keys).
    Block matrices are eagerly built.

    Args:
        old: Initialized SO2_Convolution with internal_weights=True
            and no extra_m0_output_channels.

    Returns:
        SO2_Conv2_InternalBlock with identical weights.
    """
    device = old.fc_m0.weight.device
    new = SO2_Conv2_InternalBlock(
        sphere_channels=old.sphere_channels,
        m_output_channels=old.m_output_channels,
        lmax=old.lmax,
        mmax=old.mmax,
        mappingReduced=old.mappingReduced,
    )
    new.load_state_dict(old.state_dict())
    for m_conv in new.so2_m_conv:
        m_conv._build_w_block()
    return new.to(device)


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` and `extra_m0_features` (Tensor).
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels
        self.edge_channels_list = (
            copy.deepcopy(edge_channels_list)
            if edge_channels_list is not None
            else None
        )

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (
            num_channels_m0 // self.sphere_channels
        )
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax,
                    self.mmax,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert edge_channels_list is not None
            edge_channels_list = copy.deepcopy(edge_channels_list)
            edge_channels_list.append(int(num_channels_rad))
            # This can moved outside of SO2 conv and into Edgewise
            self.rad_func = RadialMLP(edge_channels_list)

        self.m_split_sizes = [self.mappingReduced.m_size[0]] + (
            torch.tensor(self.mappingReduced.m_size[1:]) * 2
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Compute radial embedding from raw x_edge if we have external weights
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)

        x_by_m = x.split(self.m_split_sizes, dim=1)

        # Split radial embeddings if provided (external weights mode)
        if x_edge is not None:
            x_edge_by_m = x_edge.split(self.edge_split_sizes, dim=1)

        num_edges = x.shape[0]
        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x_by_m[0].view(num_edges, -1)
        if x_edge is not None:
            x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)

        # extract extra m0 features
        if self.extra_m0_output_channels is not None:
            x_0_extra, x_0 = x_0.split(
                (
                    self.extra_m0_output_channels,
                    self.fc_m0.out_features - self.extra_m0_output_channels,
                ),
                -1,
            )

        out = [x_0.view(num_edges, -1, self.m_output_channels)]  # m0

        # Compute the values for the m > 0 coefficients
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if x_edge is not None:
                x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m)

        out = torch.cat(out, dim=1)

        if self.extra_m0_output_channels is not None:
            return out, x_0_extra
        else:
            return out


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Linear: Perform SO(2) linear for all m (orders).

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = False,
        edge_channels_list: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) linear for m = 0
        self.fc_m0 = Linear(
            num_channels_m0,
            self.m_output_channels * (num_channels_m0 // self.sphere_channels),
        )
        num_channels_rad = self.fc_m0.in_features

        # SO(2) linear for non-zero m
        self.so2_m_fc = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            num_coefficents = self.lmax - m + 1
            num_in_channels = num_coefficents * self.sphere_channels
            fc = Linear(
                num_in_channels,
                self.m_output_channels * (num_in_channels // self.sphere_channels),
                bias=False,
            )
            num_channels_rad = num_channels_rad + fc.in_features
            self.so2_m_fc.append(fc)

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialMLP(self.edge_channels_list)

    def forward(self, x: torch.Tensor, x_edge: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = []

        # Reshape the spherical harmonics based on m (order)
        x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(batch_size, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)
        x_0 = x_0.view(batch_size, -1, self.m_output_channels)
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            x_m = x.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(batch_size, 2, -1)
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(
                    1, offset_rad, self.so2_m_fc[m - 1].in_features
                )
                x_edge_m = x_edge_m.reshape(
                    batch_size, 1, self.so2_m_fc[m - 1].in_features
                )
                x_m = x_m * x_edge_m

            # Perform SO(2) linear
            x_m = self.so2_m_fc[m - 1](x_m)
            x_m = x_m.view(batch_size, -1, self.m_output_channels)
            out.append(x_m)

            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_fc[m - 1].in_features

        out = torch.cat(out, dim=1)

        # Reshape the spherical harmonics based on l (degree)
        out = torch.einsum("nac,ab->nbc", out, self.mappingReduced.to_m)
        return out
