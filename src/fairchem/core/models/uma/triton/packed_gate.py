"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.triton.kernels import (
    packed_gate_bwd_kernel,
    packed_gate_fwd_kernel,
)

_PACKED_GATE_GRID_E_STRIDE = 131072


def _packed_gate_differentiable_backward(
    gy0: Tensor,
    gy1: Tensor,
    gy2: Tensor,
    x0_full: Tensor,
    x1: Tensor,
    x2: Tensor,
    channels: int,
) -> tuple[Tensor, Tensor, Tensor]:
    gate0, gate1 = x0_full[:, : 2 * channels].sigmoid().split(channels, dim=1)
    scalar, x0_l1, x0_l2 = x0_full[:, 2 * channels :].split(channels, dim=1)
    gy0_scalar, gy0_l1, gy0_l2 = gy0.split(channels, dim=1)
    gy1_0, gy1_1, gy1_2, gy1_3 = gy1.split(channels, dim=1)
    x1_0, x1_1, x1_2, x1_3 = x1.split(channels, dim=1)
    gy2_0, gy2_1 = gy2.split(channels, dim=1)
    x2_0, x2_1 = x2.split(channels, dim=1)

    scalar_sigmoid = scalar.sigmoid()
    scalar_grad = gy0_scalar * scalar_sigmoid * (1.0 + scalar * (1.0 - scalar_sigmoid))
    gate0_grad = gy0_l1 * x0_l1 + gy1_0 * x1_0 + gy1_2 * x1_2
    gate1_grad = (
        gy0_l2 * x0_l2 + gy1_1 * x1_1 + gy1_3 * x1_3 + gy2_0 * x2_0 + gy2_1 * x2_1
    )
    gx0_full = torch.cat(
        (
            gate0_grad * gate0 * (1.0 - gate0),
            gate1_grad * gate1 * (1.0 - gate1),
            scalar_grad,
            gy0_l1 * gate0,
            gy0_l2 * gate1,
        ),
        dim=1,
    )
    gx1 = gy1 * torch.cat((gate0, gate1, gate0, gate1), dim=1)
    gx2 = gy2 * torch.cat((gate1, gate1), dim=1)
    return gx0_full, gx1, gx2


def _validate_inputs(x0_full: Tensor, x1: Tensor, x2: Tensor, channels: int) -> None:
    edges = x0_full.shape[0] if x0_full.ndim == 2 else None
    if channels < 1 or channels & (channels - 1):
        raise ValueError("packed gate requires power-of-two channels")
    if x0_full.ndim != 2 or x0_full.shape[1] != 5 * channels:
        raise ValueError("x0_full must have shape [E, 5C]")
    if x1.shape != (edges, 4 * channels):
        raise ValueError("x1 must have shape [E, 4C]")
    if x2.shape != (edges, 2 * channels):
        raise ValueError("x2 must have shape [E, 2C]")
    if (
        x0_full.dtype != torch.float32
        or x1.dtype != torch.float32
        or x2.dtype != torch.float32
    ):
        raise ValueError("packed gate requires float32 inputs")
    if not x0_full.is_cuda or not x1.is_cuda or not x2.is_cuda:
        raise ValueError("packed gate requires CUDA inputs")
    if x1.device != x0_full.device or x2.device != x0_full.device:
        raise ValueError("packed gate inputs must use the same CUDA device")


@triton_op("fairchem::_packed_gate_fwd", mutates_args=("y0", "y1", "y2"))
def _packed_gate_fwd(
    x0_full: Tensor,
    x1: Tensor,
    x2: Tensor,
    y0: Tensor,
    y1: Tensor,
    y2: Tensor,
    channels: int,
) -> None:
    edges = x0_full.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(edges, _PACKED_GATE_GRID_E_STRIDE)),)

    wrap_triton(packed_gate_fwd_kernel)[grid](
        x0_full,
        x1,
        x2,
        y0,
        y1,
        y2,
        edges,
        x0_full.stride(0),
        x0_full.stride(1),
        x1.stride(0),
        x1.stride(1),
        x2.stride(0),
        x2.stride(1),
        C=channels,
        GRID_E_STRIDE=_PACKED_GATE_GRID_E_STRIDE,
        num_warps=1,
    )


@triton_op(
    "fairchem::_packed_gate_bwd",
    mutates_args=("gx0_full", "gx1", "gx2"),
)
def _packed_gate_bwd(
    gy0: Tensor,
    gy1: Tensor,
    gy2: Tensor,
    x0_full: Tensor,
    x1: Tensor,
    x2: Tensor,
    gx0_full: Tensor,
    gx1: Tensor,
    gx2: Tensor,
    channels: int,
) -> None:
    edges = x0_full.shape[0]

    def grid(_meta):
        return (torch.sym_max(1, torch.sym_min(edges, _PACKED_GATE_GRID_E_STRIDE)),)

    wrap_triton(packed_gate_bwd_kernel)[grid](
        gy0,
        gy1,
        gy2,
        x0_full,
        x1,
        x2,
        gx0_full,
        gx1,
        gx2,
        edges,
        x0_full.stride(0),
        x0_full.stride(1),
        x1.stride(0),
        x1.stride(1),
        x2.stride(0),
        x2.stride(1),
        C=channels,
        GRID_E_STRIDE=_PACKED_GATE_GRID_E_STRIDE,
        num_warps=1,
    )


class PackedGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0_full, x1, x2, channels):
        y0 = torch.empty(
            (x0_full.shape[0], 3 * channels),
            device=x0_full.device,
            dtype=x0_full.dtype,
        )
        y1 = torch.empty(x1.shape, device=x1.device, dtype=x1.dtype)
        y2 = torch.empty(x2.shape, device=x2.device, dtype=x2.dtype)
        torch.ops.fairchem._packed_gate_fwd(x0_full, x1, x2, y0, y1, y2, channels)
        ctx.save_for_backward(x0_full, x1, x2)
        ctx.channels = channels
        return y0, y1, y2

    @staticmethod
    def backward(ctx, gy0, gy1, gy2):
        x0_full, x1, x2 = ctx.saved_tensors
        if torch.is_grad_enabled():
            return (
                *_packed_gate_differentiable_backward(
                    gy0, gy1, gy2, x0_full, x1, x2, ctx.channels
                ),
                None,
            )
        gx0_full = torch.empty(
            x0_full.shape, device=x0_full.device, dtype=x0_full.dtype
        )
        gx1 = torch.empty(x1.shape, device=x1.device, dtype=x1.dtype)
        gx2 = torch.empty(x2.shape, device=x2.device, dtype=x2.dtype)
        torch.ops.fairchem._packed_gate_bwd(
            gy0.contiguous(),
            gy1.contiguous(),
            gy2.contiguous(),
            x0_full,
            x1,
            x2,
            gx0_full,
            gx1,
            gx2,
            ctx.channels,
        )
        return gx0_full, gx1, gx2, None


def packed_gate_op(
    x0_full: Tensor,
    x1: Tensor,
    x2: Tensor,
    channels: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(x0_full, x1, x2, channels)
    return PackedGateFunction.apply(x0_full, x1, x2, channels)
