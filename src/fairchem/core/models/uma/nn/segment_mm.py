"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.nn.segment_mm_cpu import (
    _segment_mm_backward_b_cpu,
    _segment_mm_forward_cpu,
)
from fairchem.core.models.uma.nn.segment_mm_gpu import (
    _segment_mm_backward_b_gpu,
    _segment_mm_forward_gpu,
)


def segment_mm_ref(
    A: torch.Tensor, B: torch.Tensor, seglen: torch.Tensor
) -> torch.Tensor:
    """
    Reference segment_mm using a for-loop of torch.matmul.

    Args:
        A: [total_rows, K]
        B: [num_segments, K, N]
        seglen: [num_segments] (CPU, int)

    Returns:
        C: [total_rows, N]
    """
    num_seg = seglen.numel()
    N = B.shape[2]
    C = torch.empty(A.shape[0], N, dtype=A.dtype, device=A.device)
    offset = 0
    for i in range(num_seg):
        m = int(seglen[i].item())
        C[offset : offset + m] = A[offset : offset + m] @ B[i]
        offset += m
    return C


# ---------------------------------------------------------------------------
# Unified dispatch: routes to CPU or GPU based on tensor device
# ---------------------------------------------------------------------------


def _segment_mm_forward_dispatch(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
    *,
    b_trans: bool,
) -> torch.Tensor:
    if A.device.type == "cuda":
        return _segment_mm_forward_gpu(A, B, seglen, b_trans=b_trans)
    return _segment_mm_forward_cpu(A, B, seglen, b_trans=b_trans)


def _segment_mm_backward_b_dispatch(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    if A.device.type == "cuda":
        return _segment_mm_backward_b_gpu(A, dC, seglen)
    return _segment_mm_backward_b_cpu(A, dC, seglen)


# ---------------------------------------------------------------------------
# Three-level autograd (identical for CPU and GPU)
# ---------------------------------------------------------------------------


class _SegmentMMLeaf(torch.autograd.Function):
    """
    Leaf-level forward: C = forward_dispatch(A, B, seg, b_trans).
    Backward is @once_differentiable (no graph through this level).
    """

    @staticmethod
    def forward(ctx, A, B, seglen, b_trans):
        ctx.save_for_backward(A, B, seglen)
        ctx.b_trans = b_trans
        return _segment_mm_forward_dispatch(A, B, seglen, b_trans=b_trans)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, ddC):
        A, B, seglen = ctx.saved_tensors
        b_trans = ctx.b_trans
        ddA = ddB = None

        if ctx.needs_input_grad[0]:
            if not b_trans:
                ddA = _segment_mm_forward_dispatch(ddC, B, seglen, b_trans=True)
            else:
                ddA = _segment_mm_forward_dispatch(ddC, B, seglen, b_trans=False)

        if ctx.needs_input_grad[1]:
            if not b_trans:
                ddB = _segment_mm_backward_b_dispatch(A, ddC, seglen)
            else:
                ddB = _segment_mm_backward_b_dispatch(ddC, A, seglen)

        return ddA, ddB, None, None


class _BackwardBLeaf(torch.autograd.Function):
    """
    Leaf-level backward-B: dB = backward_b_dispatch(A, dZ, seg).
    Backward is @once_differentiable.
    """

    @staticmethod
    def forward(ctx, A, dZ, seglen):
        ctx.save_for_backward(A, dZ, seglen)
        return _segment_mm_backward_b_dispatch(A, dZ, seglen)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, ddB):
        A, dZ, seglen = ctx.saved_tensors
        ddA = dd_dZ = None

        if ctx.needs_input_grad[0]:
            ddA = _segment_mm_forward_dispatch(dZ, ddB, seglen, b_trans=True)

        if ctx.needs_input_grad[1]:
            dd_dZ = _segment_mm_forward_dispatch(A, ddB, seglen, b_trans=False)

        return ddA, dd_dZ, None


class SegmentMM(torch.autograd.Function):
    """
    Top-level autograd: C = A @ B per segment.

    Backward uses _SegmentMMLeaf and _BackwardBLeaf so that
    create_graph=True produces a differentiable graph.
    """

    @staticmethod
    def forward(ctx, A, B, seglen):
        ctx.save_for_backward(A, B, seglen)
        return _segment_mm_forward_dispatch(A, B, seglen, b_trans=False)

    @staticmethod
    def backward(ctx, dZ):
        A, B, seglen = ctx.saved_tensors
        A_grad = B_grad = None

        if ctx.needs_input_grad[0]:
            A_grad = _SegmentMMLeaf.apply(dZ, B, seglen, True)

        if ctx.needs_input_grad[1]:
            B_grad = _BackwardBLeaf.apply(A, dZ, seglen)

        return A_grad, B_grad, None


def segment_mm_double_backward(
    A: torch.Tensor, B: torch.Tensor, seglen: torch.Tensor
) -> torch.Tensor:
    """
    Double-backward-capable segment_mm.

    Args:
        A: [total_rows, K]
        B: [num_segments, K, N]
        seglen: [num_segments] (CPU, int)

    Returns:
        C: [total_rows, N]
    """
    return SegmentMM.apply(A, B, seglen)
