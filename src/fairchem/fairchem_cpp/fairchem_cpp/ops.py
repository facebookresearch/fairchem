"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

__all__ = ["segment_mm"]

# This code is derived from Deep Graph Library DGL, licensed under the Apache License 2.0.
# See https://www.apache.org/licenses/LICENSE-2.0 for more information.
# https://github.com/dmlc/dgl

# ---------------------------------------------------------------------------
# Level 3 — Raw kernel wrappers (no autograd graph)
# ---------------------------------------------------------------------------


def _segment_mm_fwd(A, B, seglen, b_trans=False):
    """
    Raw forward: C = A @ B (or A @ B^T when b_trans=True) per segment.
    """
    A = A.contiguous()
    B = B.contiguous()
    seglen = seglen.contiguous()
    if not b_trans:
        C = torch.empty((A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype)
    else:
        C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    torch.ops.fairchem_cpp.segment_mm(A, B, C, seglen, b_trans)
    return C


def _segment_mm_bwd_b(A, dC, seglen):
    """
    Raw backward-B: dB_i = A_i^T @ dC_i per segment.
    """
    A = A.contiguous()
    dC = dC.contiguous()
    seglen = seglen.contiguous()
    dB = torch.empty(
        (seglen.numel(), A.shape[1], dC.shape[1]), device=A.device, dtype=A.dtype
    )
    torch.ops.fairchem_cpp.segment_mm_backward(A, dC, dB, seglen)
    return dB


# ---------------------------------------------------------------------------
# Level 2 — Leaf autograd Functions (@once_differentiable)
# ---------------------------------------------------------------------------


class _SegmentMMLeaf(torch.autograd.Function):
    """
    Leaf-level forward: C = _segment_mm_fwd(A, B, seg, b_trans).
    Backward uses raw kernels with @once_differentiable (no further graph).
    """

    @staticmethod
    def forward(ctx, A, B, seglen, b_trans):
        ctx.save_for_backward(A, B, seglen)
        ctx.b_trans = b_trans
        return _segment_mm_fwd(A, B, seglen, b_trans=b_trans)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, ddC):
        A, B, seglen = ctx.saved_tensors
        b_trans = ctx.b_trans
        ddA = ddB = None

        if ctx.needs_input_grad[0]:
            ddA = _segment_mm_fwd(ddC, B, seglen, b_trans=not b_trans)

        if ctx.needs_input_grad[1]:
            if not b_trans:
                ddB = _segment_mm_bwd_b(A, ddC, seglen)
            else:
                ddB = _segment_mm_bwd_b(ddC, A, seglen)

        return ddA, ddB, None, None


class _BackwardBLeaf(torch.autograd.Function):
    """
    Leaf-level backward-B: dB = _segment_mm_bwd_b(A, dZ, seg).
    Backward uses raw kernels with @once_differentiable (no further graph).
    """

    @staticmethod
    def forward(ctx, A, dZ, seglen):
        ctx.save_for_backward(A, dZ, seglen)
        return _segment_mm_bwd_b(A, dZ, seglen)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, ddB):
        A, dZ, seglen = ctx.saved_tensors
        ddA = dd_dZ = None

        if ctx.needs_input_grad[0]:
            # d(A^T @ dZ)/dA = dZ @ ddB^T
            ddA = _segment_mm_fwd(dZ, ddB, seglen, b_trans=True)

        if ctx.needs_input_grad[1]:
            # d(A^T @ dZ)/d(dZ) = A @ ddB
            dd_dZ = _segment_mm_fwd(A, ddB, seglen, b_trans=False)

        return ddA, dd_dZ, None


# ---------------------------------------------------------------------------
# Level 1 — Top-level autograd (graph-building backward for double bwd)
# ---------------------------------------------------------------------------


class _SegmentMM(torch.autograd.Function):
    """
    Top-level autograd: C = A @ B per segment.

    Backward uses _SegmentMMLeaf and _BackwardBLeaf so that
    create_graph=True produces a differentiable graph for double backward.
    """

    @staticmethod
    def forward(ctx, A, B, seglen):
        ctx.save_for_backward(A, B, seglen)
        return _segment_mm_fwd(A, B, seglen, b_trans=False)

    @staticmethod
    def backward(ctx, dZ):
        A, B, seglen = ctx.saved_tensors
        A_grad = B_grad = None

        if ctx.needs_input_grad[0]:
            # dA = dZ @ B^T
            A_grad = _SegmentMMLeaf.apply(dZ, B, seglen, True)

        if ctx.needs_input_grad[1]:
            # dB_i = A_i^T @ dZ_i
            B_grad = _BackwardBLeaf.apply(A, dZ, seglen)

        return A_grad, B_grad, None


def segment_mm(A, B, seglen_A):
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(B.shape[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return torch.cat(C)
    else:
        # if autocasting make sure weights are same type
        B = B.to(A.dtype)
        return _SegmentMM.apply(A, B, seglen_A)
