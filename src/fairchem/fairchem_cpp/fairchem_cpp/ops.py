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
# Level 0 — register_fake for the raw C++ ops
#
# The C++ ops `fairchem_cpp::segment_mm` and `segment_mm_backward` write
# in-place into a pre-allocated output tensor (C / dB) and return void.
# Their schema is `... Tensor C, ... -> ()` (no aliasing annotation), so
# dynamo cannot infer that C is mutated. We just register a no-op fake;
# the in-place mutation is hidden inside the higher-level custom_op
# wrappers below, which present a purely functional interface.
# ---------------------------------------------------------------------------


@torch.library.register_fake("fairchem_cpp::segment_mm")
def _segment_mm_fake(A, B, C, seglen, b_trans):
    return None


@torch.library.register_fake("fairchem_cpp::segment_mm_backward")
def _segment_mm_backward_fake(A, dC, dB, seglen):
    return None


# ---------------------------------------------------------------------------
# Level 1 — Raw kernel wrappers (no autograd graph, no compile visibility)
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
# Level 2 — Compile-visible functional wrappers (used inside backward)
#
# Each wrapper presents a "pure function" view of the kernel: takes
# inputs, returns a freshly-allocated output. The internal in-place
# write to the empty buffer is invisible from outside. Both have
# register_fake so dynamo can shape-propagate through them without
# graph-breaking.
# ---------------------------------------------------------------------------


@torch.library.custom_op("fairchem_cpp::segment_mm_fwd_op", mutates_args=())
def _segment_mm_fwd_op(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
    b_trans: bool,
) -> torch.Tensor:
    return _segment_mm_fwd(A, B, seglen, b_trans=b_trans)


@_segment_mm_fwd_op.register_fake
def _(A, B, seglen, b_trans):
    if b_trans:
        return torch.empty(
            (A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype
        )
    return torch.empty(
        (A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype
    )


@torch.library.custom_op("fairchem_cpp::segment_mm_bwd_b_op", mutates_args=())
def _segment_mm_bwd_b_op(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    return _segment_mm_bwd_b(A, dC, seglen)


@_segment_mm_bwd_b_op.register_fake
def _(A, dC, seglen):
    return torch.empty(
        (seglen.numel(), A.shape[1], dC.shape[1]),
        device=A.device,
        dtype=A.dtype,
    )


# ---------------------------------------------------------------------------
# Level 3 — Top-level autograd via torch.library.custom_op + register_autograd
#
# This is the single entry point the public API uses for the GPU path.
# It is the *only* op the user-visible call site exposes to dynamo:
# one named op (`fairchem_cpp::segment_mm_apply`) with a known fake
# impl, so dynamo treats it as opaque-with-known-shape and does NOT
# graph-break. Backward is supplied via register_autograd and uses the
# Level-2 functional wrappers so it is also fully traceable when
# AOTAutograd compiles the backward pass.
#
# Once-differentiable: the backward is not graph-tracked. Double-bwd is
# not supported (was already not supported by the prior _SegmentMMLeaf
# / _BackwardBLeaf pair under @once_differentiable).
# ---------------------------------------------------------------------------


@torch.library.custom_op("fairchem_cpp::segment_mm_apply", mutates_args=())
def _segment_mm_apply(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    """Top-level segment_mm: C_i = A_i @ B_i per segment."""
    return _segment_mm_fwd(A, B, seglen, b_trans=False)


@_segment_mm_apply.register_fake
def _(A, B, seglen):
    return torch.empty(
        (A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype
    )


def _segment_mm_apply_setup_context(ctx, inputs, output):
    A, B, seglen = inputs
    ctx.save_for_backward(A, B, seglen)


def _segment_mm_apply_backward(ctx, dZ):
    A, B, seglen = ctx.saved_tensors
    A_grad = B_grad = None
    if ctx.needs_input_grad[0]:
        # dA_i = dZ_i @ B_i^T  (segmented)
        A_grad = _segment_mm_fwd_op(dZ, B, seglen, True)
    if ctx.needs_input_grad[1]:
        # dB_i = A_i^T @ dZ_i  (per-segment)
        B_grad = _segment_mm_bwd_b_op(A, dZ, seglen)
    return A_grad, B_grad, None


_segment_mm_apply.register_autograd(
    _segment_mm_apply_backward,
    setup_context=_segment_mm_apply_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_mm(A, B, seglen_A):
    """
    Per-segment matrix multiply: C_i = A_i @ B_i for each segment i,
    where ``seglen_A`` defines the number of rows of A in segment i.

    For CUDA inputs the call routes through the compile-friendly
    `fairchem_cpp::segment_mm_apply` custom op (no graph break, proper
    register_autograd backward). For CPU inputs it falls back to a
    Python loop of plain matmuls.
    """
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(B.shape[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return torch.cat(C)
    else:
        # If autocasting, make sure weights are same type
        B = B.to(A.dtype)
        return torch.ops.fairchem_cpp.segment_mm_apply(A, B, seglen_A)
