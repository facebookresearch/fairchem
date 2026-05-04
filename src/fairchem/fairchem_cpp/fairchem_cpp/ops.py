"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

__all__ = ["segment_mm"]

# This code is derived from Deep Graph Library DGL, licensed under the
# Apache License 2.0. See https://www.apache.org/licenses/LICENSE-2.0 for
# more information. https://github.com/dmlc/dgl
#
# C++ schema (functional, returns Tensor — see binding.cpp):
#   fairchem_cpp::segment_mm(Tensor A, Tensor B, Tensor seglen, bool b_trans) -> Tensor
#   fairchem_cpp::segment_mm_backward(Tensor A, Tensor dC, Tensor seglen) -> Tensor
#
# Both kernels allocate their output internally and return it. This is
# what lets us use a plain torch.autograd.Function below and have it
# compose cleanly with both eager autograd and torch.compile +
# AOTAutograd. There is no out-parameter mutation visible to the
# caller, so no aliasing annotation is needed.


# ---------------------------------------------------------------------------
# Fake (abstract) impls — only consulted by torch.compile / dynamo for
# symbolic shape propagation. In eager dispatch these are never called.
# ---------------------------------------------------------------------------


@torch.library.register_fake("fairchem_cpp::segment_mm")
def _segment_mm_fake(A, B, seglen, b_trans):
    if b_trans:
        return torch.empty(
            (A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype
        )
    return torch.empty(
        (A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype
    )


@torch.library.register_fake("fairchem_cpp::segment_mm_backward")
def _segment_mm_backward_fake(A, dC, seglen):
    return torch.empty(
        (seglen.numel(), A.shape[1], dC.shape[1]),
        device=A.device,
        dtype=A.dtype,
    )


# ---------------------------------------------------------------------------
# Thin Python wrappers around the raw C++ ops. They make the inputs
# contiguous (cuBLAS / our launcher requirement) and forward to the op.
# ---------------------------------------------------------------------------


def _fwd(A, B, seglen, b_trans=False):
    A = A.contiguous()
    B = B.contiguous()
    seglen = seglen.contiguous()
    return torch.ops.fairchem_cpp.segment_mm(A, B, seglen, b_trans)


def _bwd_b(A, dC, seglen):
    A = A.contiguous()
    dC = dC.contiguous()
    seglen = seglen.contiguous()
    return torch.ops.fairchem_cpp.segment_mm_backward(A, dC, seglen)


# ---------------------------------------------------------------------------
# Single autograd.Function. Works in both eager (zero dynamo cost) and
# torch.compile (dynamo traces forward as an opaque op via the
# register_fake above; AOTAutograd handles backward).
# ---------------------------------------------------------------------------


class _SegmentMM(torch.autograd.Function):
    """C_i = A_i @ B_i per segment. Once-differentiable (no double-bwd).
    """

    @staticmethod
    def forward(ctx, A, B, seglen):
        ctx.save_for_backward(A, B, seglen)
        return _fwd(A, B, seglen, b_trans=False)

    @staticmethod
    def backward(ctx, dZ):
        A, B, seglen = ctx.saved_tensors
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            # dA_i = dZ_i @ B_i^T (per-segment)
            A_grad = _fwd(dZ, B, seglen, b_trans=True)
        if ctx.needs_input_grad[1]:
            # dB_i = A_i^T @ dZ_i (per-segment)
            B_grad = _bwd_b(A, dZ, seglen)
        return A_grad, B_grad, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_mm(A, B, seglen_A):
    """
    Per-segment matrix multiply: C_i = A_i @ B_i for each segment i,
    where ``seglen_A`` defines the number of rows of A in segment i.

    For CUDA inputs the call routes through the
    `fairchem_cpp::segment_mm` C++ op via a torch.autograd.Function
    (zero dynamo overhead in eager; clean trace under torch.compile via
    register_fake on the op). For CPU inputs it falls back to a Python
    loop of plain matmuls.
    """
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(B.shape[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return torch.cat(C)
    # If autocasting, make sure weights are same type
    B = B.to(A.dtype)
    return _SegmentMM.apply(A, B, seglen_A)
