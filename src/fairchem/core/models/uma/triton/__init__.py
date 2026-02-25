"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Triton-accelerated operations for UMA model.

Structure:
    - ops.py: Public torch.autograd.Function classes (main API)
    - constants.py: Shared constants (permutation indices, block sizes)
    - _kernels/: Internal Triton JIT kernels (implementation details)
"""

from __future__ import annotations

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from .ops import (
        FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        FusedMToLThenWignerLmax2Function,
        MToLThenWignerLmax2Function,
    )

__all__ = [
    "HAS_TRITON",
    "FusedEdgeGatherWignerL2MTritonBwdEmitFunction",
    "FusedEdgeGatherWignerL2MTritonV2BwdFunction",
    "FusedMToLThenWignerLmax2Function",
    "MToLThenWignerLmax2Function",
]
