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

from .ops import (
    UMASFastGPUNodeToEdgeWignerPermute,
    UMASFastGPUPermuteWignerInvEdgeToNode,
)

__all__ = [
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]
