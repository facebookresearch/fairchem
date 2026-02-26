"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from .edge_gather_wigner_bwd import (
        FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
    )
    from .wigner_ops import (
        FusedMToLThenWignerLmax2Function,
    )

    # Export with new names (same underlying classes)
    UMASFastGPUNodeToEdgeWignerPermute = FusedEdgeGatherWignerL2MTritonBwdEmitFunction
    UMASFastGPUPermuteWignerInvEdgeToNode = FusedMToLThenWignerLmax2Function

__all__ = [
    "HAS_TRITON",
    "FusedEdgeGatherWignerL2MTritonBwdEmitFunction",
    "FusedMToLThenWignerLmax2Function",
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]
