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
        FusedEdgeGatherWignerL2MPyTorchBwdFunction,
        FusedEdgeGatherWignerL2MRecomputeFunction,
        FusedEdgeGatherWignerL2MTritonBwdFunction,
        FusedEdgeGatherWignerL2MTritonV2BwdFunction,
    )
    from .gate_activation import gate_activation_triton
    from .wigner_ops import m_to_l_then_wigner_lmax2

__all__ = [
    "HAS_TRITON",
    "FusedEdgeGatherWignerL2MTritonBwdFunction",
    "FusedEdgeGatherWignerL2MTritonV2BwdFunction",
    "FusedEdgeGatherWignerL2MPyTorchBwdFunction",
    "FusedEdgeGatherWignerL2MRecomputeFunction",
    "m_to_l_then_wigner_lmax2",
    "gate_activation_triton",
]
