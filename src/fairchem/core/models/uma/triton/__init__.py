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
    from .node_to_edge_wigner_permute import (
        NodeToEdgeWignerPermuteFunction as UMASFastGPUNodeToEdgeWignerPermute,
    )
    from .permute_wigner_inv_edge_to_node import (
        PermuteWignerInvEdgeToNodeFunction as UMASFastGPUPermuteWignerInvEdgeToNode,
    )

__all__ = [
    "HAS_TRITON",
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]
