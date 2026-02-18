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


from .fused_edge_wigner_bwd import fused_edge_gather_wigner_l2m_lmax2_with_bwd
from .wigner_ops import (
    L_TO_M_GATHER_IDX,
    M_TO_L_GATHER_IDX,
    m_to_l_then_wigner_lmax2,
)

# Public API
triton_gather_rotate_l2m = fused_edge_gather_wigner_l2m_lmax2_with_bwd
triton_rotate_m2l = m_to_l_then_wigner_lmax2

__all__ = [
    "HAS_TRITON",
    "triton_gather_rotate_l2m",
    "triton_rotate_m2l",
    "L_TO_M_GATHER_IDX",
    "M_TO_L_GATHER_IDX",
    "fused_edge_gather_wigner_l2m_lmax2_with_bwd",
]
