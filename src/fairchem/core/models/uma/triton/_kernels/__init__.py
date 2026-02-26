"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Internal Triton kernel implementations for Wigner D-matrix transforms (lmax=2).

NOTE: This is internal API. Import from fairchem.core.models.uma.triton instead.

=============================================================================
WIGNER D-MATRIX STRUCTURE (lmax=2)
=============================================================================

The Wigner D-matrix for SO(3) rotation is block-diagonal by angular momentum L:

    ┌─────┬─────────┬─────────────────┐
    │ L=0 │    0    │        0        │   L=0: 1x1 block (scalar)
    │ 1x1 │         │                 │
    ├─────┼─────────┼─────────────────┤
    │  0  │   L=1   │        0        │   L=1: 3x3 block (m=-1,0,+1)
    │     │   3x3   │                 │
    ├─────┼─────────┼─────────────────┤
    │  0  │    0    │       L=2       │   L=2: 5x5 block (m=-2,-1,0,+1,+2)
    │     │         │       5x5       │
    └─────┴─────────┴─────────────────┘

Total: 9x9 matrix but only 1 + 9 + 25 = 35 non-zero entries.
Storage: W[E, 9, 9] flattened to W[E, 81]

=============================================================================
L-MAJOR vs M-MAJOR ORDERING (lmax=2)
=============================================================================

L-major (sorted by L, then m):
    idx:  0      1      2      3      4      5      6      7      8
    L:    0      1      1      1      2      2      2      2      2
    m:    0     -1      0     +1     -2     -1      0     +1     +2

M-major (sorted by |m|, then sign, then L):
    idx:  0      1      2      3      4      5      6      7      8
    m:    0      0      0     +1     +1     -1     -1     +2     -2
    L:    0      1      2      1      2      1      2      2      2

Conversion indices:
    L→M: out_m[i] = in_l[L_TO_M_IDX[i]]   L_TO_M_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    M→L: out_l[i] = in_m[M_TO_L_IDX[i]]   M_TO_L_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]

=============================================================================
KERNEL FILES
=============================================================================

Forward kernels (no suffix):
    - node_to_edge_wigner_l2m.py: node_to_edge_wigner_l2m_kernel
        Gather + Wigner + L→M permutation (main forward op)
    - wigner_m2l.py: wigner_m2l_kernel
        M→L permutation + Wigner (edge-level, no gather)

Backward kernels (_bwd suffix):
    - wigner_m2l_bwd.py: wigner_m2l_bwd_kernel
        M→L permutation + W^T (backward for node_to_edge, no scatter)
    - wigner_l2m_bwd.py: wigner_l2m_bwd_kernel
        W^T + L→M permutation (backward for wigner_m2l)
    - wigner_weight_bwd.py: wigner_weight_bwd_kernel
        dW = dy @ x^T (weight gradient for forces)
"""

from __future__ import annotations
