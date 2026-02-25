"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Shared constants for Triton kernels.
"""

from __future__ import annotations

# =============================================================================
# Constants for lmax=2 L<->M permutation
# =============================================================================

# L-major order: [l=0, l=1 m=-1, l=1 m=0, l=1 m=1, l=2 m=-2, l=2 m=-1, l=2 m=0, l=2 m=1, l=2 m=2]
# M-major order: [m=0 (l=0,1,2), m=1 (l=1,2), m=-1 (l=1,2), m=2 (l=2), m=-2 (l=2)]
#                positions: [0,1,2] [3,4] [5,6] [7] [8]

# L_TO_M_GATHER_IDX[i] = j means: M-major position i gets L-major position j
# When reordering L->M: out_m[i] = inp_l[L_TO_M_GATHER_IDX[i]]
L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]

# M_TO_L_GATHER_IDX[i] = j means: L-major position i gets M-major position j
# When reordering M->L: out_l[i] = inp_m[M_TO_L_GATHER_IDX[i]]
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]

# Channel block size for Triton kernels
BLOCK_C = 128
