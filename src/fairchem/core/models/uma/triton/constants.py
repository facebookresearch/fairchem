"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Constants for Triton kernel operations.
"""

from __future__ import annotations

# Block size for channel vectorization in Triton kernels.
# Must be a power of 2. sphere_channels must be divisible by this.
BLOCK_C = 128

# Permutation indices for L-major to M-major ordering.
# For lmax=2: coefficients ordered as (l=0), (l=1, m=-1,0,1), (l=2, m=-2,-1,0,1,2)
# L-major: [0, 1, 2, 3, 4, 5, 6, 7, 8] = [l0, l1m-1, l1m0, l1m1, l2m-2, l2m-1, l2m0, l2m1, l2m2]
# M-major: [0, 2, 6, 3, 7, 1, 5, 8, 4] = reordered by m value
L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]

# Permutation indices for M-major to L-major ordering (inverse of above).
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
