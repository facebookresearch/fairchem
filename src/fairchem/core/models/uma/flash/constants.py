"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compile-time constants shared by the umas_flash Triton kernels.

The Wigner-D matrix for lmax=2 is block diagonal: a 1x1 identity for l=0, a
3x3 block for l=1 and a 5x5 block for l=2. Only the 34 entries of the l=1 and
l=2 blocks are ever needed, so the geometry kernel emits them as a flat vector
and every consumer indexes it with the offsets below. Wnm names the (n, m)
entry in the original dense layout.
"""

from __future__ import annotations

import triton.language as tl

# l=1 block (3x3 = 9 entries)
W11 = tl.constexpr(0)
W12 = tl.constexpr(1)
W13 = tl.constexpr(2)
W21 = tl.constexpr(3)
W22 = tl.constexpr(4)
W23 = tl.constexpr(5)
W31 = tl.constexpr(6)
W32 = tl.constexpr(7)
W33 = tl.constexpr(8)

# l=2 block (5x5 = 25 entries)
W44 = tl.constexpr(9)
W45 = tl.constexpr(10)
W46 = tl.constexpr(11)
W47 = tl.constexpr(12)
W48 = tl.constexpr(13)
W54 = tl.constexpr(14)
W55 = tl.constexpr(15)
W56 = tl.constexpr(16)
W57 = tl.constexpr(17)
W58 = tl.constexpr(18)
W64 = tl.constexpr(19)
W65 = tl.constexpr(20)
W66 = tl.constexpr(21)
W67 = tl.constexpr(22)
W68 = tl.constexpr(23)
W74 = tl.constexpr(24)
W75 = tl.constexpr(25)
W76 = tl.constexpr(26)
W77 = tl.constexpr(27)
W78 = tl.constexpr(28)
W84 = tl.constexpr(29)
W85 = tl.constexpr(30)
W86 = tl.constexpr(31)
W87 = tl.constexpr(32)
W88 = tl.constexpr(33)

NUM_WIGNER_ENTRIES = 34

# Multiples of sqrt(3) appearing in the l=2 Wigner polynomials.
S3 = tl.constexpr(3.4641016151377544)  # sqrt(3) * 2
S3X2 = tl.constexpr(6.928203230275509)  # sqrt(3) * 4
S3X3 = tl.constexpr(10.392304845413264)  # sqrt(3) * 6
S3X4 = tl.constexpr(13.856406460551018)  # sqrt(3) * 8

# Channel multiples of the three m-order blocks. The dense [18C], [11C] and
# [9C] layouts carry structural zeros; these are the packed widths that the
# SO(2) convolutions actually multiply.
X_M0_MULT = tl.constexpr(6)
X_M1_MULT = tl.constexpr(8)
X_M2_MULT = tl.constexpr(4)
Y_M0_MULT = tl.constexpr(5)
Y_M1_MULT = tl.constexpr(4)
Y_M2_MULT = tl.constexpr(2)
Z_M0_MULT = tl.constexpr(3)
Z_M1_MULT = tl.constexpr(4)
Z_M2_MULT = tl.constexpr(2)

TRUNC_Y_M0_MULT = tl.constexpr(5)
TRUNC_Z_M0_MULT = tl.constexpr(3)
