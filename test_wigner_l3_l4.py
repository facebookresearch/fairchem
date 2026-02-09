#!/usr/bin/env python3
"""
Test that the derived l=3 and l=4 quaternion-to-Wigner D formulas
match the existing matrix exponential reference implementation.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import math

# Import the reference implementation
from fairchem.core.models.uma.common.wigner_d_axis_angle import (
    get_so3_generators,
    quaternion_to_axis_angle,
    quaternion_to_wigner_d_l2,
)


def quaternion_to_wigner_d_l3(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion directly to 7x7 l=3 Wigner D matrix.

    Uses degree-6 polynomial formulas in quaternion components (w,x,y,z).
    The matrix is built using torch.stack to minimize autograd graph nodes,
    which significantly speeds up the backward pass compared to indexed
    assignment (D[:, i, j] = ...).

    Output matches the Euler-aligned basis of the axis-angle implementation.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute square root constants
    sqrt6 = 2.4494897427831779  # math.sqrt(6)
    sqrt10 = 3.1622776601683795  # math.sqrt(10)
    sqrt15 = 3.8729833462074170  # math.sqrt(15)

    # Powers of w
    w2 = w*w
    w3 = w2*w
    w4 = w3*w
    w5 = w4*w
    w6 = w5*w

    # Powers of x
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    x5 = x4*x
    x6 = x5*x

    # Powers of y
    y2 = y*y
    y3 = y2*y
    y4 = y3*y
    y5 = y4*y
    y6 = y5*y

    # Powers of z
    z2 = z*z
    z3 = z2*z
    z4 = z3*z
    z5 = z4*z
    z6 = z5*z

    # Build all 49 elements
    # Row 0
    d00 = -z6 - y6 + 15*x2*z4 - 15*x4*z2 + x6 + 15*w2*y4 - 15*w4*y2 + w6
    d01 = -sqrt6*y*z5 + sqrt6*y5*z + 10*sqrt6*x2*y*z3 - 5*sqrt6*x4*y*z - 5*sqrt6*w*x*z4 + 5*sqrt6*w*x*y4 + 10*sqrt6*w*x3*z2 - sqrt6*w*x5 - 10*sqrt6*w2*y3*z - 10*sqrt6*w3*x*y2 + 5*sqrt6*w4*y*z + sqrt6*w5*x
    d02 = -sqrt15*y2*z4 - sqrt15*y4*z2 + 6*sqrt15*x2*y2*z2 + sqrt15*x2*y4 - sqrt15*x4*y2 - 8*sqrt15*w*x*y*z3 - 8*sqrt15*w*x*y3*z + 8*sqrt15*w*x3*y*z + sqrt15*w2*z4 + 6*sqrt15*w2*y2*z2 - 6*sqrt15*w2*x2*z2 - 6*sqrt15*w2*x2*y2 + sqrt15*w2*x4 + 8*sqrt15*w3*x*y*z - sqrt15*w4*z2 + sqrt15*w4*x2
    d03 = 6*sqrt10*x*y3*z2 - 2*sqrt10*x3*y3 - 6*sqrt10*w*y2*z3 + 18*sqrt10*w*x2*y2*z - 18*sqrt10*w2*x*y*z2 + 6*sqrt10*w2*x3*y + 2*sqrt10*w3*z3 - 6*sqrt10*w3*x2*z
    d04 = 4*sqrt15*x*y2*z3 - 2*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z - 2*sqrt15*w*y*z4 + 4*sqrt15*w*y3*z2 + 12*sqrt15*w*x2*y*z2 - 4*sqrt15*w*x2*y3 - 2*sqrt15*w*x4*y - 4*sqrt15*w2*x*z3 + 12*sqrt15*w2*x*y2*z + 4*sqrt15*w2*x3*z - 4*sqrt15*w3*y*z2 + 4*sqrt15*w3*x2*y - 2*sqrt15*w4*x*z
    d05 = 5*sqrt6*x*y*z4 + sqrt6*x*y5 - 10*sqrt6*x3*y*z2 + sqrt6*x5*y - sqrt6*w*z5 - 5*sqrt6*w*y4*z + 10*sqrt6*w*x2*z3 - 5*sqrt6*w*x4*z - 10*sqrt6*w2*x*y3 + 10*sqrt6*w3*y2*z + 5*sqrt6*w4*x*y - sqrt6*w5*z
    d06 = 6*x*z5 - 20*x3*z3 + 6*x5*z + 6*w*y5 - 20*w3*y3 + 6*w5*y

    # Row 1
    d10 = -sqrt6*y*z5 + sqrt6*y5*z + 10*sqrt6*x2*y*z3 - 5*sqrt6*x4*y*z + 5*sqrt6*w*x*z4 - 5*sqrt6*w*x*y4 - 10*sqrt6*w*x3*z2 + sqrt6*w*x5 - 10*sqrt6*w2*y3*z + 10*sqrt6*w3*x*y2 + 5*sqrt6*w4*y*z - sqrt6*w5*x
    d11 = z6 - 5*y2*z4 - 5*y4*z2 + y6 - 5*x2*z4 + 30*x2*y2*z2 - 5*x2*y4 - 5*x4*z2 - 5*x4*y2 + x6 - 5*w2*z4 + 30*w2*y2*z2 - 5*w2*y4 + 30*w2*x2*z2 + 30*w2*x2*y2 - 5*w2*x4 - 5*w4*z2 - 5*w4*y2 - 5*w4*x2 + w6
    d12 = sqrt10*y*z5 - sqrt10*y5*z - 2*sqrt10*x2*y*z3 + 8*sqrt10*x2*y3*z - 3*sqrt10*x4*y*z + 3*sqrt10*w*x*z4 - 3*sqrt10*w*x*y4 + 2*sqrt10*w*x3*z2 + 8*sqrt10*w*x3*y2 - sqrt10*w*x5 - 8*sqrt10*w2*y*z3 + 2*sqrt10*w2*y3*z - 8*sqrt10*w3*x*z2 - 2*sqrt10*w3*x*y2 + 3*sqrt10*w4*y*z + sqrt10*w5*x
    d13 = -4*sqrt15*x*y2*z3 + 4*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z + 4*sqrt15*w*y*z4 - 4*sqrt15*w*y3*z2 + 4*sqrt15*w*x2*y3 - 4*sqrt15*w*x4*y + 4*sqrt15*w2*x*z3 + 4*sqrt15*w2*x3*z - 4*sqrt15*w3*y*z2 + 4*sqrt15*w3*x2*y - 4*sqrt15*w4*x*z
    d14 = -3*sqrt10*x*y*z4 + 8*sqrt10*x*y3*z2 - sqrt10*x*y5 - 2*sqrt10*x3*y*z2 + sqrt10*x5*y + sqrt10*w*z5 - 8*sqrt10*w*y2*z3 + 3*sqrt10*w*y4*z - 2*sqrt10*w*x2*z3 - 3*sqrt10*w*x4*z + 2*sqrt10*w2*x*y3 - 8*sqrt10*w2*x3*y + 2*sqrt10*w3*y2*z + 8*sqrt10*w3*x2*z + 3*sqrt10*w4*x*y - sqrt10*w5*z
    d15 = -4*x*z5 + 20*x*y2*z3 - 20*x3*y2*z + 4*x5*z + 20*w*y3*z2 - 4*w*y5 + 20*w*x2*y3 + 20*w2*x*z3 - 20*w2*x3*z - 20*w3*y*z2 - 20*w3*x2*y + 4*w5*y
    d16 = 5*sqrt6*x*y*z4 - sqrt6*x*y5 - 10*sqrt6*x3*y*z2 + sqrt6*x5*y + sqrt6*w*z5 - 5*sqrt6*w*y4*z - 10*sqrt6*w*x2*z3 + 5*sqrt6*w*x4*z + 10*sqrt6*w2*x*y3 + 10*sqrt6*w3*y2*z - 5*sqrt6*w4*x*y - sqrt6*w5*z

    # Row 2
    d20 = -sqrt15*y2*z4 - sqrt15*y4*z2 + 6*sqrt15*x2*y2*z2 + sqrt15*x2*y4 - sqrt15*x4*y2 + 8*sqrt15*w*x*y*z3 + 8*sqrt15*w*x*y3*z - 8*sqrt15*w*x3*y*z + sqrt15*w2*z4 + 6*sqrt15*w2*y2*z2 - 6*sqrt15*w2*x2*z2 - 6*sqrt15*w2*x2*y2 + sqrt15*w2*x4 - 8*sqrt15*w3*x*y*z - sqrt15*w4*z2 + sqrt15*w4*x2
    d21 = sqrt10*y*z5 - sqrt10*y5*z - 2*sqrt10*x2*y*z3 + 8*sqrt10*x2*y3*z - 3*sqrt10*x4*y*z - 3*sqrt10*w*x*z4 + 3*sqrt10*w*x*y4 - 2*sqrt10*w*x3*z2 - 8*sqrt10*w*x3*y2 + sqrt10*w*x5 - 8*sqrt10*w2*y*z3 + 2*sqrt10*w2*y3*z + 8*sqrt10*w3*x*z2 + 2*sqrt10*w3*x*y2 + 3*sqrt10*w4*y*z - sqrt10*w5*x
    d22 = -z6 + 2*y2*z4 + 2*y4*z2 - y6 - x2*z4 - 12*x2*y2*z2 + 14*x2*y4 + x4*z2 - 14*x4*y2 + x6 + 14*w2*z4 - 12*w2*y2*z2 - w2*y4 + 12*w2*x2*z2 + 12*w2*x2*y2 - 2*w2*x4 - 14*w4*z2 + w4*y2 - 2*w4*x2 + w6
    d23 = 2*sqrt6*x*y*z4 - 6*sqrt6*x*y3*z2 + 2*sqrt6*x*y5 + 4*sqrt6*x3*y*z2 - 6*sqrt6*x3*y3 + 2*sqrt6*x5*y - 2*sqrt6*w*z5 + 6*sqrt6*w*y2*z3 - 2*sqrt6*w*y4*z - 4*sqrt6*w*x2*z3 + 6*sqrt6*w*x2*y2*z - 2*sqrt6*w*x4*z - 6*sqrt6*w2*x*y*z2 + 4*sqrt6*w2*x*y3 - 6*sqrt6*w2*x3*y + 6*sqrt6*w3*z3 - 4*sqrt6*w3*y2*z + 6*sqrt6*w3*x2*z + 2*sqrt6*w4*x*y - 2*sqrt6*w5*z
    d24 = 2*x*z5 - 16*x*y2*z3 + 12*x*y4*z + 4*x3*z3 - 16*x3*y2*z + 2*x5*z + 12*w*y*z4 - 16*w*y3*z2 + 2*w*y5 + 24*w*x2*y*z2 - 16*w*x2*y3 + 12*w*x4*y - 16*w2*x*z3 + 24*w2*x*y2*z - 16*w2*x3*z - 16*w3*y*z2 + 4*w3*y3 - 16*w3*x2*y + 12*w4*x*z + 2*w5*y
    d25 = -3*sqrt10*x*y*z4 + 4*sqrt10*x*y3*z2 + sqrt10*x*y5 - 2*sqrt10*x3*y*z2 - 4*sqrt10*x3*y3 + sqrt10*x5*y - sqrt10*w*z5 - 4*sqrt10*w*y2*z3 + 3*sqrt10*w*y4*z + 2*sqrt10*w*x2*z3 - 12*sqrt10*w*x2*y2*z + 3*sqrt10*w*x4*z + 12*sqrt10*w2*x*y*z2 - 2*sqrt10*w2*x*y3 + 4*sqrt10*w2*x3*y + 4*sqrt10*w3*z3 + 2*sqrt10*w3*y2*z - 4*sqrt10*w3*x2*z - 3*sqrt10*w4*x*y - sqrt10*w5*z
    d26 = 4*sqrt15*x*y2*z3 + 2*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z + 2*sqrt15*w*y*z4 + 4*sqrt15*w*y3*z2 - 12*sqrt15*w*x2*y*z2 - 4*sqrt15*w*x2*y3 + 2*sqrt15*w*x4*y - 4*sqrt15*w2*x*z3 - 12*sqrt15*w2*x*y2*z + 4*sqrt15*w2*x3*z - 4*sqrt15*w3*y*z2 + 4*sqrt15*w3*x2*y + 2*sqrt15*w4*x*z

    # Row 3
    d30 = 6*sqrt10*x*y3*z2 - 2*sqrt10*x3*y3 + 6*sqrt10*w*y2*z3 - 18*sqrt10*w*x2*y2*z - 18*sqrt10*w2*x*y*z2 + 6*sqrt10*w2*x3*y - 2*sqrt10*w3*z3 + 6*sqrt10*w3*x2*z
    d31 = -4*sqrt15*x*y2*z3 + 4*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z - 4*sqrt15*w*y*z4 + 4*sqrt15*w*y3*z2 - 4*sqrt15*w*x2*y3 + 4*sqrt15*w*x4*y + 4*sqrt15*w2*x*z3 + 4*sqrt15*w2*x3*z + 4*sqrt15*w3*y*z2 - 4*sqrt15*w3*x2*y - 4*sqrt15*w4*x*z
    d32 = 2*sqrt6*x*y*z4 - 6*sqrt6*x*y3*z2 + 2*sqrt6*x*y5 + 4*sqrt6*x3*y*z2 - 6*sqrt6*x3*y3 + 2*sqrt6*x5*y + 2*sqrt6*w*z5 - 6*sqrt6*w*y2*z3 + 2*sqrt6*w*y4*z + 4*sqrt6*w*x2*z3 - 6*sqrt6*w*x2*y2*z + 2*sqrt6*w*x4*z - 6*sqrt6*w2*x*y*z2 + 4*sqrt6*w2*x*y3 - 6*sqrt6*w2*x3*y - 6*sqrt6*w3*z3 + 4*sqrt6*w3*y2*z - 6*sqrt6*w3*x2*z + 2*sqrt6*w4*x*y + 2*sqrt6*w5*z
    d33 = -z6 + 9*y2*z4 - 9*y4*z2 + y6 - 3*x2*z4 + 18*x2*y2*z2 - 9*x2*y4 - 3*x4*z2 + 9*x4*y2 - x6 + 9*w2*z4 - 18*w2*y2*z2 + 3*w2*y4 + 18*w2*x2*z2 - 18*w2*x2*y2 + 9*w2*x4 - 9*w4*z2 + 3*w4*y2 - 9*w4*x2 + w6
    d34 = 2*sqrt6*y*z5 - 6*sqrt6*y3*z3 + 2*sqrt6*y5*z + 4*sqrt6*x2*y*z3 - 6*sqrt6*x2*y3*z + 2*sqrt6*x4*y*z - 2*sqrt6*w*x*z4 + 6*sqrt6*w*x*y2*z2 - 2*sqrt6*w*x*y4 - 4*sqrt6*w*x3*z2 + 6*sqrt6*w*x3*y2 - 2*sqrt6*w*x5 - 6*sqrt6*w2*y*z3 + 4*sqrt6*w2*y3*z - 6*sqrt6*w2*x2*y*z + 6*sqrt6*w3*x*z2 - 4*sqrt6*w3*x*y2 + 6*sqrt6*w3*x3 + 2*sqrt6*w4*y*z - 2*sqrt6*w5*x
    d35 = -2*sqrt15*y2*z4 + 2*sqrt15*y4*z2 - 2*sqrt15*x2*y4 + 2*sqrt15*x4*y2 + 8*sqrt15*w*x*y*z3 - 8*sqrt15*w*x*y3*z + 8*sqrt15*w*x3*y*z + 2*sqrt15*w2*z4 - 2*sqrt15*w2*x4 - 8*sqrt15*w3*x*y*z - 2*sqrt15*w4*z2 + 2*sqrt15*w4*x2
    d36 = 2*sqrt10*y3*z3 - 6*sqrt10*x2*y3*z - 18*sqrt10*w*x*y2*z2 + 6*sqrt10*w*x3*y2 - 6*sqrt10*w2*y*z3 + 18*sqrt10*w2*x2*y*z + 6*sqrt10*w3*x*z2 - 2*sqrt10*w3*x3

    # Row 4
    d40 = 4*sqrt15*x*y2*z3 - 2*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z + 2*sqrt15*w*y*z4 - 4*sqrt15*w*y3*z2 - 12*sqrt15*w*x2*y*z2 + 4*sqrt15*w*x2*y3 + 2*sqrt15*w*x4*y - 4*sqrt15*w2*x*z3 + 12*sqrt15*w2*x*y2*z + 4*sqrt15*w2*x3*z + 4*sqrt15*w3*y*z2 - 4*sqrt15*w3*x2*y - 2*sqrt15*w4*x*z
    d41 = -3*sqrt10*x*y*z4 + 8*sqrt10*x*y3*z2 - sqrt10*x*y5 - 2*sqrt10*x3*y*z2 + sqrt10*x5*y - sqrt10*w*z5 + 8*sqrt10*w*y2*z3 - 3*sqrt10*w*y4*z + 2*sqrt10*w*x2*z3 + 3*sqrt10*w*x4*z + 2*sqrt10*w2*x*y3 - 8*sqrt10*w2*x3*y - 2*sqrt10*w3*y2*z - 8*sqrt10*w3*x2*z + 3*sqrt10*w4*x*y + sqrt10*w5*z
    d42 = 2*x*z5 - 16*x*y2*z3 + 12*x*y4*z + 4*x3*z3 - 16*x3*y2*z + 2*x5*z - 12*w*y*z4 + 16*w*y3*z2 - 2*w*y5 - 24*w*x2*y*z2 + 16*w*x2*y3 - 12*w*x4*y - 16*w2*x*z3 + 24*w2*x*y2*z - 16*w2*x3*z + 16*w3*y*z2 - 4*w3*y3 + 16*w3*x2*y + 12*w4*x*z - 2*w5*y
    d43 = 2*sqrt6*y*z5 - 6*sqrt6*y3*z3 + 2*sqrt6*y5*z + 4*sqrt6*x2*y*z3 - 6*sqrt6*x2*y3*z + 2*sqrt6*x4*y*z + 2*sqrt6*w*x*z4 - 6*sqrt6*w*x*y2*z2 + 2*sqrt6*w*x*y4 + 4*sqrt6*w*x3*z2 - 6*sqrt6*w*x3*y2 + 2*sqrt6*w*x5 - 6*sqrt6*w2*y*z3 + 4*sqrt6*w2*y3*z - 6*sqrt6*w2*x2*y*z - 6*sqrt6*w3*x*z2 + 4*sqrt6*w3*x*y2 - 6*sqrt6*w3*x3 + 2*sqrt6*w4*y*z + 2*sqrt6*w5*x
    d44 = z6 - 14*y2*z4 + 14*y4*z2 - y6 + x2*z4 - 12*x2*y2*z2 + 2*x2*y4 - x4*z2 + 2*x4*y2 - x6 - 2*w2*z4 + 12*w2*y2*z2 - w2*y4 + 12*w2*x2*z2 - 12*w2*x2*y2 + 14*w2*x4 - 2*w4*z2 + w4*y2 - 14*w4*x2 + w6
    d45 = -sqrt10*y*z5 + 4*sqrt10*y3*z3 - sqrt10*y5*z + 2*sqrt10*x2*y*z3 - 4*sqrt10*x2*y3*z + 3*sqrt10*x4*y*z + 3*sqrt10*w*x*z4 - 12*sqrt10*w*x*y2*z2 + 3*sqrt10*w*x*y4 + 2*sqrt10*w*x3*z2 - 4*sqrt10*w*x3*y2 - sqrt10*w*x5 - 4*sqrt10*w2*y*z3 + 2*sqrt10*w2*y3*z - 12*sqrt10*w2*x2*y*z - 4*sqrt10*w3*x*z2 + 2*sqrt10*w3*x*y2 + 4*sqrt10*w3*x3 + 3*sqrt10*w4*y*z - sqrt10*w5*x
    d46 = sqrt15*y2*z4 - sqrt15*y4*z2 - 6*sqrt15*x2*y2*z2 + sqrt15*x2*y4 + sqrt15*x4*y2 - 8*sqrt15*w*x*y*z3 + 8*sqrt15*w*x*y3*z + 8*sqrt15*w*x3*y*z - sqrt15*w2*z4 + 6*sqrt15*w2*y2*z2 + 6*sqrt15*w2*x2*z2 - 6*sqrt15*w2*x2*y2 - sqrt15*w2*x4 - 8*sqrt15*w3*x*y*z - sqrt15*w4*z2 + sqrt15*w4*x2

    # Row 5
    d50 = 5*sqrt6*x*y*z4 + sqrt6*x*y5 - 10*sqrt6*x3*y*z2 + sqrt6*x5*y + sqrt6*w*z5 + 5*sqrt6*w*y4*z - 10*sqrt6*w*x2*z3 + 5*sqrt6*w*x4*z - 10*sqrt6*w2*x*y3 - 10*sqrt6*w3*y2*z + 5*sqrt6*w4*x*y + sqrt6*w5*z
    d51 = -4*x*z5 + 20*x*y2*z3 - 20*x3*y2*z + 4*x5*z - 20*w*y3*z2 + 4*w*y5 - 20*w*x2*y3 + 20*w2*x*z3 - 20*w2*x3*z + 20*w3*y*z2 + 20*w3*x2*y - 4*w5*y
    d52 = -3*sqrt10*x*y*z4 + 4*sqrt10*x*y3*z2 + sqrt10*x*y5 - 2*sqrt10*x3*y*z2 - 4*sqrt10*x3*y3 + sqrt10*x5*y + sqrt10*w*z5 + 4*sqrt10*w*y2*z3 - 3*sqrt10*w*y4*z - 2*sqrt10*w*x2*z3 + 12*sqrt10*w*x2*y2*z - 3*sqrt10*w*x4*z + 12*sqrt10*w2*x*y*z2 - 2*sqrt10*w2*x*y3 + 4*sqrt10*w2*x3*y - 4*sqrt10*w3*z3 - 2*sqrt10*w3*y2*z + 4*sqrt10*w3*x2*z - 3*sqrt10*w4*x*y + sqrt10*w5*z
    d53 = -2*sqrt15*y2*z4 + 2*sqrt15*y4*z2 - 2*sqrt15*x2*y4 + 2*sqrt15*x4*y2 - 8*sqrt15*w*x*y*z3 + 8*sqrt15*w*x*y3*z - 8*sqrt15*w*x3*y*z + 2*sqrt15*w2*z4 - 2*sqrt15*w2*x4 + 8*sqrt15*w3*x*y*z - 2*sqrt15*w4*z2 + 2*sqrt15*w4*x2
    d54 = -sqrt10*y*z5 + 4*sqrt10*y3*z3 - sqrt10*y5*z + 2*sqrt10*x2*y*z3 - 4*sqrt10*x2*y3*z + 3*sqrt10*x4*y*z - 3*sqrt10*w*x*z4 + 12*sqrt10*w*x*y2*z2 - 3*sqrt10*w*x*y4 - 2*sqrt10*w*x3*z2 + 4*sqrt10*w*x3*y2 + sqrt10*w*x5 - 4*sqrt10*w2*y*z3 + 2*sqrt10*w2*y3*z - 12*sqrt10*w2*x2*y*z + 4*sqrt10*w3*x*z2 - 2*sqrt10*w3*x*y2 - 4*sqrt10*w3*x3 + 3*sqrt10*w4*y*z + sqrt10*w5*x
    d55 = -z6 + 5*y2*z4 - 5*y4*z2 + y6 + 5*x2*z4 - 30*x2*y2*z2 - 5*x2*y4 + 5*x4*z2 + 5*x4*y2 - x6 + 5*w2*z4 + 30*w2*y2*z2 - 5*w2*y4 - 30*w2*x2*z2 + 30*w2*x2*y2 + 5*w2*x4 - 5*w4*z2 - 5*w4*y2 - 5*w4*x2 + w6
    d56 = sqrt6*y*z5 + sqrt6*y5*z - 10*sqrt6*x2*y*z3 + 5*sqrt6*x4*y*z - 5*sqrt6*w*x*z4 - 5*sqrt6*w*x*y4 + 10*sqrt6*w*x3*z2 - sqrt6*w*x5 - 10*sqrt6*w2*y3*z + 10*sqrt6*w3*x*y2 + 5*sqrt6*w4*y*z - sqrt6*w5*x

    # Row 6
    d60 = 6*x*z5 - 20*x3*z3 + 6*x5*z - 6*w*y5 + 20*w3*y3 - 6*w5*y
    d61 = 5*sqrt6*x*y*z4 - sqrt6*x*y5 - 10*sqrt6*x3*y*z2 + sqrt6*x5*y - sqrt6*w*z5 + 5*sqrt6*w*y4*z + 10*sqrt6*w*x2*z3 - 5*sqrt6*w*x4*z + 10*sqrt6*w2*x*y3 - 10*sqrt6*w3*y2*z - 5*sqrt6*w4*x*y + sqrt6*w5*z
    d62 = 4*sqrt15*x*y2*z3 + 2*sqrt15*x*y4*z - 4*sqrt15*x3*y2*z - 2*sqrt15*w*y*z4 - 4*sqrt15*w*y3*z2 + 12*sqrt15*w*x2*y*z2 + 4*sqrt15*w*x2*y3 - 2*sqrt15*w*x4*y - 4*sqrt15*w2*x*z3 - 12*sqrt15*w2*x*y2*z + 4*sqrt15*w2*x3*z + 4*sqrt15*w3*y*z2 - 4*sqrt15*w3*x2*y + 2*sqrt15*w4*x*z
    d63 = 2*sqrt10*y3*z3 - 6*sqrt10*x2*y3*z + 18*sqrt10*w*x*y2*z2 - 6*sqrt10*w*x3*y2 - 6*sqrt10*w2*y*z3 + 18*sqrt10*w2*x2*y*z - 6*sqrt10*w3*x*z2 + 2*sqrt10*w3*x3
    d64 = sqrt15*y2*z4 - sqrt15*y4*z2 - 6*sqrt15*x2*y2*z2 + sqrt15*x2*y4 + sqrt15*x4*y2 + 8*sqrt15*w*x*y*z3 - 8*sqrt15*w*x*y3*z - 8*sqrt15*w*x3*y*z - sqrt15*w2*z4 + 6*sqrt15*w2*y2*z2 + 6*sqrt15*w2*x2*z2 - 6*sqrt15*w2*x2*y2 - sqrt15*w2*x4 + 8*sqrt15*w3*x*y*z - sqrt15*w4*z2 + sqrt15*w4*x2
    d65 = sqrt6*y*z5 + sqrt6*y5*z - 10*sqrt6*x2*y*z3 + 5*sqrt6*x4*y*z + 5*sqrt6*w*x*z4 + 5*sqrt6*w*x*y4 - 10*sqrt6*w*x3*z2 + sqrt6*w*x5 - 10*sqrt6*w2*y3*z - 10*sqrt6*w3*x*y2 + 5*sqrt6*w4*y*z + sqrt6*w5*x
    d66 = z6 - y6 - 15*x2*z4 + 15*x4*z2 - x6 + 15*w2*y4 - 15*w4*y2 + w6

    # Stack into matrix
    D = torch.stack([
        torch.stack([d00, d01, d02, d03, d04, d05, d06], dim=-1),
        torch.stack([d10, d11, d12, d13, d14, d15, d16], dim=-1),
        torch.stack([d20, d21, d22, d23, d24, d25, d26], dim=-1),
        torch.stack([d30, d31, d32, d33, d34, d35, d36], dim=-1),
        torch.stack([d40, d41, d42, d43, d44, d45, d46], dim=-1),
        torch.stack([d50, d51, d52, d53, d54, d55, d56], dim=-1),
        torch.stack([d60, d61, d62, d63, d64, d65, d66], dim=-1),
    ], dim=-2)

    return D


def compute_reference_wigner_d(q: torch.Tensor, ell: int) -> torch.Tensor:
    """
    Compute Wigner D matrix using the reference matrix exponential method.

    This uses the SO(3) generators with Euler-matching transformation
    and torch.linalg.matrix_exp.
    """
    size = 2 * ell + 1
    n_samples = q.shape[0]

    gens = get_so3_generators(ell, q.dtype, q.device)
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)

    D_ref = torch.zeros(n_samples, size, size, dtype=q.dtype, device=q.device)
    for i in range(n_samples):
        n = axis[i]
        K = n[0] * K_x + n[1] * K_y + n[2] * K_z
        D_ref[i] = torch.linalg.matrix_exp(angle[i] * K)

    return D_ref


def test_l2_matches_reference():
    """Verify l=2 formula matches reference (sanity check)."""
    print("Testing l=2 formula against reference...")

    torch.manual_seed(42)
    n_samples = 100
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    D_formula = quaternion_to_wigner_d_l2(q)
    D_ref = compute_reference_wigner_d(q, ell=2)

    max_err = (D_formula - D_ref).abs().max().item()
    print(f"  Max error: {max_err:.2e}")

    assert max_err < 1e-10, f"l=2 formula error too large: {max_err}"
    print("  PASSED")
    return True


def test_l3_matches_reference():
    """Verify l=3 formula matches reference."""
    print("Testing l=3 formula against reference...")

    torch.manual_seed(42)
    n_samples = 100
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    D_formula = quaternion_to_wigner_d_l3(q)
    D_ref = compute_reference_wigner_d(q, ell=3)

    max_err = (D_formula - D_ref).abs().max().item()
    print(f"  Max error: {max_err:.2e}")

    assert max_err < 1e-10, f"l=3 formula error too large: {max_err}"
    print("  PASSED")
    return True


def test_l3_orthogonality():
    """Verify l=3 Wigner D matrices are orthogonal."""
    print("Testing l=3 orthogonality...")

    torch.manual_seed(123)
    n_samples = 50
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    D = quaternion_to_wigner_d_l3(q)
    I = torch.eye(7, dtype=torch.float64)

    for i in range(n_samples):
        product = D[i] @ D[i].T
        err = (product - I).abs().max().item()
        assert err < 1e-10, f"Sample {i}: orthogonality error = {err}"

    print("  PASSED")
    return True


def test_l3_special_quaternions():
    """Test l=3 formula on special quaternions (identity, axis rotations)."""
    print("Testing l=3 on special quaternions...")

    test_cases = [
        ("identity", torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)),
        ("90° about X", torch.tensor([[0.7071067811865476, 0.7071067811865476, 0.0, 0.0]], dtype=torch.float64)),
        ("90° about Y", torch.tensor([[0.7071067811865476, 0.0, 0.7071067811865476, 0.0]], dtype=torch.float64)),
        ("90° about Z", torch.tensor([[0.7071067811865476, 0.0, 0.0, 0.7071067811865476]], dtype=torch.float64)),
        ("180° about X", torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)),
        ("180° about Y", torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)),
        ("180° about Z", torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)),
    ]

    for name, q in test_cases:
        D_formula = quaternion_to_wigner_d_l3(q)
        D_ref = compute_reference_wigner_d(q, ell=3)
        max_err = (D_formula - D_ref).abs().max().item()
        assert max_err < 1e-10, f"{name}: error = {max_err}"
        print(f"  {name}: error = {max_err:.2e}")

    print("  PASSED")
    return True


def test_l3_gradients():
    """Test that gradients flow through l=3 formula."""
    print("Testing l=3 gradient flow...")

    torch.manual_seed(456)
    q = torch.randn(10, 4, dtype=torch.float64, requires_grad=True)
    q_normalized = q / q.norm(dim=1, keepdim=True)

    D = quaternion_to_wigner_d_l3(q_normalized)
    loss = D.sum()
    loss.backward()

    assert q.grad is not None, "No gradient computed"
    assert not torch.isnan(q.grad).any(), "NaN in gradients"
    assert not torch.isinf(q.grad).any(), "Inf in gradients"

    print(f"  Gradient norm: {q.grad.norm().item():.4f}")
    print("  PASSED")
    return True


# Import l=4 from generated file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from wigner_l4_clean import quaternion_to_wigner_d_l4


def test_l4_matches_reference():
    """Verify l=4 formula matches reference."""
    print("Testing l=4 formula against reference...")

    torch.manual_seed(42)
    n_samples = 100
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    D_formula = quaternion_to_wigner_d_l4(q)
    D_ref = compute_reference_wigner_d(q, ell=4)

    max_err = (D_formula - D_ref).abs().max().item()
    print(f"  Max error: {max_err:.2e}")

    assert max_err < 1e-10, f"l=4 formula error too large: {max_err}"
    print("  PASSED")
    return True


def test_l4_orthogonality():
    """Verify l=4 Wigner D matrices are orthogonal."""
    print("Testing l=4 orthogonality...")

    torch.manual_seed(123)
    n_samples = 50
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    D = quaternion_to_wigner_d_l4(q)
    I = torch.eye(9, dtype=torch.float64)

    for i in range(n_samples):
        product = D[i] @ D[i].T
        err = (product - I).abs().max().item()
        assert err < 1e-10, f"Sample {i}: orthogonality error = {err}"

    print("  PASSED")
    return True


def test_l4_special_quaternions():
    """Test l=4 formula on special quaternions (identity, axis rotations)."""
    print("Testing l=4 on special quaternions...")

    test_cases = [
        ("identity", torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)),
        ("90° about X", torch.tensor([[0.7071067811865476, 0.7071067811865476, 0.0, 0.0]], dtype=torch.float64)),
        ("90° about Y", torch.tensor([[0.7071067811865476, 0.0, 0.7071067811865476, 0.0]], dtype=torch.float64)),
        ("90° about Z", torch.tensor([[0.7071067811865476, 0.0, 0.0, 0.7071067811865476]], dtype=torch.float64)),
        ("180° about X", torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)),
        ("180° about Y", torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)),
        ("180° about Z", torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)),
    ]

    for name, q in test_cases:
        D_formula = quaternion_to_wigner_d_l4(q)
        D_ref = compute_reference_wigner_d(q, ell=4)
        max_err = (D_formula - D_ref).abs().max().item()
        assert max_err < 1e-10, f"{name}: error = {max_err}"
        print(f"  {name}: error = {max_err:.2e}")

    print("  PASSED")
    return True


def test_l4_gradients():
    """Test that gradients flow through l=4 formula."""
    print("Testing l=4 gradient flow...")

    torch.manual_seed(456)
    q = torch.randn(10, 4, dtype=torch.float64, requires_grad=True)
    q_normalized = q / q.norm(dim=1, keepdim=True)

    D = quaternion_to_wigner_d_l4(q_normalized)
    loss = D.sum()
    loss.backward()

    assert q.grad is not None, "No gradient computed"
    assert not torch.isnan(q.grad).any(), "NaN in gradients"
    assert not torch.isinf(q.grad).any(), "Inf in gradients"

    print(f"  Gradient norm: {q.grad.norm().item():.4f}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Quaternion to Wigner D Formulas")
    print("=" * 60)
    print()

    all_passed = True

    # l=2 sanity check
    all_passed &= test_l2_matches_reference()
    print()

    # l=3 tests
    all_passed &= test_l3_matches_reference()
    print()
    all_passed &= test_l3_orthogonality()
    print()
    all_passed &= test_l3_special_quaternions()
    print()
    all_passed &= test_l3_gradients()
    print()

    # l=4 tests
    all_passed &= test_l4_matches_reference()
    print()
    all_passed &= test_l4_orthogonality()
    print()
    all_passed &= test_l4_special_quaternions()
    print()
    all_passed &= test_l4_gradients()
    print()

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
