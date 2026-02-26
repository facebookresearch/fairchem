"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Backward: Wigner weight gradient (dW = dy @ x^T).

Computes the gradient of the loss w.r.t. the Wigner D-matrix weights.
This is required for force computation when gradients flow through the
rotation matrices (stress/pressure calculations).

The gradient is an outer product reduced over channels:
    dW[i,j] = sum_c dy[i,c] * x[j,c]

Only the non-zero blocks are computed (block-diagonal structure):
    - L=0: 1x1 (1 element)
    - L=1: 3x3 (9 elements)
    - L=2: 5x5 (25 elements)
    Total: 35 non-zero entries out of 81

Data flow:
    dy[E, 9, C], x[E, 9, C] → dW[E, 81] (block-diagonal)
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def wigner_weight_bwd_kernel(
    DY_ptr,
    X_ptr,
    DW_ptr,
    C: tl.constexpr,
):
    """
    Wigner weight gradient: dW[i,j] = sum_c dy[i,c] * x[j,c].

    Args:
        DY_ptr: Upstream gradient [E, 9, C]
        X_ptr: Saved forward input [E, 9, C] (x_l in L-major order)
        DW_ptr: Output Wigner gradient [E, 81] (only non-zero blocks written)
        C: Number of channels (compile-time constant, must be ≤128)

    Grid: (num_edges,)
        - One thread block per edge
        - Each block computes all 35 non-zero dW entries for one edge
    """
    edge_id = tl.program_id(0)

    dy_base = edge_id * 9 * C
    x_base = edge_id * 9 * C
    dw_base = edge_id * 81

    # Channel range (all channels in one block)
    c_range = tl.arange(0, 128)
    c_mask = c_range < C

    # Load all 9 coefficients for dy and x
    dy0 = tl.load(DY_ptr + dy_base + 0 * C + c_range, mask=c_mask, other=0.0)
    dy1 = tl.load(DY_ptr + dy_base + 1 * C + c_range, mask=c_mask, other=0.0)
    dy2 = tl.load(DY_ptr + dy_base + 2 * C + c_range, mask=c_mask, other=0.0)
    dy3 = tl.load(DY_ptr + dy_base + 3 * C + c_range, mask=c_mask, other=0.0)
    dy4 = tl.load(DY_ptr + dy_base + 4 * C + c_range, mask=c_mask, other=0.0)
    dy5 = tl.load(DY_ptr + dy_base + 5 * C + c_range, mask=c_mask, other=0.0)
    dy6 = tl.load(DY_ptr + dy_base + 6 * C + c_range, mask=c_mask, other=0.0)
    dy7 = tl.load(DY_ptr + dy_base + 7 * C + c_range, mask=c_mask, other=0.0)
    dy8 = tl.load(DY_ptr + dy_base + 8 * C + c_range, mask=c_mask, other=0.0)

    x0 = tl.load(X_ptr + x_base + 0 * C + c_range, mask=c_mask, other=0.0)
    x1 = tl.load(X_ptr + x_base + 1 * C + c_range, mask=c_mask, other=0.0)
    x2 = tl.load(X_ptr + x_base + 2 * C + c_range, mask=c_mask, other=0.0)
    x3 = tl.load(X_ptr + x_base + 3 * C + c_range, mask=c_mask, other=0.0)
    x4 = tl.load(X_ptr + x_base + 4 * C + c_range, mask=c_mask, other=0.0)
    x5 = tl.load(X_ptr + x_base + 5 * C + c_range, mask=c_mask, other=0.0)
    x6 = tl.load(X_ptr + x_base + 6 * C + c_range, mask=c_mask, other=0.0)
    x7 = tl.load(X_ptr + x_base + 7 * C + c_range, mask=c_mask, other=0.0)
    x8 = tl.load(X_ptr + x_base + 8 * C + c_range, mask=c_mask, other=0.0)

    # =========================================================================
    # L=0 block (1x1): dW[0,0] = sum_c dy[0,c] * x[0,c]
    # =========================================================================
    dw_00 = tl.sum(dy0 * x0)
    tl.store(DW_ptr + dw_base + 0, dw_00)

    # =========================================================================
    # L=1 block (3x3): dW[i,j] = sum_c dy[i,c] * x[j,c] for i,j in {1,2,3}
    # =========================================================================
    dw_11 = tl.sum(dy1 * x1)
    dw_12 = tl.sum(dy1 * x2)
    dw_13 = tl.sum(dy1 * x3)
    dw_21 = tl.sum(dy2 * x1)
    dw_22 = tl.sum(dy2 * x2)
    dw_23 = tl.sum(dy2 * x3)
    dw_31 = tl.sum(dy3 * x1)
    dw_32 = tl.sum(dy3 * x2)
    dw_33 = tl.sum(dy3 * x3)

    tl.store(DW_ptr + dw_base + 1 * 9 + 1, dw_11)
    tl.store(DW_ptr + dw_base + 1 * 9 + 2, dw_12)
    tl.store(DW_ptr + dw_base + 1 * 9 + 3, dw_13)
    tl.store(DW_ptr + dw_base + 2 * 9 + 1, dw_21)
    tl.store(DW_ptr + dw_base + 2 * 9 + 2, dw_22)
    tl.store(DW_ptr + dw_base + 2 * 9 + 3, dw_23)
    tl.store(DW_ptr + dw_base + 3 * 9 + 1, dw_31)
    tl.store(DW_ptr + dw_base + 3 * 9 + 2, dw_32)
    tl.store(DW_ptr + dw_base + 3 * 9 + 3, dw_33)

    # =========================================================================
    # L=2 block (5x5): dW[i,j] = sum_c dy[i,c] * x[j,c] for i,j in {4,5,6,7,8}
    # =========================================================================

    # Row 4
    dw_44 = tl.sum(dy4 * x4)
    dw_45 = tl.sum(dy4 * x5)
    dw_46 = tl.sum(dy4 * x6)
    dw_47 = tl.sum(dy4 * x7)
    dw_48 = tl.sum(dy4 * x8)
    tl.store(DW_ptr + dw_base + 4 * 9 + 4, dw_44)
    tl.store(DW_ptr + dw_base + 4 * 9 + 5, dw_45)
    tl.store(DW_ptr + dw_base + 4 * 9 + 6, dw_46)
    tl.store(DW_ptr + dw_base + 4 * 9 + 7, dw_47)
    tl.store(DW_ptr + dw_base + 4 * 9 + 8, dw_48)

    # Row 5
    dw_54 = tl.sum(dy5 * x4)
    dw_55 = tl.sum(dy5 * x5)
    dw_56 = tl.sum(dy5 * x6)
    dw_57 = tl.sum(dy5 * x7)
    dw_58 = tl.sum(dy5 * x8)
    tl.store(DW_ptr + dw_base + 5 * 9 + 4, dw_54)
    tl.store(DW_ptr + dw_base + 5 * 9 + 5, dw_55)
    tl.store(DW_ptr + dw_base + 5 * 9 + 6, dw_56)
    tl.store(DW_ptr + dw_base + 5 * 9 + 7, dw_57)
    tl.store(DW_ptr + dw_base + 5 * 9 + 8, dw_58)

    # Row 6
    dw_64 = tl.sum(dy6 * x4)
    dw_65 = tl.sum(dy6 * x5)
    dw_66 = tl.sum(dy6 * x6)
    dw_67 = tl.sum(dy6 * x7)
    dw_68 = tl.sum(dy6 * x8)
    tl.store(DW_ptr + dw_base + 6 * 9 + 4, dw_64)
    tl.store(DW_ptr + dw_base + 6 * 9 + 5, dw_65)
    tl.store(DW_ptr + dw_base + 6 * 9 + 6, dw_66)
    tl.store(DW_ptr + dw_base + 6 * 9 + 7, dw_67)
    tl.store(DW_ptr + dw_base + 6 * 9 + 8, dw_68)

    # Row 7
    dw_74 = tl.sum(dy7 * x4)
    dw_75 = tl.sum(dy7 * x5)
    dw_76 = tl.sum(dy7 * x6)
    dw_77 = tl.sum(dy7 * x7)
    dw_78 = tl.sum(dy7 * x8)
    tl.store(DW_ptr + dw_base + 7 * 9 + 4, dw_74)
    tl.store(DW_ptr + dw_base + 7 * 9 + 5, dw_75)
    tl.store(DW_ptr + dw_base + 7 * 9 + 6, dw_76)
    tl.store(DW_ptr + dw_base + 7 * 9 + 7, dw_77)
    tl.store(DW_ptr + dw_base + 7 * 9 + 8, dw_78)

    # Row 8
    dw_84 = tl.sum(dy8 * x4)
    dw_85 = tl.sum(dy8 * x5)
    dw_86 = tl.sum(dy8 * x6)
    dw_87 = tl.sum(dy8 * x7)
    dw_88 = tl.sum(dy8 * x8)
    tl.store(DW_ptr + dw_base + 8 * 9 + 4, dw_84)
    tl.store(DW_ptr + dw_base + 8 * 9 + 5, dw_85)
    tl.store(DW_ptr + dw_base + 8 * 9 + 6, dw_86)
    tl.store(DW_ptr + dw_base + 8 * 9 + 7, dw_87)
    tl.store(DW_ptr + dw_base + 8 * 9 + 8, dw_88)
