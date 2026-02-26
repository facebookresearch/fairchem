"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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


if HAS_TRITON:
    # =========================================================================
    # Kernel: Block-diagonal Wigner multiply (lmax=2) - Backward dW
    # dW[e,i,j] = sum_c dy[e,i,c] * x[e,j,c]  (outer product per edge)
    # Required for force computation!
    # =========================================================================

    @triton.jit
    def wigner_lmax2_bwd_dw_kernel(
        DY_ptr,
        X_ptr,
        DW_ptr,
        E: tl.constexpr,
        C: tl.constexpr,
    ):
        """
        Backward for Wigner: compute dW = dy @ x^T (block-diagonal).

        dW[e,i,j] = sum_c dy[e,i,c] * x[e,j,c]

        Only compute non-zero blocks:
        - L=0: dW[0,0]
        - L=1: dW[1:4, 1:4]
        - L=2: dW[4:9, 4:9]

        Each thread block handles one edge.
        Load all channels at once (assuming C <= 128).
        """
        edge_id = tl.program_id(0)

        dy_base = edge_id * 9 * C
        x_base = edge_id * 9 * C
        dw_base = edge_id * 81

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

        # L=0 block (1x1): dW[0,0] = sum_c dy[0,c] * x[0,c]
        dw_00 = tl.sum(dy0 * x0)
        tl.store(DW_ptr + dw_base + 0, dw_00)

        # L=1 block (3x3): dW[1:4, 1:4]
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

        # L=2 block (5x5): dW[4:9, 4:9]
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

    # =========================================================================
    # Kernel: Fused M->L permutation + Wigner multiply (lmax=2) - Forward
    # Loads from M-major positions, computes W @ x_l, stores in L-major order.
    # Optionally saves x_l for backward dW computation.
    # Uses 2D grid (num_edges, num_c_blocks) to handle C > 128.
    # =========================================================================

    @triton.jit
    def fused_m_to_l_wigner_lmax2_kernel(
        X_ptr,
        W_ptr,
        OUT_ptr,
        XL_ptr,
        num_edges,
        sphere_channels,
        BLOCK_C: tl.constexpr,
        SAVE_XL: tl.constexpr,
    ):
        """
        Fused M->L permutation + block-diagonal Wigner multiply for lmax=2.

        Loads input from M-major positions using M_TO_L_GATHER_IDX,
        computes W @ x_l using block-diagonal structure, and stores
        in L-major order. Optionally writes the permuted x_l to a
        second buffer for backward dW computation.

        Grid: (num_edges, num_c_blocks)
        """
        edge_id = tl.program_id(0)
        c_block_id = tl.program_id(1)

        if edge_id >= num_edges:
            return

        c_start = c_block_id * BLOCK_C
        c_range = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_range < sphere_channels

        w_base = edge_id * 81
        x_base = edge_id * 9 * sphere_channels
        out_base = edge_id * 9 * sphere_channels

        # Load from M-major positions using M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
        # x_l[i] = x_m[M_TO_L_GATHER_IDX[i]]
        x0 = tl.load(
            X_ptr + x_base + 0 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=0 <- M=0
        x1 = tl.load(
            X_ptr + x_base + 5 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=1 <- M=5
        x2 = tl.load(
            X_ptr + x_base + 1 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=2 <- M=1
        x3 = tl.load(
            X_ptr + x_base + 3 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=3 <- M=3
        x4 = tl.load(
            X_ptr + x_base + 8 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=4 <- M=8
        x5 = tl.load(
            X_ptr + x_base + 6 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=5 <- M=6
        x6 = tl.load(
            X_ptr + x_base + 2 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=6 <- M=2
        x7 = tl.load(
            X_ptr + x_base + 4 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=7 <- M=4
        x8 = tl.load(
            X_ptr + x_base + 7 * sphere_channels + c_range, mask=c_mask, other=0.0
        )  # L=8 <- M=7

        # Optionally save x_l for backward dW computation
        if SAVE_XL:
            xl_base = edge_id * 9 * sphere_channels
            tl.store(XL_ptr + xl_base + 0 * sphere_channels + c_range, x0, mask=c_mask)
            tl.store(XL_ptr + xl_base + 1 * sphere_channels + c_range, x1, mask=c_mask)
            tl.store(XL_ptr + xl_base + 2 * sphere_channels + c_range, x2, mask=c_mask)
            tl.store(XL_ptr + xl_base + 3 * sphere_channels + c_range, x3, mask=c_mask)
            tl.store(XL_ptr + xl_base + 4 * sphere_channels + c_range, x4, mask=c_mask)
            tl.store(XL_ptr + xl_base + 5 * sphere_channels + c_range, x5, mask=c_mask)
            tl.store(XL_ptr + xl_base + 6 * sphere_channels + c_range, x6, mask=c_mask)
            tl.store(XL_ptr + xl_base + 7 * sphere_channels + c_range, x7, mask=c_mask)
            tl.store(XL_ptr + xl_base + 8 * sphere_channels + c_range, x8, mask=c_mask)

        # L=0 block (1x1)
        w00 = tl.load(W_ptr + w_base + 0)
        y0 = w00 * x0

        # L=1 block (3x3) - indices 1,2,3
        w11 = tl.load(W_ptr + w_base + 1 * 9 + 1)
        w12 = tl.load(W_ptr + w_base + 1 * 9 + 2)
        w13 = tl.load(W_ptr + w_base + 1 * 9 + 3)
        w21 = tl.load(W_ptr + w_base + 2 * 9 + 1)
        w22 = tl.load(W_ptr + w_base + 2 * 9 + 2)
        w23 = tl.load(W_ptr + w_base + 2 * 9 + 3)
        w31 = tl.load(W_ptr + w_base + 3 * 9 + 1)
        w32 = tl.load(W_ptr + w_base + 3 * 9 + 2)
        w33 = tl.load(W_ptr + w_base + 3 * 9 + 3)

        y1 = w11 * x1 + w12 * x2 + w13 * x3
        y2 = w21 * x1 + w22 * x2 + w23 * x3
        y3 = w31 * x1 + w32 * x2 + w33 * x3

        # L=2 block (5x5) - indices 4,5,6,7,8
        w44 = tl.load(W_ptr + w_base + 4 * 9 + 4)
        w45 = tl.load(W_ptr + w_base + 4 * 9 + 5)
        w46 = tl.load(W_ptr + w_base + 4 * 9 + 6)
        w47 = tl.load(W_ptr + w_base + 4 * 9 + 7)
        w48 = tl.load(W_ptr + w_base + 4 * 9 + 8)
        y4 = w44 * x4 + w45 * x5 + w46 * x6 + w47 * x7 + w48 * x8

        w54 = tl.load(W_ptr + w_base + 5 * 9 + 4)
        w55 = tl.load(W_ptr + w_base + 5 * 9 + 5)
        w56 = tl.load(W_ptr + w_base + 5 * 9 + 6)
        w57 = tl.load(W_ptr + w_base + 5 * 9 + 7)
        w58 = tl.load(W_ptr + w_base + 5 * 9 + 8)
        y5 = w54 * x4 + w55 * x5 + w56 * x6 + w57 * x7 + w58 * x8

        w64 = tl.load(W_ptr + w_base + 6 * 9 + 4)
        w65 = tl.load(W_ptr + w_base + 6 * 9 + 5)
        w66 = tl.load(W_ptr + w_base + 6 * 9 + 6)
        w67 = tl.load(W_ptr + w_base + 6 * 9 + 7)
        w68 = tl.load(W_ptr + w_base + 6 * 9 + 8)
        y6 = w64 * x4 + w65 * x5 + w66 * x6 + w67 * x7 + w68 * x8

        w74 = tl.load(W_ptr + w_base + 7 * 9 + 4)
        w75 = tl.load(W_ptr + w_base + 7 * 9 + 5)
        w76 = tl.load(W_ptr + w_base + 7 * 9 + 6)
        w77 = tl.load(W_ptr + w_base + 7 * 9 + 7)
        w78 = tl.load(W_ptr + w_base + 7 * 9 + 8)
        y7 = w74 * x4 + w75 * x5 + w76 * x6 + w77 * x7 + w78 * x8

        w84 = tl.load(W_ptr + w_base + 8 * 9 + 4)
        w85 = tl.load(W_ptr + w_base + 8 * 9 + 5)
        w86 = tl.load(W_ptr + w_base + 8 * 9 + 6)
        w87 = tl.load(W_ptr + w_base + 8 * 9 + 7)
        w88 = tl.load(W_ptr + w_base + 8 * 9 + 8)
        y8 = w84 * x4 + w85 * x5 + w86 * x6 + w87 * x7 + w88 * x8

        # Store in L-major order (sequential)
        tl.store(OUT_ptr + out_base + 0 * sphere_channels + c_range, y0, mask=c_mask)
        tl.store(OUT_ptr + out_base + 1 * sphere_channels + c_range, y1, mask=c_mask)
        tl.store(OUT_ptr + out_base + 2 * sphere_channels + c_range, y2, mask=c_mask)
        tl.store(OUT_ptr + out_base + 3 * sphere_channels + c_range, y3, mask=c_mask)
        tl.store(OUT_ptr + out_base + 4 * sphere_channels + c_range, y4, mask=c_mask)
        tl.store(OUT_ptr + out_base + 5 * sphere_channels + c_range, y5, mask=c_mask)
        tl.store(OUT_ptr + out_base + 6 * sphere_channels + c_range, y6, mask=c_mask)
        tl.store(OUT_ptr + out_base + 7 * sphere_channels + c_range, y7, mask=c_mask)
        tl.store(OUT_ptr + out_base + 8 * sphere_channels + c_range, y8, mask=c_mask)

    # =========================================================================
    # Kernel: Fused Wigner backward dx + L->M permutation (lmax=2)
    # Loads dy in L-major order, computes dx_l = W^T @ dy, stores to M-major
    # positions using L_TO_M_GATHER_IDX.
    # Uses 2D grid (num_edges, num_c_blocks) to handle C > 128.
    # =========================================================================

    @triton.jit
    def fused_wigner_bwd_dx_l_to_m_kernel(
        DY_ptr,
        W_ptr,
        DX_ptr,
        num_edges,
        sphere_channels,
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused Wigner backward dx + L->M permutation for lmax=2.

        Loads dy in L-major order (sequential reads), computes dx_l = W^T @ dy,
        and stores to M-major positions using L_TO_M_GATHER_IDX.

        Grid: (num_edges, num_c_blocks)
        """
        edge_id = tl.program_id(0)
        c_block_id = tl.program_id(1)

        if edge_id >= num_edges:
            return

        c_start = c_block_id * BLOCK_C
        c_range = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_range < sphere_channels

        w_base = edge_id * 81
        dy_base = edge_id * 9 * sphere_channels
        dx_base = edge_id * 9 * sphere_channels

        # Load dy in L-major order (sequential reads)
        dy0 = tl.load(
            DY_ptr + dy_base + 0 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy1 = tl.load(
            DY_ptr + dy_base + 1 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy2 = tl.load(
            DY_ptr + dy_base + 2 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy3 = tl.load(
            DY_ptr + dy_base + 3 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy4 = tl.load(
            DY_ptr + dy_base + 4 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy5 = tl.load(
            DY_ptr + dy_base + 5 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy6 = tl.load(
            DY_ptr + dy_base + 6 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy7 = tl.load(
            DY_ptr + dy_base + 7 * sphere_channels + c_range, mask=c_mask, other=0.0
        )
        dy8 = tl.load(
            DY_ptr + dy_base + 8 * sphere_channels + c_range, mask=c_mask, other=0.0
        )

        # L=0 block (1x1) - transpose is same
        w00 = tl.load(W_ptr + w_base + 0)
        dx0 = w00 * dy0

        # L=1 block (3x3) - W^T @ dy
        w11 = tl.load(W_ptr + w_base + 1 * 9 + 1)
        w12 = tl.load(W_ptr + w_base + 1 * 9 + 2)
        w13 = tl.load(W_ptr + w_base + 1 * 9 + 3)
        w21 = tl.load(W_ptr + w_base + 2 * 9 + 1)
        w22 = tl.load(W_ptr + w_base + 2 * 9 + 2)
        w23 = tl.load(W_ptr + w_base + 2 * 9 + 3)
        w31 = tl.load(W_ptr + w_base + 3 * 9 + 1)
        w32 = tl.load(W_ptr + w_base + 3 * 9 + 2)
        w33 = tl.load(W_ptr + w_base + 3 * 9 + 3)

        dx1 = w11 * dy1 + w21 * dy2 + w31 * dy3
        dx2 = w12 * dy1 + w22 * dy2 + w32 * dy3
        dx3 = w13 * dy1 + w23 * dy2 + w33 * dy3

        # L=2 block (5x5) - W^T @ dy
        w44 = tl.load(W_ptr + w_base + 4 * 9 + 4)
        w45 = tl.load(W_ptr + w_base + 4 * 9 + 5)
        w46 = tl.load(W_ptr + w_base + 4 * 9 + 6)
        w47 = tl.load(W_ptr + w_base + 4 * 9 + 7)
        w48 = tl.load(W_ptr + w_base + 4 * 9 + 8)

        w54 = tl.load(W_ptr + w_base + 5 * 9 + 4)
        w55 = tl.load(W_ptr + w_base + 5 * 9 + 5)
        w56 = tl.load(W_ptr + w_base + 5 * 9 + 6)
        w57 = tl.load(W_ptr + w_base + 5 * 9 + 7)
        w58 = tl.load(W_ptr + w_base + 5 * 9 + 8)

        w64 = tl.load(W_ptr + w_base + 6 * 9 + 4)
        w65 = tl.load(W_ptr + w_base + 6 * 9 + 5)
        w66 = tl.load(W_ptr + w_base + 6 * 9 + 6)
        w67 = tl.load(W_ptr + w_base + 6 * 9 + 7)
        w68 = tl.load(W_ptr + w_base + 6 * 9 + 8)

        w74 = tl.load(W_ptr + w_base + 7 * 9 + 4)
        w75 = tl.load(W_ptr + w_base + 7 * 9 + 5)
        w76 = tl.load(W_ptr + w_base + 7 * 9 + 6)
        w77 = tl.load(W_ptr + w_base + 7 * 9 + 7)
        w78 = tl.load(W_ptr + w_base + 7 * 9 + 8)

        w84 = tl.load(W_ptr + w_base + 8 * 9 + 4)
        w85 = tl.load(W_ptr + w_base + 8 * 9 + 5)
        w86 = tl.load(W_ptr + w_base + 8 * 9 + 6)
        w87 = tl.load(W_ptr + w_base + 8 * 9 + 7)
        w88 = tl.load(W_ptr + w_base + 8 * 9 + 8)

        dx4 = w44 * dy4 + w54 * dy5 + w64 * dy6 + w74 * dy7 + w84 * dy8
        dx5 = w45 * dy4 + w55 * dy5 + w65 * dy6 + w75 * dy7 + w85 * dy8
        dx6 = w46 * dy4 + w56 * dy5 + w66 * dy6 + w76 * dy7 + w86 * dy8
        dx7 = w47 * dy4 + w57 * dy5 + w67 * dy6 + w77 * dy7 + w87 * dy8
        dx8 = w48 * dy4 + w58 * dy5 + w68 * dy6 + w78 * dy7 + w88 * dy8

        # Store to M-major positions using L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
        # out_m[i] = dx_l[L_TO_M_GATHER_IDX[i]]
        tl.store(
            DX_ptr + dx_base + 0 * sphere_channels + c_range, dx0, mask=c_mask
        )  # M=0 <- L=0
        tl.store(
            DX_ptr + dx_base + 1 * sphere_channels + c_range, dx2, mask=c_mask
        )  # M=1 <- L=2
        tl.store(
            DX_ptr + dx_base + 2 * sphere_channels + c_range, dx6, mask=c_mask
        )  # M=2 <- L=6
        tl.store(
            DX_ptr + dx_base + 3 * sphere_channels + c_range, dx3, mask=c_mask
        )  # M=3 <- L=3
        tl.store(
            DX_ptr + dx_base + 4 * sphere_channels + c_range, dx7, mask=c_mask
        )  # M=4 <- L=7
        tl.store(
            DX_ptr + dx_base + 5 * sphere_channels + c_range, dx1, mask=c_mask
        )  # M=5 <- L=1
        tl.store(
            DX_ptr + dx_base + 6 * sphere_channels + c_range, dx5, mask=c_mask
        )  # M=6 <- L=5
        tl.store(
            DX_ptr + dx_base + 7 * sphere_channels + c_range, dx8, mask=c_mask
        )  # M=7 <- L=8
        tl.store(
            DX_ptr + dx_base + 8 * sphere_channels + c_range, dx4, mask=c_mask
        )  # M=8 <- L=4


# =============================================================================
# Python Wrappers for lmax=2 kernels
# =============================================================================

BLOCK_C = 128  # Channel block size for Triton kernels


def _wigner_lmax2_bwd_dw(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Backward pass for Wigner: compute dW = dy @ x^T (block-diagonal).
    """
    dy = dy.contiguous()  # Grad tensors from autograd may not be contiguous
    E, num_coeffs, C = dy.shape
    dw = torch.zeros(E, 9, 9, device=dy.device, dtype=dy.dtype)

    if C <= BLOCK_C:
        wigner_lmax2_bwd_dw_kernel[(E,)](dy, x, dw, E, C)
        return dw

    # Tile over channel blocks and accumulate gradients
    # Each kernel call computes partial sum over a channel block
    # We use a temporary buffer and accumulate to get the full sum
    dw_temp = torch.empty(E, 9, 9, device=dy.device, dtype=dy.dtype)
    for c_start in range(0, C, BLOCK_C):
        c_end = min(c_start + BLOCK_C, C)
        block_c = c_end - c_start
        dy_block = dy[:, :, c_start:c_end].contiguous()
        x_block = x[:, :, c_start:c_end].contiguous()
        wigner_lmax2_bwd_dw_kernel[(E,)](dy_block, x_block, dw_temp, E, block_c)
        dw += dw_temp  # Accumulate partial gradient
    return dw


def _fused_m_to_l_wigner_fwd(
    x: torch.Tensor, wigner: torch.Tensor, save_xl: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fused forward: M->L permutation + Wigner multiply in one kernel.

    Returns (y_l, x_l) if save_xl=True, else (y_l, None).
    """
    E, num_coeffs, C = x.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    out = torch.empty_like(x)
    x_l = torch.empty_like(x) if save_xl else x  # dummy if not saving
    fused_m_to_l_wigner_lmax2_kernel[(E, num_c_blocks)](
        x,
        wigner,
        out,
        x_l,
        E,
        C,
        BLOCK_C=BLOCK_C,
        SAVE_XL=save_xl,
    )
    return out, x_l if save_xl else None


def _fused_wigner_bwd_dx_l_to_m(dy: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor:
    """
    Fused backward dx: Wigner^T @ dy + L->M permutation in one kernel.
    """
    dy = dy.contiguous()
    E, num_coeffs, C = dy.shape
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    dx = torch.empty_like(dy)
    fused_wigner_bwd_dx_l_to_m_kernel[(E, num_c_blocks)](
        dy,
        wigner,
        dx,
        E,
        C,
        BLOCK_C=BLOCK_C,
    )
    return dx


# =============================================================================
# Autograd Function Class
# =============================================================================


class FusedMToLThenWignerLmax2Function(torch.autograd.Function):
    """
    Autograd function for fused M->L + Wigner (lmax=2).

    Forward: Single kernel fuses M->L permutation + W @ x_l.
    Backward:
        dx_m = fused kernel (W^T @ dy + L->M permutation)
        dW = dy_l @ x_l^T (reuses existing kernel)

    Eliminates separate permutation kernel launches in both directions.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor:
        y_l, x_l = _fused_m_to_l_wigner_fwd(x, wigner, save_xl=True)
        ctx.save_for_backward(x_l, wigner)
        return y_l

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, wigner = ctx.saved_tensors
        grad_x_m = _fused_wigner_bwd_dx_l_to_m(grad_output, wigner)
        grad_wigner = _wigner_lmax2_bwd_dw(grad_output, x_l)
        return grad_x_m, grad_wigner
