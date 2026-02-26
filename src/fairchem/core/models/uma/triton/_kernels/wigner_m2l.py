"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Forward: M→L permutation + Wigner transform (edge-level, no gather).

Standalone edge-level kernel that combines:
    1. Permute M-major → L-major (reorder coefficients)
    2. Apply Wigner rotation: y = W @ x

Used in UMASFastGPUPermuteWignerInvEdgeToNode.forward() for the inverse
rotation operation (edge features back to node-aligned frame).

Data flow:
    x_m[E, 9, C] → permute M→L → x_l[E, 9, C] → W @ x_l → y_l[E, 9, C]
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def wigner_m2l_kernel(
    X_ptr,
    W_ptr,
    OUT_ptr,
    XL_ptr,
    num_edges,
    sphere_channels,
    BLOCK_C: tl.constexpr,
):
    """
    Fused M→L permutation + block-diagonal Wigner rotation.

    Args:
        X_ptr: Input [E, 9, C] in M-major order
        W_ptr: Per-edge Wigner matrices [E, 81] (flattened 9x9)
        OUT_ptr: Output [E, 9, C] in L-major order (y = W @ x_l)
        XL_ptr: Buffer to save x_l [E, 9, C] for backward dW computation
        num_edges: Number of edges E
        sphere_channels: Number of channels C
        BLOCK_C: Channel block size for vectorization

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

    # =========================================================================
    # Step 1: Load from M-major positions and permute to L-major
    # M_TO_L_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    # x_l[i] = x_m[M_TO_L_IDX[i]]
    # =========================================================================
    x0 = tl.load(
        X_ptr + x_base + 0 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=0 ← M=0
    x1 = tl.load(
        X_ptr + x_base + 5 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=1 ← M=5
    x2 = tl.load(
        X_ptr + x_base + 1 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=2 ← M=1
    x3 = tl.load(
        X_ptr + x_base + 3 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=3 ← M=3
    x4 = tl.load(
        X_ptr + x_base + 8 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=4 ← M=8
    x5 = tl.load(
        X_ptr + x_base + 6 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=5 ← M=6
    x6 = tl.load(
        X_ptr + x_base + 2 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=6 ← M=2
    x7 = tl.load(
        X_ptr + x_base + 4 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=7 ← M=4
    x8 = tl.load(
        X_ptr + x_base + 7 * sphere_channels + c_range, mask=c_mask, other=0.0
    )  # L=8 ← M=7

    # =========================================================================
    # Step 2: Save x_l for backward dW = dy @ x_l^T computation
    # =========================================================================
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

    # =========================================================================
    # Step 3: Apply Wigner rotation y = W @ x (block-diagonal)
    # =========================================================================

    # L=0 block: 1x1 scalar
    w00 = tl.load(W_ptr + w_base + 0)
    y0 = w00 * x0

    # L=1 block: 3x3 matrix multiply (indices 1,2,3)
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

    # L=2 block: 5x5 matrix multiply (indices 4,5,6,7,8)
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

    # =========================================================================
    # Step 4: Store output in L-major order (sequential)
    # =========================================================================
    tl.store(OUT_ptr + out_base + 0 * sphere_channels + c_range, y0, mask=c_mask)
    tl.store(OUT_ptr + out_base + 1 * sphere_channels + c_range, y1, mask=c_mask)
    tl.store(OUT_ptr + out_base + 2 * sphere_channels + c_range, y2, mask=c_mask)
    tl.store(OUT_ptr + out_base + 3 * sphere_channels + c_range, y3, mask=c_mask)
    tl.store(OUT_ptr + out_base + 4 * sphere_channels + c_range, y4, mask=c_mask)
    tl.store(OUT_ptr + out_base + 5 * sphere_channels + c_range, y5, mask=c_mask)
    tl.store(OUT_ptr + out_base + 6 * sphere_channels + c_range, y6, mask=c_mask)
    tl.store(OUT_ptr + out_base + 7 * sphere_channels + c_range, y7, mask=c_mask)
    tl.store(OUT_ptr + out_base + 8 * sphere_channels + c_range, y8, mask=c_mask)
