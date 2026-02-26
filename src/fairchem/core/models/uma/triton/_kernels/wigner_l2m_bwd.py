"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Backward: W^T transform + L→M permutation (edge-level, no scatter).

This is the backward kernel for wigner_m2l_kernel. Computes gradient w.r.t.
input features (dx). The inverse operations in reverse order:
    Forward: x_m → permute M→L → x_l → W @ x_l → y_l
    Backward: dy_l → W^T @ dy → dx_l → permute L→M → dx_m

Note: There is no corresponding forward kernel for this operation because
L→M permutation + Wigner is fused into node_to_edge_wigner_l2m_kernel.

Data flow:
    dy_l[E, 9, C] → W^T @ dy → dx_l[E, 9, C] → permute L→M → dx_m[E, 9, C]
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def wigner_l2m_bwd_kernel(
    DY_ptr,
    W_ptr,
    DX_ptr,
    num_edges,
    sphere_channels,
    BLOCK_C: tl.constexpr,
):
    """
    Backward dx: Wigner transpose + L→M permutation.

    Args:
        DY_ptr: Upstream gradient [E, 9, C] in L-major order
        W_ptr: Per-edge Wigner matrices [E, 81] (flattened 9x9)
        DX_ptr: Output gradient [E, 9, C] in M-major order
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
    dy_base = edge_id * 9 * sphere_channels
    dx_base = edge_id * 9 * sphere_channels

    # =========================================================================
    # Step 1: Load dy in L-major order (sequential reads)
    # =========================================================================
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

    # =========================================================================
    # Step 2: Apply W^T (transpose Wigner) - block diagonal
    # dx = W^T @ dy (standard backprop through linear layer)
    # =========================================================================

    # L=0 block: 1x1 (transpose = same)
    w00 = tl.load(W_ptr + w_base + 0)
    dx0 = w00 * dy0

    # L=1 block: 3x3 transpose
    # dx[i] = sum_j W[j,i] * dy[j]
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

    # L=2 block: 5x5 transpose
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

    # =========================================================================
    # Step 3: Store to M-major positions (L→M permutation)
    # L_TO_M_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    # dx_m[i] = dx_l[L_TO_M_IDX[i]]
    # =========================================================================
    tl.store(
        DX_ptr + dx_base + 0 * sphere_channels + c_range, dx0, mask=c_mask
    )  # M=0 ← L=0
    tl.store(
        DX_ptr + dx_base + 1 * sphere_channels + c_range, dx2, mask=c_mask
    )  # M=1 ← L=2
    tl.store(
        DX_ptr + dx_base + 2 * sphere_channels + c_range, dx6, mask=c_mask
    )  # M=2 ← L=6
    tl.store(
        DX_ptr + dx_base + 3 * sphere_channels + c_range, dx3, mask=c_mask
    )  # M=3 ← L=3
    tl.store(
        DX_ptr + dx_base + 4 * sphere_channels + c_range, dx7, mask=c_mask
    )  # M=4 ← L=7
    tl.store(
        DX_ptr + dx_base + 5 * sphere_channels + c_range, dx1, mask=c_mask
    )  # M=5 ← L=1
    tl.store(
        DX_ptr + dx_base + 6 * sphere_channels + c_range, dx5, mask=c_mask
    )  # M=6 ← L=5
    tl.store(
        DX_ptr + dx_base + 7 * sphere_channels + c_range, dx8, mask=c_mask
    )  # M=7 ← L=8
    tl.store(
        DX_ptr + dx_base + 8 * sphere_channels + c_range, dx4, mask=c_mask
    )  # M=8 ← L=4
