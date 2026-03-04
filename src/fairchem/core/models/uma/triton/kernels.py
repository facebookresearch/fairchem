"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Triton GPU kernels for block-diagonal Wigner operations at lmax=2.

This file contains ONLY Triton @jit kernels. Python wrappers and autograd
functions are in separate files for readability.

Kernels for node_to_edge_wigner_permute:
- node_to_edge_wigner_permute_kernel: Forward (gather + Wigner + L→M)
- node_to_edge_wigner_permute_bwd_dx_kernel: Backward w.r.t. input x

Kernels for permute_wigner_inv_edge_to_node:
- permute_wigner_inv_edge_to_node_kernel: Forward (M→L + Wigner^{-1})
- permute_wigner_inv_edge_to_node_bwd_dx_kernel: Backward w.r.t. input x
- permute_wigner_inv_edge_to_node_bwd_dw_kernel: Backward w.r.t. Wigner matrices
"""

from __future__ import annotations

import triton
import triton.language as tl

# =============================================================================
# node_to_edge_wigner_permute: Forward Kernel
# Gather x[src], x[tgt] -> Wigner rotate -> L→M permute
# =============================================================================


@triton.jit
def node_to_edge_wigner_permute_kernel(
    x_ptr,
    edge_index_ptr,
    wigner_ptr,
    out_ptr,
    x_edge_ptr,
    num_edges,
    sphere_channels,
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    out_stride_e,
    out_stride_l,
    out_stride_c,
    x_edge_stride_e,
    x_edge_stride_l,
    x_edge_stride_c,
    BLOCK_C: tl.constexpr,
):
    """
    Forward: Node-to-edge gather + block-diagonal Wigner rotation + L→M permutation.

    Performs:
        1. Gather features from source and target nodes
        2. Block-diagonal Wigner rotation (exploits lmax=2 sparsity)
        3. L→M permutation
        4. Store both rotated output and pre-Wigner x_edge (for backward)

    The x_edge side output is stored as [E, 9, 2C] with src at [:C], tgt at [C:2C].
    These values are already in registers so the extra stores are free.

    Grid: (num_edges, num_c_blocks)
    """
    edge_id = tl.program_id(0)
    c_block_id = tl.program_id(1)

    if edge_id >= num_edges:
        return

    # Load node indices for this edge
    idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
    idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)

    # Channel vectorization with block offset
    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    # Wigner base pointer (flattened 9x9 = 81 per edge)
    w_base = edge_id * 81
    out_base = edge_id * out_stride_e

    # =========================================================================
    # Load all 9 coefficients from both nodes
    # =========================================================================
    x0_src = tl.load(
        x_ptr + idx0 * x_stride_n + 0 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x0_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 0 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x1_src = tl.load(
        x_ptr + idx0 * x_stride_n + 1 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x1_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 1 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x2_src = tl.load(
        x_ptr + idx0 * x_stride_n + 2 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x2_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 2 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x3_src = tl.load(
        x_ptr + idx0 * x_stride_n + 3 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x3_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 3 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x4_src = tl.load(
        x_ptr + idx0 * x_stride_n + 4 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x4_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 4 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x5_src = tl.load(
        x_ptr + idx0 * x_stride_n + 5 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x5_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 5 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x6_src = tl.load(
        x_ptr + idx0 * x_stride_n + 6 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x6_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 6 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x7_src = tl.load(
        x_ptr + idx0 * x_stride_n + 7 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x7_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 7 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    x8_src = tl.load(
        x_ptr + idx0 * x_stride_n + 8 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )
    x8_tgt = tl.load(
        x_ptr + idx1 * x_stride_n + 8 * x_stride_m + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # =========================================================================
    # Store x_edge side outputs (for backward dW computation)
    # =========================================================================
    x_edge_base = edge_id * x_edge_stride_e
    # Source at [:C]
    tl.store(
        x_edge_ptr + x_edge_base + 0 * x_edge_stride_l + c_range * x_edge_stride_c,
        x0_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 1 * x_edge_stride_l + c_range * x_edge_stride_c,
        x1_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 2 * x_edge_stride_l + c_range * x_edge_stride_c,
        x2_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 3 * x_edge_stride_l + c_range * x_edge_stride_c,
        x3_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 4 * x_edge_stride_l + c_range * x_edge_stride_c,
        x4_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 5 * x_edge_stride_l + c_range * x_edge_stride_c,
        x5_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 6 * x_edge_stride_l + c_range * x_edge_stride_c,
        x6_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 7 * x_edge_stride_l + c_range * x_edge_stride_c,
        x7_src,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr + x_edge_base + 8 * x_edge_stride_l + c_range * x_edge_stride_c,
        x8_src,
        mask=c_mask,
    )
    # Target at [C:2C]
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 0 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x0_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 1 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x1_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 2 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x2_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 3 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x3_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 4 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x4_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 5 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x5_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 6 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x6_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 7 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x7_tgt,
        mask=c_mask,
    )
    tl.store(
        x_edge_ptr
        + x_edge_base
        + 8 * x_edge_stride_l
        + sphere_channels * x_edge_stride_c
        + c_range * x_edge_stride_c,
        x8_tgt,
        mask=c_mask,
    )

    # =========================================================================
    # Block-diagonal Wigner rotation (exploits lmax=2 sparsity)
    # =========================================================================

    # L=0 block (1x1)
    w00 = tl.load(wigner_ptr + w_base + 0)
    y0_src = w00 * x0_src
    y0_tgt = w00 * x0_tgt

    # L=1 block (3x3)
    w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
    w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
    w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
    w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
    w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
    w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
    w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
    w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
    w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)

    y1_src = w11 * x1_src + w12 * x2_src + w13 * x3_src
    y2_src = w21 * x1_src + w22 * x2_src + w23 * x3_src
    y3_src = w31 * x1_src + w32 * x2_src + w33 * x3_src
    y1_tgt = w11 * x1_tgt + w12 * x2_tgt + w13 * x3_tgt
    y2_tgt = w21 * x1_tgt + w22 * x2_tgt + w23 * x3_tgt
    y3_tgt = w31 * x1_tgt + w32 * x2_tgt + w33 * x3_tgt

    # L=2 block (5x5)
    w44 = tl.load(wigner_ptr + w_base + 4 * 9 + 4)
    w45 = tl.load(wigner_ptr + w_base + 4 * 9 + 5)
    w46 = tl.load(wigner_ptr + w_base + 4 * 9 + 6)
    w47 = tl.load(wigner_ptr + w_base + 4 * 9 + 7)
    w48 = tl.load(wigner_ptr + w_base + 4 * 9 + 8)
    w54 = tl.load(wigner_ptr + w_base + 5 * 9 + 4)
    w55 = tl.load(wigner_ptr + w_base + 5 * 9 + 5)
    w56 = tl.load(wigner_ptr + w_base + 5 * 9 + 6)
    w57 = tl.load(wigner_ptr + w_base + 5 * 9 + 7)
    w58 = tl.load(wigner_ptr + w_base + 5 * 9 + 8)
    w64 = tl.load(wigner_ptr + w_base + 6 * 9 + 4)
    w65 = tl.load(wigner_ptr + w_base + 6 * 9 + 5)
    w66 = tl.load(wigner_ptr + w_base + 6 * 9 + 6)
    w67 = tl.load(wigner_ptr + w_base + 6 * 9 + 7)
    w68 = tl.load(wigner_ptr + w_base + 6 * 9 + 8)
    w74 = tl.load(wigner_ptr + w_base + 7 * 9 + 4)
    w75 = tl.load(wigner_ptr + w_base + 7 * 9 + 5)
    w76 = tl.load(wigner_ptr + w_base + 7 * 9 + 6)
    w77 = tl.load(wigner_ptr + w_base + 7 * 9 + 7)
    w78 = tl.load(wigner_ptr + w_base + 7 * 9 + 8)
    w84 = tl.load(wigner_ptr + w_base + 8 * 9 + 4)
    w85 = tl.load(wigner_ptr + w_base + 8 * 9 + 5)
    w86 = tl.load(wigner_ptr + w_base + 8 * 9 + 6)
    w87 = tl.load(wigner_ptr + w_base + 8 * 9 + 7)
    w88 = tl.load(wigner_ptr + w_base + 8 * 9 + 8)

    y4_src = w44 * x4_src + w45 * x5_src + w46 * x6_src + w47 * x7_src + w48 * x8_src
    y5_src = w54 * x4_src + w55 * x5_src + w56 * x6_src + w57 * x7_src + w58 * x8_src
    y6_src = w64 * x4_src + w65 * x5_src + w66 * x6_src + w67 * x7_src + w68 * x8_src
    y7_src = w74 * x4_src + w75 * x5_src + w76 * x6_src + w77 * x7_src + w78 * x8_src
    y8_src = w84 * x4_src + w85 * x5_src + w86 * x6_src + w87 * x7_src + w88 * x8_src
    y4_tgt = w44 * x4_tgt + w45 * x5_tgt + w46 * x6_tgt + w47 * x7_tgt + w48 * x8_tgt
    y5_tgt = w54 * x4_tgt + w55 * x5_tgt + w56 * x6_tgt + w57 * x7_tgt + w58 * x8_tgt
    y6_tgt = w64 * x4_tgt + w65 * x5_tgt + w66 * x6_tgt + w67 * x7_tgt + w68 * x8_tgt
    y7_tgt = w74 * x4_tgt + w75 * x5_tgt + w76 * x6_tgt + w77 * x7_tgt + w78 * x8_tgt
    y8_tgt = w84 * x4_tgt + w85 * x5_tgt + w86 * x6_tgt + w87 * x7_tgt + w88 * x8_tgt

    # =========================================================================
    # Store with L→M permutation
    # L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    # out_m[i] = y_l[L_TO_M_GATHER_IDX[i]]
    # =========================================================================

    # M=0 <- L=0
    tl.store(
        out_ptr + out_base + 0 * out_stride_l + c_range * out_stride_c,
        y0_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 0 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y0_tgt,
        mask=c_mask,
    )

    # M=1 <- L=2
    tl.store(
        out_ptr + out_base + 1 * out_stride_l + c_range * out_stride_c,
        y2_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 1 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y2_tgt,
        mask=c_mask,
    )

    # M=2 <- L=6
    tl.store(
        out_ptr + out_base + 2 * out_stride_l + c_range * out_stride_c,
        y6_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 2 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y6_tgt,
        mask=c_mask,
    )

    # M=3 <- L=3
    tl.store(
        out_ptr + out_base + 3 * out_stride_l + c_range * out_stride_c,
        y3_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 3 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y3_tgt,
        mask=c_mask,
    )

    # M=4 <- L=7
    tl.store(
        out_ptr + out_base + 4 * out_stride_l + c_range * out_stride_c,
        y7_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 4 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y7_tgt,
        mask=c_mask,
    )

    # M=5 <- L=1
    tl.store(
        out_ptr + out_base + 5 * out_stride_l + c_range * out_stride_c,
        y1_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 5 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y1_tgt,
        mask=c_mask,
    )

    # M=6 <- L=5
    tl.store(
        out_ptr + out_base + 6 * out_stride_l + c_range * out_stride_c,
        y5_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 6 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y5_tgt,
        mask=c_mask,
    )

    # M=7 <- L=8
    tl.store(
        out_ptr + out_base + 7 * out_stride_l + c_range * out_stride_c,
        y8_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 7 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y8_tgt,
        mask=c_mask,
    )

    # M=8 <- L=4
    tl.store(
        out_ptr + out_base + 8 * out_stride_l + c_range * out_stride_c,
        y4_src,
        mask=c_mask,
    )
    tl.store(
        out_ptr
        + out_base
        + 8 * out_stride_l
        + sphere_channels * out_stride_c
        + c_range * out_stride_c,
        y4_tgt,
        mask=c_mask,
    )


# =============================================================================
# node_to_edge_wigner_permute: Backward Kernel (w.r.t. input x)
# Computes M→L + W^T @ grad, writes per-edge gradient (no scatter)
# =============================================================================


@triton.jit
def node_to_edge_wigner_permute_bwd_dx_kernel(
    grad_out_ptr,  # [E, 9, 2C] gradient from downstream (M-major)
    wigner_ptr,  # [E, 81] Wigner matrices (flattened 9x9)
    grad_edge_ptr,  # [E, 9, 2C] output gradient per edge (no scatter)
    num_edges,
    sphere_channels,
    grad_stride_e,
    grad_stride_l,
    grad_stride_c,
    out_stride_e,
    out_stride_l,
    out_stride_c,
    BLOCK_C: tl.constexpr,
):
    """
    Backward w.r.t. input x: M→L permutation + W^T @ grad (NO scatter).

    Writes to per-edge buffer instead of atomic scatter.
    The scatter step is done separately using PyTorch's index_add_.
    This avoids atomic contention which is the main bottleneck.

    Grid: (num_edges,)
    """
    edge_id = tl.program_id(0)
    if edge_id >= num_edges:
        return

    # Channel vectorization
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    # Wigner and gradient base pointers
    w_base = edge_id * 81
    grad_base = edge_id * grad_stride_e
    out_base = edge_id * out_stride_e

    # =========================================================================
    # Load gradient (M-major) and apply M→L permutation inline
    # M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    # =========================================================================

    # L=0 <- M=0
    dy_l0_src = tl.load(
        grad_out_ptr + grad_base + 0 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l0_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 0 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=1 <- M=5
    dy_l1_src = tl.load(
        grad_out_ptr + grad_base + 5 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l1_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 5 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=2 <- M=1
    dy_l2_src = tl.load(
        grad_out_ptr + grad_base + 1 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l2_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 1 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=3 <- M=3
    dy_l3_src = tl.load(
        grad_out_ptr + grad_base + 3 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l3_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 3 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=4 <- M=8
    dy_l4_src = tl.load(
        grad_out_ptr + grad_base + 8 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l4_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 8 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=5 <- M=6
    dy_l5_src = tl.load(
        grad_out_ptr + grad_base + 6 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l5_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 6 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=6 <- M=2
    dy_l6_src = tl.load(
        grad_out_ptr + grad_base + 2 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l6_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 2 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=7 <- M=4
    dy_l7_src = tl.load(
        grad_out_ptr + grad_base + 4 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l7_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 4 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # L=8 <- M=7
    dy_l8_src = tl.load(
        grad_out_ptr + grad_base + 7 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy_l8_tgt = tl.load(
        grad_out_ptr
        + grad_base
        + 7 * grad_stride_l
        + sphere_channels
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # =========================================================================
    # Apply W^T @ dy using block-diagonal sparsity
    # =========================================================================

    # L=0 block: 1x1
    w00 = tl.load(wigner_ptr + w_base + 0)
    dx0_src = w00 * dy_l0_src
    dx0_tgt = w00 * dy_l0_tgt

    # L=1 block: 3x3 at [1:4, 1:4]
    w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
    w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
    w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
    w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
    w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
    w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
    w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
    w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
    w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)

    # W^T @ dy: dx[j] = sum_i W[i,j] * dy[i]
    dx1_src = w11 * dy_l1_src + w21 * dy_l2_src + w31 * dy_l3_src
    dx2_src = w12 * dy_l1_src + w22 * dy_l2_src + w32 * dy_l3_src
    dx3_src = w13 * dy_l1_src + w23 * dy_l2_src + w33 * dy_l3_src

    dx1_tgt = w11 * dy_l1_tgt + w21 * dy_l2_tgt + w31 * dy_l3_tgt
    dx2_tgt = w12 * dy_l1_tgt + w22 * dy_l2_tgt + w32 * dy_l3_tgt
    dx3_tgt = w13 * dy_l1_tgt + w23 * dy_l2_tgt + w33 * dy_l3_tgt

    # L=2 block: 5x5 at [4:9, 4:9]
    w44 = tl.load(wigner_ptr + w_base + 4 * 9 + 4)
    w45 = tl.load(wigner_ptr + w_base + 4 * 9 + 5)
    w46 = tl.load(wigner_ptr + w_base + 4 * 9 + 6)
    w47 = tl.load(wigner_ptr + w_base + 4 * 9 + 7)
    w48 = tl.load(wigner_ptr + w_base + 4 * 9 + 8)

    w54 = tl.load(wigner_ptr + w_base + 5 * 9 + 4)
    w55 = tl.load(wigner_ptr + w_base + 5 * 9 + 5)
    w56 = tl.load(wigner_ptr + w_base + 5 * 9 + 6)
    w57 = tl.load(wigner_ptr + w_base + 5 * 9 + 7)
    w58 = tl.load(wigner_ptr + w_base + 5 * 9 + 8)

    w64 = tl.load(wigner_ptr + w_base + 6 * 9 + 4)
    w65 = tl.load(wigner_ptr + w_base + 6 * 9 + 5)
    w66 = tl.load(wigner_ptr + w_base + 6 * 9 + 6)
    w67 = tl.load(wigner_ptr + w_base + 6 * 9 + 7)
    w68 = tl.load(wigner_ptr + w_base + 6 * 9 + 8)

    w74 = tl.load(wigner_ptr + w_base + 7 * 9 + 4)
    w75 = tl.load(wigner_ptr + w_base + 7 * 9 + 5)
    w76 = tl.load(wigner_ptr + w_base + 7 * 9 + 6)
    w77 = tl.load(wigner_ptr + w_base + 7 * 9 + 7)
    w78 = tl.load(wigner_ptr + w_base + 7 * 9 + 8)

    w84 = tl.load(wigner_ptr + w_base + 8 * 9 + 4)
    w85 = tl.load(wigner_ptr + w_base + 8 * 9 + 5)
    w86 = tl.load(wigner_ptr + w_base + 8 * 9 + 6)
    w87 = tl.load(wigner_ptr + w_base + 8 * 9 + 7)
    w88 = tl.load(wigner_ptr + w_base + 8 * 9 + 8)

    # W^T @ dy for L=2 block
    dx4_src = (
        w44 * dy_l4_src
        + w54 * dy_l5_src
        + w64 * dy_l6_src
        + w74 * dy_l7_src
        + w84 * dy_l8_src
    )
    dx5_src = (
        w45 * dy_l4_src
        + w55 * dy_l5_src
        + w65 * dy_l6_src
        + w75 * dy_l7_src
        + w85 * dy_l8_src
    )
    dx6_src = (
        w46 * dy_l4_src
        + w56 * dy_l5_src
        + w66 * dy_l6_src
        + w76 * dy_l7_src
        + w86 * dy_l8_src
    )
    dx7_src = (
        w47 * dy_l4_src
        + w57 * dy_l5_src
        + w67 * dy_l6_src
        + w77 * dy_l7_src
        + w87 * dy_l8_src
    )
    dx8_src = (
        w48 * dy_l4_src
        + w58 * dy_l5_src
        + w68 * dy_l6_src
        + w78 * dy_l7_src
        + w88 * dy_l8_src
    )

    dx4_tgt = (
        w44 * dy_l4_tgt
        + w54 * dy_l5_tgt
        + w64 * dy_l6_tgt
        + w74 * dy_l7_tgt
        + w84 * dy_l8_tgt
    )
    dx5_tgt = (
        w45 * dy_l4_tgt
        + w55 * dy_l5_tgt
        + w65 * dy_l6_tgt
        + w75 * dy_l7_tgt
        + w85 * dy_l8_tgt
    )
    dx6_tgt = (
        w46 * dy_l4_tgt
        + w56 * dy_l5_tgt
        + w66 * dy_l6_tgt
        + w76 * dy_l7_tgt
        + w86 * dy_l8_tgt
    )
    dx7_tgt = (
        w47 * dy_l4_tgt
        + w57 * dy_l5_tgt
        + w67 * dy_l6_tgt
        + w77 * dy_l7_tgt
        + w87 * dy_l8_tgt
    )
    dx8_tgt = (
        w48 * dy_l4_tgt
        + w58 * dy_l5_tgt
        + w68 * dy_l6_tgt
        + w78 * dy_l7_tgt
        + w88 * dy_l8_tgt
    )

    # =========================================================================
    # Store per-edge gradient (L-major order for subsequent scatter)
    # =========================================================================
    tl.store(
        grad_edge_ptr + out_base + 0 * out_stride_l + c_range * out_stride_c,
        dx0_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 1 * out_stride_l + c_range * out_stride_c,
        dx1_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 2 * out_stride_l + c_range * out_stride_c,
        dx2_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 3 * out_stride_l + c_range * out_stride_c,
        dx3_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 4 * out_stride_l + c_range * out_stride_c,
        dx4_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 5 * out_stride_l + c_range * out_stride_c,
        dx5_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 6 * out_stride_l + c_range * out_stride_c,
        dx6_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 7 * out_stride_l + c_range * out_stride_c,
        dx7_src,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr + out_base + 8 * out_stride_l + c_range * out_stride_c,
        dx8_src,
        mask=c_mask,
    )

    # Target gradients at offset sphere_channels
    tl.store(
        grad_edge_ptr
        + out_base
        + 0 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx0_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 1 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx1_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 2 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx2_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 3 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx3_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 4 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx4_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 5 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx5_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 6 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx6_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 7 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx7_tgt,
        mask=c_mask,
    )
    tl.store(
        grad_edge_ptr
        + out_base
        + 8 * out_stride_l
        + sphere_channels
        + c_range * out_stride_c,
        dx8_tgt,
        mask=c_mask,
    )


# =============================================================================
# permute_wigner_inv_edge_to_node: Forward Kernel
# M→L permutation + Wigner^{-1} rotation
# =============================================================================


@triton.jit
def permute_wigner_inv_edge_to_node_kernel(
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
    Forward: M→L permutation + block-diagonal Wigner^{-1} rotation.

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


# =============================================================================
# permute_wigner_inv_edge_to_node: Backward Kernel (w.r.t. input x)
# W^T @ dy + L→M permutation
# =============================================================================


@triton.jit
def permute_wigner_inv_edge_to_node_bwd_dx_kernel(
    DY_ptr,
    W_ptr,
    DX_ptr,
    num_edges,
    sphere_channels,
    BLOCK_C: tl.constexpr,
):
    """
    Backward w.r.t. input x: W^T @ dy + L→M permutation.

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
# permute_wigner_inv_edge_to_node: Backward Kernel (w.r.t. Wigner)
# dW = dy @ x^T (block-diagonal outer product)
# =============================================================================


@triton.jit
def permute_wigner_inv_edge_to_node_bwd_dw_kernel(
    DY_ptr,
    X_ptr,
    DW_ptr,
    E: tl.constexpr,
    C: tl.constexpr,
):
    """
    Backward w.r.t. Wigner: dW = dy @ x^T (block-diagonal).

    dW[e,i,j] = sum_c dy[e,i,c] * x[e,j,c]

    Only compute non-zero blocks:
    - L=0: dW[0,0]
    - L=1: dW[1:4, 1:4]
    - L=2: dW[4:9, 4:9]

    Each thread block handles one edge.
    Load all channels at once (assuming C <= 128).

    Grid: (num_edges,)
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
