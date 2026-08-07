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

Kernels for fused_wigner_conv1 (producer-side conv1 fusion):
- wigner_conv1_fused_fwd_kernel: Forward (gather + Wigner + L→M + radial scale + pack)
- wigner_conv1_fused_bwd_kernel: Backward w.r.t. x, Wigner, radial

Kernels for fused_wigner_inv_conv2 (consumer-side conv2 fusion):
- wigner_inv_conv2_fused_fwd_kernel: Forward (unpack GEMM buffers + M→L + Wigner^{-1})
- wigner_inv_conv2_fused_bwd_kernel: Backward w.r.t. GEMM buffers and Wigner
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
    GRID_E_STRIDE: tl.constexpr,
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

    # Channel vectorization with block offset
    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    while edge_id < num_edges:
        # Load node indices for this edge
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)

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

        y4_src = (
            w44 * x4_src + w45 * x5_src + w46 * x6_src + w47 * x7_src + w48 * x8_src
        )
        y5_src = (
            w54 * x4_src + w55 * x5_src + w56 * x6_src + w57 * x7_src + w58 * x8_src
        )
        y6_src = (
            w64 * x4_src + w65 * x5_src + w66 * x6_src + w67 * x7_src + w68 * x8_src
        )
        y7_src = (
            w74 * x4_src + w75 * x5_src + w76 * x6_src + w77 * x7_src + w78 * x8_src
        )
        y8_src = (
            w84 * x4_src + w85 * x5_src + w86 * x6_src + w87 * x7_src + w88 * x8_src
        )
        y4_tgt = (
            w44 * x4_tgt + w45 * x5_tgt + w46 * x6_tgt + w47 * x7_tgt + w48 * x8_tgt
        )
        y5_tgt = (
            w54 * x4_tgt + w55 * x5_tgt + w56 * x6_tgt + w57 * x7_tgt + w58 * x8_tgt
        )
        y6_tgt = (
            w64 * x4_tgt + w65 * x5_tgt + w66 * x6_tgt + w67 * x7_tgt + w68 * x8_tgt
        )
        y7_tgt = (
            w74 * x4_tgt + w75 * x5_tgt + w76 * x6_tgt + w77 * x7_tgt + w78 * x8_tgt
        )
        y8_tgt = (
            w84 * x4_tgt + w85 * x5_tgt + w86 * x6_tgt + w87 * x7_tgt + w88 * x8_tgt
        )

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
        edge_id += GRID_E_STRIDE


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
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Backward w.r.t. input x: M→L permutation + W^T @ grad (NO scatter).

    Writes to per-edge buffer instead of atomic scatter.
    The scatter step is done separately using PyTorch's index_add_.
    This avoids atomic contention which is the main bottleneck.

    Grid: (num_edges,)
    """
    edge_id = tl.program_id(0)

    # Channel vectorization
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    while edge_id < num_edges:
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
        edge_id += GRID_E_STRIDE


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
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Forward: M→L permutation + block-diagonal Wigner^{-1} rotation.

    Loads input from M-major positions using M_TO_L_GATHER_IDX,
    computes W @ x_l using block-diagonal structure, and stores
    in L-major order. Writes the permuted x_l to a second buffer
    for backward dW computation.

    Grid: (num_edges, num_c_blocks)
    """
    edge_id = tl.program_id(0)
    c_block_id = tl.program_id(1)

    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    while edge_id < num_edges:
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

        # Save x_l for backward dW computation
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
        edge_id += GRID_E_STRIDE


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
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Backward w.r.t. input x: W^T @ dy + L→M permutation.

    Loads dy in L-major order (sequential reads), computes dx_l = W^T @ dy,
    and stores to M-major positions using L_TO_M_GATHER_IDX.

    Grid: (num_edges, num_c_blocks)
    """
    edge_id = tl.program_id(0)
    c_block_id = tl.program_id(1)

    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    while edge_id < num_edges:
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
        edge_id += GRID_E_STRIDE


# =============================================================================
# permute_wigner_inv_edge_to_node: Backward Kernel (w.r.t. Wigner)
# dW = dy @ x^T (block-diagonal outer product)
# =============================================================================


@triton.jit
def permute_wigner_inv_edge_to_node_bwd_dw_kernel(
    DY_ptr,
    X_ptr,
    DW_ptr,
    num_edges,
    C: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Backward w.r.t. Wigner: dW = dy @ x^T (block-diagonal).

    dW[e,i,j] = sum_c dy[e,i,c] * x[e,j,c]

    Only compute non-zero blocks:
    - L=0: dW[0,0]
    - L=1: dW[1:4, 1:4]
    - L=2: dW[4:9, 4:9]

    Each thread block handles one edge.
    Loads all C channels at once.

    Grid: (num_edges,)
    """
    edge_id = tl.program_id(0)

    c_range = tl.arange(0, C)
    c_mask = c_range < C

    while edge_id < num_edges:
        dy_base = edge_id * 9 * C
        x_base = edge_id * 9 * C
        dw_base = edge_id * 81

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
        edge_id += GRID_E_STRIDE


# =============================================================================
# Shared block-diagonal Wigner helpers (lmax=2: blocks L0={0}, L1={1,2,3},
# L2={4,5,6,7,8}). The @triton.jit loops over each contiguous block unroll at
# compile time; the left-to-right accumulation is bit-identical to the explicit
# FMA expression, so no numerical change vs the original unrolled kernels.
# =============================================================================


@triton.jit
def _wig_rot9(w_ptr, wb, x0, x1, x2, x3, x4, x5, x6, x7, x8):
    """
    Block-diagonal Wigner rotate y = W @ x for the 9 lmax=2 coefficients.

    Loops over the three diagonal blocks (rows/cols only over nonzeros).
    Returns the 9 rotated coefficients y0..y8 as a tuple.
    """
    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8)
    y0 = tl.load(w_ptr + wb + 0) * xs[0]
    y1 = tl.load(w_ptr + wb + 1 * 9 + 1) * xs[1]
    y2 = tl.load(w_ptr + wb + 2 * 9 + 1) * xs[1]
    y3 = tl.load(w_ptr + wb + 3 * 9 + 1) * xs[1]
    for j in tl.static_range(2, 4):
        y1 += tl.load(w_ptr + wb + 1 * 9 + j) * xs[j]
        y2 += tl.load(w_ptr + wb + 2 * 9 + j) * xs[j]
        y3 += tl.load(w_ptr + wb + 3 * 9 + j) * xs[j]
    y4 = tl.load(w_ptr + wb + 4 * 9 + 4) * xs[4]
    y5 = tl.load(w_ptr + wb + 5 * 9 + 4) * xs[4]
    y6 = tl.load(w_ptr + wb + 6 * 9 + 4) * xs[4]
    y7 = tl.load(w_ptr + wb + 7 * 9 + 4) * xs[4]
    y8 = tl.load(w_ptr + wb + 8 * 9 + 4) * xs[4]
    for j in tl.static_range(5, 9):
        y4 += tl.load(w_ptr + wb + 4 * 9 + j) * xs[j]
        y5 += tl.load(w_ptr + wb + 5 * 9 + j) * xs[j]
        y6 += tl.load(w_ptr + wb + 6 * 9 + j) * xs[j]
        y7 += tl.load(w_ptr + wb + 7 * 9 + j) * xs[j]
        y8 += tl.load(w_ptr + wb + 8 * 9 + j) * xs[j]
    return y0, y1, y2, y3, y4, y5, y6, y7, y8


@triton.jit
def _wig_rotT9(w_ptr, wb, g0, g1, g2, g3, g4, g5, g6, g7, g8):
    """
    Block-diagonal transpose rotate dx = W^T @ g for the 9 lmax=2 coefficients.

    dx[j] = sum_i W[i, j] * g[i] over each diagonal block. Returns dx0..dx8.
    """
    gs = (g0, g1, g2, g3, g4, g5, g6, g7, g8)
    dx0 = tl.load(w_ptr + wb + 0) * gs[0]
    dx1 = tl.load(w_ptr + wb + 1 * 9 + 1) * gs[1]
    dx2 = tl.load(w_ptr + wb + 1 * 9 + 2) * gs[1]
    dx3 = tl.load(w_ptr + wb + 1 * 9 + 3) * gs[1]
    for i in tl.static_range(2, 4):
        dx1 += tl.load(w_ptr + wb + i * 9 + 1) * gs[i]
        dx2 += tl.load(w_ptr + wb + i * 9 + 2) * gs[i]
        dx3 += tl.load(w_ptr + wb + i * 9 + 3) * gs[i]
    dx4 = tl.load(w_ptr + wb + 4 * 9 + 4) * gs[4]
    dx5 = tl.load(w_ptr + wb + 4 * 9 + 5) * gs[4]
    dx6 = tl.load(w_ptr + wb + 4 * 9 + 6) * gs[4]
    dx7 = tl.load(w_ptr + wb + 4 * 9 + 7) * gs[4]
    dx8 = tl.load(w_ptr + wb + 4 * 9 + 8) * gs[4]
    for i in tl.static_range(5, 9):
        dx4 += tl.load(w_ptr + wb + i * 9 + 4) * gs[i]
        dx5 += tl.load(w_ptr + wb + i * 9 + 5) * gs[i]
        dx6 += tl.load(w_ptr + wb + i * 9 + 6) * gs[i]
        dx7 += tl.load(w_ptr + wb + i * 9 + 7) * gs[i]
        dx8 += tl.load(w_ptr + wb + i * 9 + 8) * gs[i]
    return dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8


@triton.jit
def _wig_dw_store1(dw_ptr, wb, a, b):
    """
    Store block-diagonal dW[i,j] = sum_c a[i]*b[j] (single-term outer product).

    a, b are 9-tuples of per-coefficient tensors; only the three diagonal
    blocks are written, looping over their contiguous row/col ranges.
    """
    tl.store(dw_ptr + wb + 0, tl.sum(a[0] * b[0]))
    for i in tl.static_range(1, 4):
        for j in tl.static_range(1, 4):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a[i] * b[j]))
    for i in tl.static_range(4, 9):
        for j in tl.static_range(4, 9):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a[i] * b[j]))


@triton.jit
def _wig_dw_store2(dw_ptr, wb, a1, b1, a2, b2):
    """
    Store block-diagonal dW[i,j] = sum_c (a1[i]*b1[j] + a2[i]*b2[j]).

    Two-term variant summing src and tgt contributions in one pass.
    """
    tl.store(dw_ptr + wb + 0, tl.sum(a1[0] * b1[0] + a2[0] * b2[0]))
    for i in tl.static_range(1, 4):
        for j in tl.static_range(1, 4):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))
    for i in tl.static_range(4, 9):
        for j in tl.static_range(4, 9):
            tl.store(dw_ptr + wb + i * 9 + j, tl.sum(a1[i] * b1[j] + a2[i] * b2[j]))


# =============================================================================
# fused_wigner_conv1: Producer-side conv1 fusion (lmax=mmax=2)
# Forward: gather + Wigner + L->M permute + per-m radial scale + pack into
# three GEMM-ready buffers (m0,m1,m2). The [E,9,2C] x_message never materializes.
# Backward: re-gather x from x_full + edge_index, recompute rotated y from x +
# Wigner; emit grads wrt node features (per-edge), Wigner, and radial in one
# kernel.
# =============================================================================


@triton.jit
def wigner_conv1_fused_fwd_kernel(
    x_ptr,  # [N, 9, C] node feats (L-major)
    edge_index_ptr,  # [2, E]
    wigner_ptr,  # [E, 81] flattened 9x9
    radial_ptr,  # [E, RTOT] conv1 radial (RTOT = 6*C for lmax2: 768+512+256)
    m0_ptr,  # [E, 3*2C]
    m1_ptr,  # [E, 4*2C]
    m2_ptr,  # [E, 2*2C]
    num_edges,
    C: tl.constexpr,  # sphere_channels (128)
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    BLOCK_C: tl.constexpr,  # == C
    GRID_E_STRIDE: tl.constexpr,
):
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C

    C2 = 2 * C  # width of one packed M-row (src|tgt)
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    # radial block offsets (in units of C, radial row is [E, 6C])
    RAD = 6 * C2  # 1536 for C=128 (radial row width = 768+512+256)

    while edge_id < num_edges:
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        w_base = edge_id * 81

        # ---- load all 9 L-major coeffs, src & tgt ----
        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(x_ptr + s_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )
        xt = (
            tl.load(x_ptr + t_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )

        # ---- block-diagonal Wigner rotate (M->L y indices below) ----
        y0s, y1s, y2s, y3s, y4s, y5s, y6s, y7s, y8s = _wig_rot9(wigner_ptr, w_base, *xs)
        y0t, y1t, y2t, y3t, y4t, y5t, y6t, y7t, y8t = _wig_rot9(wigner_ptr, w_base, *xt)

        # ---- M-major rows (M<-L permute) then per-m radial scale + pack ----
        # M0<-y0 M1<-y2 M2<-y6 | M3<-y3 M4<-y7 M5<-y1 M6<-y5 | M7<-y8 M8<-y4
        # radial src at [base + c], tgt at [base + C + c]
        rb = edge_id * RAD
        rs = (
            tl.load(radial_ptr + rb + 0 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + c, mask=c_mask, other=0.0),
        )
        rt = (
            tl.load(radial_ptr + rb + 0 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 1 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 2 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 3 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 4 * C2 + C + c, mask=c_mask, other=0.0),
            tl.load(radial_ptr + rb + 5 * C2 + C + c, mask=c_mask, other=0.0),
        )
        # per-M-row operands, pre-permuted to M-order (radial block per row:
        # 0,1,2 | 3,4,3,4 | 5,5) so only the direct loop var indexes each tuple
        ys_perm = (y0s, y2s, y6s, y3s, y7s, y1s, y5s, y8s, y4s)
        yt_perm = (y0t, y2t, y6t, y3t, y7t, y1t, y5t, y8t, y4t)
        rs_m = (rs[0], rs[1], rs[2], rs[3], rs[4], rs[3], rs[4], rs[5], rs[5])
        rt_m = (rt[0], rt[1], rt[2], rt[3], rt[4], rt[3], rt[4], rt[5], rt[5])
        mb = edge_id * m0_row
        m1b = edge_id * m1_row
        m2b = edge_id * m2_row
        for m in tl.static_range(3):  # m0 buffer: rows M0,M1,M2
            tl.store(m0_ptr + mb + m * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m0_ptr + mb + m * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(3, 7):  # m1 buffer: rows M3,M4,M5,M6
            row = m - 3
            tl.store(m1_ptr + m1b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m1_ptr + m1b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)
        for m in tl.static_range(7, 9):  # m2 buffer: rows M7,M8
            row = m - 7
            tl.store(m2_ptr + m2b + row * C2 + c, ys_perm[m] * rs_m[m], mask=c_mask)
            tl.store(m2_ptr + m2b + row * C2 + C + c, yt_perm[m] * rt_m[m], mask=c_mask)

        edge_id += GRID_E_STRIDE


@triton.jit
def wigner_conv1_fused_bwd_kernel(
    gm0_ptr,  # [E, 3*2C] grad wrt m0 buffer
    gm1_ptr,  # [E, 4*2C]
    gm2_ptr,  # [E, 2*2C]
    wigner_ptr,  # [E, 81]
    radial_ptr,  # [E, 6C] conv1 radial
    x_ptr,  # [N, 9, C] node feats (re-gathered for x recompute)
    edge_index_ptr,  # [2, E]
    grad_edge_ptr,  # [E, 9, 2C] out: per-edge grad wrt x (L-major, for scatter)
    gwig_ptr,  # [E, 81] out: grad wrt wigner (block-diagonal)
    grad_rad_ptr,  # [E, 6C] out: grad wrt radial
    num_edges,
    x_stride_n,
    x_stride_m,
    x_stride_c,
    edge_stride,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Fused conv1+wigner backward. For each edge:
      1. re-gather L-major x (src & tgt) from x_full + edge_index
      2. recompute rotated y-values from x + wigner
      3. grad_radial = sum over reuse of (grad_m * y)
      4. grad wrt rotated y (M-major) g_y = grad_m * radial ; permute M->L to g_l
      5. grad_x = W^T @ g_l  (block-diagonal) -> per-edge L-major buffer
      6. grad_W = g_l @ x_l^T  (block-diagonal outer product)
    """
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    C2 = 2 * C
    m0_row = 3 * C2
    m1_row = 4 * C2
    m2_row = 2 * C2
    RAD = 6 * C2  # 1536 for C=128

    while edge_id < num_edges:
        w_base = edge_id * 81

        # ---- re-gather L-major x (src & tgt) from x_full + edge_index ----
        idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
        idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)
        s_base = idx0 * x_stride_n + c * x_stride_c
        t_base = idx1 * x_stride_n + c * x_stride_c
        xs = (
            tl.load(x_ptr + s_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + s_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )
        xt = (
            tl.load(x_ptr + t_base + 0 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 1 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 2 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 3 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 4 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 5 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 6 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 7 * x_stride_m, mask=c_mask, other=0.0),
            tl.load(x_ptr + t_base + 8 * x_stride_m, mask=c_mask, other=0.0),
        )

        # ---- load wigner block-diagonal (once; reused by y recompute + dx) ----
        # kept explicit so the y-recompute and grad_x below match the original
        # kernel's FMA contraction bit-for-bit (grad_x is 1-ULP sensitive)
        w00 = tl.load(wigner_ptr + w_base + 0)
        w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
        w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
        w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
        w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
        w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
        w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
        w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
        w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
        w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)
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

        # ---- recompute rotated y (L-major) from x + wigner ----
        x0s, x1s, x2s, x3s, x4s, x5s, x6s, x7s, x8s = xs
        x0t, x1t, x2t, x3t, x4t, x5t, x6t, x7t, x8t = xt
        y0s = w00 * x0s
        y0t = w00 * x0t
        y1s = w11 * x1s + w12 * x2s + w13 * x3s
        y2s = w21 * x1s + w22 * x2s + w23 * x3s
        y3s = w31 * x1s + w32 * x2s + w33 * x3s
        y1t = w11 * x1t + w12 * x2t + w13 * x3t
        y2t = w21 * x1t + w22 * x2t + w23 * x3t
        y3t = w31 * x1t + w32 * x2t + w33 * x3t
        y4s = w44 * x4s + w45 * x5s + w46 * x6s + w47 * x7s + w48 * x8s
        y5s = w54 * x4s + w55 * x5s + w56 * x6s + w57 * x7s + w58 * x8s
        y6s = w64 * x4s + w65 * x5s + w66 * x6s + w67 * x7s + w68 * x8s
        y7s = w74 * x4s + w75 * x5s + w76 * x6s + w77 * x7s + w78 * x8s
        y8s = w84 * x4s + w85 * x5s + w86 * x6s + w87 * x7s + w88 * x8s
        y4t = w44 * x4t + w45 * x5t + w46 * x6t + w47 * x7t + w48 * x8t
        y5t = w54 * x4t + w55 * x5t + w56 * x6t + w57 * x7t + w58 * x8t
        y6t = w64 * x4t + w65 * x5t + w66 * x6t + w67 * x7t + w68 * x8t
        y7t = w74 * x4t + w75 * x5t + w76 * x6t + w77 * x7t + w78 * x8t
        y8t = w84 * x4t + w85 * x5t + w86 * x6t + w87 * x7t + w88 * x8t

        # ---- load radial (6 blocks, src then tgt) ----
        rbb = edge_id * RAD
        r0s = tl.load(radial_ptr + rbb + 0 * C2 + c, mask=c_mask, other=0.0)
        r0t = tl.load(radial_ptr + rbb + 0 * C2 + C + c, mask=c_mask, other=0.0)
        r1s = tl.load(radial_ptr + rbb + 1 * C2 + c, mask=c_mask, other=0.0)
        r1t = tl.load(radial_ptr + rbb + 1 * C2 + C + c, mask=c_mask, other=0.0)
        r2s = tl.load(radial_ptr + rbb + 2 * C2 + c, mask=c_mask, other=0.0)
        r2t = tl.load(radial_ptr + rbb + 2 * C2 + C + c, mask=c_mask, other=0.0)
        ra_s = tl.load(radial_ptr + rbb + 3 * C2 + c, mask=c_mask, other=0.0)
        ra_t = tl.load(radial_ptr + rbb + 3 * C2 + C + c, mask=c_mask, other=0.0)
        rb_s = tl.load(radial_ptr + rbb + 4 * C2 + c, mask=c_mask, other=0.0)
        rb_t = tl.load(radial_ptr + rbb + 4 * C2 + C + c, mask=c_mask, other=0.0)
        rc_s = tl.load(radial_ptr + rbb + 5 * C2 + c, mask=c_mask, other=0.0)
        rc_t = tl.load(radial_ptr + rbb + 5 * C2 + C + c, mask=c_mask, other=0.0)

        # ---- load grad_m (M-major packed) ----
        mb = edge_id * m0_row
        g_m0s = tl.load(gm0_ptr + mb + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m0t = tl.load(gm0_ptr + mb + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m1s = tl.load(gm0_ptr + mb + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m1t = tl.load(gm0_ptr + mb + 1 * C2 + C + c, mask=c_mask, other=0.0)
        g_m2s = tl.load(gm0_ptr + mb + 2 * C2 + c, mask=c_mask, other=0.0)
        g_m2t = tl.load(gm0_ptr + mb + 2 * C2 + C + c, mask=c_mask, other=0.0)
        m1b = edge_id * m1_row
        g_m3s = tl.load(gm1_ptr + m1b + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m3t = tl.load(gm1_ptr + m1b + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m4s = tl.load(gm1_ptr + m1b + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m4t = tl.load(gm1_ptr + m1b + 1 * C2 + C + c, mask=c_mask, other=0.0)
        g_m5s = tl.load(gm1_ptr + m1b + 2 * C2 + c, mask=c_mask, other=0.0)
        g_m5t = tl.load(gm1_ptr + m1b + 2 * C2 + C + c, mask=c_mask, other=0.0)
        g_m6s = tl.load(gm1_ptr + m1b + 3 * C2 + c, mask=c_mask, other=0.0)
        g_m6t = tl.load(gm1_ptr + m1b + 3 * C2 + C + c, mask=c_mask, other=0.0)
        m2b = edge_id * m2_row
        g_m7s = tl.load(gm2_ptr + m2b + 0 * C2 + c, mask=c_mask, other=0.0)
        g_m7t = tl.load(gm2_ptr + m2b + 0 * C2 + C + c, mask=c_mask, other=0.0)
        g_m8s = tl.load(gm2_ptr + m2b + 1 * C2 + c, mask=c_mask, other=0.0)
        g_m8t = tl.load(gm2_ptr + m2b + 1 * C2 + C + c, mask=c_mask, other=0.0)

        # ---- grad_radial = sum over reuse of grad_m * y (y in M order) ----
        # M0<-y0 (r0), M1<-y2 (r1), M2<-y6 (r2); blk a (r3): M3<-y3,M5<-y1;
        # blk b (r4): M4<-y7,M6<-y5; blk c (r5): M7<-y8,M8<-y4
        gr_s = (
            g_m0s * y0s,
            g_m1s * y2s,
            g_m2s * y6s,
            g_m3s * y3s + g_m5s * y1s,
            g_m4s * y7s + g_m6s * y5s,
            g_m7s * y8s + g_m8s * y4s,
        )
        gr_t = (
            g_m0t * y0t,
            g_m1t * y2t,
            g_m2t * y6t,
            g_m3t * y3t + g_m5t * y1t,
            g_m4t * y7t + g_m6t * y5t,
            g_m7t * y8t + g_m8t * y4t,
        )
        for r in tl.static_range(6):
            tl.store(grad_rad_ptr + rbb + r * C2 + c, gr_s[r], mask=c_mask)
            tl.store(grad_rad_ptr + rbb + r * C2 + C + c, gr_t[r], mask=c_mask)

        # ---- grad wrt rotated y (put back into L order) g_y = grad_m * radial ----
        gys = (
            g_m0s * r0s,  # L0 (M0)
            g_m5s * ra_s,  # L1 (M5)
            g_m1s * r1s,  # L2 (M1)
            g_m3s * ra_s,  # L3 (M3)
            g_m8s * rc_s,  # L4 (M8)
            g_m6s * rb_s,  # L5 (M6)
            g_m2s * r2s,  # L6 (M2)
            g_m4s * rb_s,  # L7 (M4)
            g_m7s * rc_s,  # L8 (M7)
        )
        gyt = (
            g_m0t * r0t,
            g_m5t * ra_t,
            g_m1t * r1t,
            g_m3t * ra_t,
            g_m8t * rc_t,
            g_m6t * rb_t,
            g_m2t * r2t,
            g_m4t * rb_t,
            g_m7t * rc_t,
        )

        # ---- grad_x = W^T @ g_l (block-diagonal); g_l is L-major grad wrt y ----
        # explicit expressions reusing the single wigner load above; matches the
        # original bit-for-bit (grad_x FMA contraction is 1-ULP sensitive)
        gy0s, gy1s, gy2s, gy3s, gy4s, gy5s, gy6s, gy7s, gy8s = gys
        gy0t, gy1t, gy2t, gy3t, gy4t, gy5t, gy6t, gy7t, gy8t = gyt
        dxs = (
            w00 * gy0s,
            w11 * gy1s + w21 * gy2s + w31 * gy3s,
            w12 * gy1s + w22 * gy2s + w32 * gy3s,
            w13 * gy1s + w23 * gy2s + w33 * gy3s,
            w44 * gy4s + w54 * gy5s + w64 * gy6s + w74 * gy7s + w84 * gy8s,
            w45 * gy4s + w55 * gy5s + w65 * gy6s + w75 * gy7s + w85 * gy8s,
            w46 * gy4s + w56 * gy5s + w66 * gy6s + w76 * gy7s + w86 * gy8s,
            w47 * gy4s + w57 * gy5s + w67 * gy6s + w77 * gy7s + w87 * gy8s,
            w48 * gy4s + w58 * gy5s + w68 * gy6s + w78 * gy7s + w88 * gy8s,
        )
        dxt = (
            w00 * gy0t,
            w11 * gy1t + w21 * gy2t + w31 * gy3t,
            w12 * gy1t + w22 * gy2t + w32 * gy3t,
            w13 * gy1t + w23 * gy2t + w33 * gy3t,
            w44 * gy4t + w54 * gy5t + w64 * gy6t + w74 * gy7t + w84 * gy8t,
            w45 * gy4t + w55 * gy5t + w65 * gy6t + w75 * gy7t + w85 * gy8t,
            w46 * gy4t + w56 * gy5t + w66 * gy6t + w76 * gy7t + w86 * gy8t,
            w47 * gy4t + w57 * gy5t + w67 * gy6t + w77 * gy7t + w87 * gy8t,
            w48 * gy4t + w58 * gy5t + w68 * gy6t + w78 * gy7t + w88 * gy8t,
        )

        # store per-edge grad_x (L-major src|tgt)
        gb = edge_id * 9 * C2
        for i in tl.static_range(9):
            tl.store(grad_edge_ptr + gb + i * C2 + c, dxs[i], mask=c_mask)
            tl.store(grad_edge_ptr + gb + i * C2 + C + c, dxt[i], mask=c_mask)

        # ---- grad_W = g_l @ x_l^T (block-diagonal), summed over src & tgt ----
        # dW[i,j] = sum_c ( gy_i_src*x_j_src + gy_i_tgt*x_j_tgt )
        _wig_dw_store2(gwig_ptr, w_base, gys, xs, gyt, xt)

        edge_id += GRID_E_STRIDE


# =============================================================================
# fused_wigner_inv_conv2: Consumer-side conv2 fusion (lmax=mmax=2)
# Forward: read the three conv2 GEMM buffers (g0,g1,g2) directly, absorbing the
# M->L unpack, then block-diagonal inverse-Wigner rotate -> x_rotated [E,9,C].
# The [E,9,C] M-major intermediate never materializes; scatter stays outside.
# Backward: dx_l = W^T @ dy scattered to g0/g1/g2 layout; dW = dy @ x_l^T with
# x_l recomputed from the saved GEMM buffers.
# =============================================================================


@triton.jit
def wigner_inv_conv2_fused_fwd_kernel(
    g0_ptr,  # [E, 3*C] conv2 fc_m0 output (rows M0,M1,M2)
    g1_ptr,  # [E, 4*C] conv2 m=1 block-GEMM output (rows M3,M4,M5,M6)
    g2_ptr,  # [E, 2*C] conv2 m=2 block-GEMM output (rows M7,M8)
    W_ptr,  # [E, 81] flattened inverse-wigner 9x9
    OUT_ptr,  # [E, 9, C] rotated features (L-major) for scatter
    num_edges,
    C: tl.constexpr,  # sphere_channels (128)
    BLOCK_C: tl.constexpr,  # == C
    GRID_E_STRIDE: tl.constexpr,
):
    edge_id = tl.program_id(0)
    c_block_id = tl.program_id(1)
    c_start = c_block_id * BLOCK_C
    c = c_start + tl.arange(0, BLOCK_C)
    c_mask = c < C

    g0_row = 3 * C
    g1_row = 4 * C
    g2_row = 2 * C

    while edge_id < num_edges:
        w_base = edge_id * 81
        out_base = edge_id * 9 * C

        g0b = edge_id * g0_row
        g1b = edge_id * g1_row
        g2b = edge_id * g2_row

        # ---- load L-major x from GEMM buffers (M->L permute absorbed) ----
        x0 = tl.load(g0_ptr + g0b + 0 * C + c, mask=c_mask, other=0.0)  # L0<-M0
        x1 = tl.load(g1_ptr + g1b + 2 * C + c, mask=c_mask, other=0.0)  # L1<-M5
        x2 = tl.load(g0_ptr + g0b + 1 * C + c, mask=c_mask, other=0.0)  # L2<-M1
        x3 = tl.load(g1_ptr + g1b + 0 * C + c, mask=c_mask, other=0.0)  # L3<-M3
        x4 = tl.load(g2_ptr + g2b + 1 * C + c, mask=c_mask, other=0.0)  # L4<-M8
        x5 = tl.load(g1_ptr + g1b + 3 * C + c, mask=c_mask, other=0.0)  # L5<-M6
        x6 = tl.load(g0_ptr + g0b + 2 * C + c, mask=c_mask, other=0.0)  # L6<-M2
        x7 = tl.load(g1_ptr + g1b + 1 * C + c, mask=c_mask, other=0.0)  # L7<-M4
        x8 = tl.load(g2_ptr + g2b + 0 * C + c, mask=c_mask, other=0.0)  # L8<-M7

        # ---- block-diagonal inverse-Wigner rotate (W @ x_l) ----
        ys = _wig_rot9(W_ptr, w_base, x0, x1, x2, x3, x4, x5, x6, x7, x8)

        # ---- store x_rotated (L-major, sequential) ----
        for i in tl.static_range(9):
            tl.store(OUT_ptr + out_base + i * C + c, ys[i], mask=c_mask)

        edge_id += GRID_E_STRIDE


@triton.jit
def wigner_inv_conv2_fused_bwd_kernel(
    dy_ptr,  # [E, 9, C] grad wrt x_rotated (L-major)
    g0_ptr,  # [E, 3*C] saved conv2 GEMM buffers (for dW recompute of x_l)
    g1_ptr,  # [E, 4*C]
    g2_ptr,  # [E, 2*C]
    W_ptr,  # [E, 81] inverse-wigner
    dg0_ptr,  # [E, 3*C] out: grad wrt g0
    dg1_ptr,  # [E, 4*C] out: grad wrt g1
    dg2_ptr,  # [E, 2*C] out: grad wrt g2
    dw_ptr,  # [E, 81] out: grad wrt wigner (block-diagonal)
    num_edges,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,  # == C (all channels; needed for tl.sum over C in dW)
    GRID_E_STRIDE: tl.constexpr,
):
    """
    Fused conv2-buffer + inv-wigner backward. For each edge:
      1. load dy (L-major) and W (block-diagonal)
      2. dx_l = W^T @ dy  (block-diagonal) -> scatter to g0/g1/g2 (L->M layout)
      3. recompute x_l from saved GEMM buffers (M->L)
      4. dW = dy @ x_l^T  (block-diagonal outer product)
    """
    edge_id = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C

    g0_row = 3 * C
    g1_row = 4 * C
    g2_row = 2 * C

    while edge_id < num_edges:
        w_base = edge_id * 81
        dy_base = edge_id * 9 * C

        # ---- load dy (L-major) ----
        dy = (
            tl.load(dy_ptr + dy_base + 0 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 1 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 2 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 3 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 4 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 5 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 6 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 7 * C + c, mask=c_mask, other=0.0),
            tl.load(dy_ptr + dy_base + 8 * C + c, mask=c_mask, other=0.0),
        )

        # ---- dx_l = W^T @ dy (block-diagonal) ----
        dx = _wig_rotT9(W_ptr, w_base, *dy)

        # ---- scatter dx_l (L-major) into GEMM-buffer layout (L->M) ----
        g0b = edge_id * g0_row
        g1b = edge_id * g1_row
        g2b = edge_id * g2_row
        # L0->M0=g0[0], L2->M1=g0[1], L6->M2=g0[2]
        tl.store(dg0_ptr + g0b + 0 * C + c, dx[0], mask=c_mask)
        tl.store(dg0_ptr + g0b + 1 * C + c, dx[2], mask=c_mask)
        tl.store(dg0_ptr + g0b + 2 * C + c, dx[6], mask=c_mask)
        # L3->M3=g1[0], L7->M4=g1[1], L1->M5=g1[2], L5->M6=g1[3]
        tl.store(dg1_ptr + g1b + 0 * C + c, dx[3], mask=c_mask)
        tl.store(dg1_ptr + g1b + 1 * C + c, dx[7], mask=c_mask)
        tl.store(dg1_ptr + g1b + 2 * C + c, dx[1], mask=c_mask)
        tl.store(dg1_ptr + g1b + 3 * C + c, dx[5], mask=c_mask)
        # L8->M7=g2[0], L4->M8=g2[1]
        tl.store(dg2_ptr + g2b + 0 * C + c, dx[8], mask=c_mask)
        tl.store(dg2_ptr + g2b + 1 * C + c, dx[4], mask=c_mask)

        # ---- recompute x_l from saved GEMM buffers (M->L) ----
        x = (
            tl.load(g0_ptr + g0b + 0 * C + c, mask=c_mask, other=0.0),  # L0<-M0
            tl.load(g1_ptr + g1b + 2 * C + c, mask=c_mask, other=0.0),  # L1<-M5
            tl.load(g0_ptr + g0b + 1 * C + c, mask=c_mask, other=0.0),  # L2<-M1
            tl.load(g1_ptr + g1b + 0 * C + c, mask=c_mask, other=0.0),  # L3<-M3
            tl.load(g2_ptr + g2b + 1 * C + c, mask=c_mask, other=0.0),  # L4<-M8
            tl.load(g1_ptr + g1b + 3 * C + c, mask=c_mask, other=0.0),  # L5<-M6
            tl.load(g0_ptr + g0b + 2 * C + c, mask=c_mask, other=0.0),  # L6<-M2
            tl.load(g1_ptr + g1b + 1 * C + c, mask=c_mask, other=0.0),  # L7<-M4
            tl.load(g2_ptr + g2b + 0 * C + c, mask=c_mask, other=0.0),  # L8<-M7
        )

        # ---- dW = dy @ x_l^T (block-diagonal outer product) ----
        _wig_dw_store1(dw_ptr, w_base, dy, x)

        edge_id += GRID_E_STRIDE
