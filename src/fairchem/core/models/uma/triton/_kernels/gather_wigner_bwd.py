"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import triton
import triton.language as tl

# =============================================================================
# Triton Backward Kernel (transform only, no scatter) - V2 OPTIMIZED
# =============================================================================


@triton.jit
def wigner_transform_bwd_kernel(
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
    """Triton backward kernel: M→L + W^T @ grad (NO scatter).

    V2 optimized: Writes to per-edge buffer instead of atomic scatter.
    The scatter step is done separately using PyTorch's index_add_.

    This avoids atomic contention which is the main bottleneck in the
    original fused_wigner_scatter_bwd_kernel.
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
    # Apply W^T (transpose Wigner) - block diagonal structure
    # =========================================================================

    # L=0 block: 1x1
    w00 = tl.load(wigner_ptr + w_base + 0)
    dx0_src = w00 * dy_l0_src
    dx0_tgt = w00 * dy_l0_tgt

    # L=1 block: 3x3
    w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
    w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
    w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
    w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
    w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
    w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
    w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
    w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
    w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)

    dx1_src = w11 * dy_l1_src + w21 * dy_l2_src + w31 * dy_l3_src
    dx2_src = w12 * dy_l1_src + w22 * dy_l2_src + w32 * dy_l3_src
    dx3_src = w13 * dy_l1_src + w23 * dy_l2_src + w33 * dy_l3_src

    dx1_tgt = w11 * dy_l1_tgt + w21 * dy_l2_tgt + w31 * dy_l3_tgt
    dx2_tgt = w12 * dy_l1_tgt + w22 * dy_l2_tgt + w32 * dy_l3_tgt
    dx3_tgt = w13 * dy_l1_tgt + w23 * dy_l2_tgt + w33 * dy_l3_tgt

    # L=2 block: 5x5
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
    # Store to per-edge buffer (no atomics!) - [E, 9, 2C] with src in [:C], tgt in [C:]
    # =========================================================================

    # Store source gradients (first C channels)
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

    # Store target gradients (second C channels)
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
