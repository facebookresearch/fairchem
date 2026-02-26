"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Forward: Gather node features to edges + Wigner transform + L→M permutation.

This is the main forward kernel for the edge message passing operation.
Combines three operations to minimize memory traffic:
    1. Gather: x[node] → x_edge[edge] for both src and tgt nodes
    2. Rotate: y = W @ x (block-diagonal Wigner D-matrix)
    3. Permute: L-major → M-major ordering

Data flow:
    x[N, 9, C] → gather[edge_index] → W @ x_l → permute L→M → out[E, 9, 2C]

Side output:
    x_edge[E, 9, 2C] - pre-rotation gathered features (for backward dW)
    Layout: src features at [:, :, :C], tgt features at [:, :, C:2C]

Note: There is no corresponding backward kernel for this operation. The backward
for dx uses wigner_m2l_bwd_kernel (M→L + W^T), and the scatter is handled by
PyTorch's index_add_ which is faster than Triton atomics.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def node_to_edge_wigner_l2m_kernel(
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
    Fused gather + Wigner rotation + L→M permutation.

    Args:
        x_ptr: Node features [N, 9, C]
        edge_index_ptr: Edge indices [2, E] (row 0 = src, row 1 = tgt)
        wigner_ptr: Per-edge Wigner matrices [E, 81] (flattened 9x9)
        out_ptr: Output [E, 9, 2C] in M-major order (rotated)
        x_edge_ptr: Side output [E, 9, 2C] in L-major order (pre-rotation)
        num_edges: Number of edges E
        sphere_channels: Number of channels C
        *_stride_*: Tensor strides for flexible memory layouts
        BLOCK_C: Channel block size for vectorization

    Grid: (num_edges, num_c_blocks)
        - Each thread block handles one edge and BLOCK_C channels
        - 2D grid allows processing more channels than BLOCK_C
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
    # Step 1: Gather features from both nodes (src and tgt)
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
    # Step 2: Store x_edge side output for backward dW computation
    # Layout: [E, 9, 2C] with src at [:C] and tgt at [C:2C]
    # These stores are essentially free since values are already in registers
    # =========================================================================
    x_edge_base = edge_id * x_edge_stride_e

    tl.store(
        x_edge_ptr + x_edge_base + 0 * x_edge_stride_l + c_range * x_edge_stride_c,
        x0_src,
        mask=c_mask,
    )
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
        x_edge_ptr + x_edge_base + 1 * x_edge_stride_l + c_range * x_edge_stride_c,
        x1_src,
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
        x_edge_ptr + x_edge_base + 2 * x_edge_stride_l + c_range * x_edge_stride_c,
        x2_src,
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
        x_edge_ptr + x_edge_base + 3 * x_edge_stride_l + c_range * x_edge_stride_c,
        x3_src,
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
        x_edge_ptr + x_edge_base + 4 * x_edge_stride_l + c_range * x_edge_stride_c,
        x4_src,
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
        x_edge_ptr + x_edge_base + 5 * x_edge_stride_l + c_range * x_edge_stride_c,
        x5_src,
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
        x_edge_ptr + x_edge_base + 6 * x_edge_stride_l + c_range * x_edge_stride_c,
        x6_src,
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
        x_edge_ptr + x_edge_base + 7 * x_edge_stride_l + c_range * x_edge_stride_c,
        x7_src,
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
        x_edge_ptr + x_edge_base + 8 * x_edge_stride_l + c_range * x_edge_stride_c,
        x8_src,
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
    # Step 3: Apply Wigner rotation (y = W @ x)
    # Block-diagonal structure: L=0 (1x1), L=1 (3x3), L=2 (5x5)
    # =========================================================================

    # L=0 block: scalar multiply
    w00 = tl.load(wigner_ptr + w_base + 0)
    y0_src = w00 * x0_src
    y0_tgt = w00 * x0_tgt

    # L=1 block: 3x3 matrix multiply
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

    # L=2 block: 5x5 matrix multiply
    w44 = tl.load(wigner_ptr + w_base + 4 * 9 + 4)
    w45 = tl.load(wigner_ptr + w_base + 4 * 9 + 5)
    w46 = tl.load(wigner_ptr + w_base + 4 * 9 + 6)
    w47 = tl.load(wigner_ptr + w_base + 4 * 9 + 7)
    w48 = tl.load(wigner_ptr + w_base + 4 * 9 + 8)
    y4_src = w44 * x4_src + w45 * x5_src + w46 * x6_src + w47 * x7_src + w48 * x8_src
    y4_tgt = w44 * x4_tgt + w45 * x5_tgt + w46 * x6_tgt + w47 * x7_tgt + w48 * x8_tgt

    w54 = tl.load(wigner_ptr + w_base + 5 * 9 + 4)
    w55 = tl.load(wigner_ptr + w_base + 5 * 9 + 5)
    w56 = tl.load(wigner_ptr + w_base + 5 * 9 + 6)
    w57 = tl.load(wigner_ptr + w_base + 5 * 9 + 7)
    w58 = tl.load(wigner_ptr + w_base + 5 * 9 + 8)
    y5_src = w54 * x4_src + w55 * x5_src + w56 * x6_src + w57 * x7_src + w58 * x8_src
    y5_tgt = w54 * x4_tgt + w55 * x5_tgt + w56 * x6_tgt + w57 * x7_tgt + w58 * x8_tgt

    w64 = tl.load(wigner_ptr + w_base + 6 * 9 + 4)
    w65 = tl.load(wigner_ptr + w_base + 6 * 9 + 5)
    w66 = tl.load(wigner_ptr + w_base + 6 * 9 + 6)
    w67 = tl.load(wigner_ptr + w_base + 6 * 9 + 7)
    w68 = tl.load(wigner_ptr + w_base + 6 * 9 + 8)
    y6_src = w64 * x4_src + w65 * x5_src + w66 * x6_src + w67 * x7_src + w68 * x8_src
    y6_tgt = w64 * x4_tgt + w65 * x5_tgt + w66 * x6_tgt + w67 * x7_tgt + w68 * x8_tgt

    w74 = tl.load(wigner_ptr + w_base + 7 * 9 + 4)
    w75 = tl.load(wigner_ptr + w_base + 7 * 9 + 5)
    w76 = tl.load(wigner_ptr + w_base + 7 * 9 + 6)
    w77 = tl.load(wigner_ptr + w_base + 7 * 9 + 7)
    w78 = tl.load(wigner_ptr + w_base + 7 * 9 + 8)
    y7_src = w74 * x4_src + w75 * x5_src + w76 * x6_src + w77 * x7_src + w78 * x8_src
    y7_tgt = w74 * x4_tgt + w75 * x5_tgt + w76 * x6_tgt + w77 * x7_tgt + w78 * x8_tgt

    w84 = tl.load(wigner_ptr + w_base + 8 * 9 + 4)
    w85 = tl.load(wigner_ptr + w_base + 8 * 9 + 5)
    w86 = tl.load(wigner_ptr + w_base + 8 * 9 + 6)
    w87 = tl.load(wigner_ptr + w_base + 8 * 9 + 7)
    w88 = tl.load(wigner_ptr + w_base + 8 * 9 + 8)
    y8_src = w84 * x4_src + w85 * x5_src + w86 * x6_src + w87 * x7_src + w88 * x8_src
    y8_tgt = w84 * x4_tgt + w85 * x5_tgt + w86 * x6_tgt + w87 * x7_tgt + w88 * x8_tgt

    # =========================================================================
    # Step 4: L→M permutation and store output
    # L_TO_M_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    # out_m[i] = y_l[L_TO_M_IDX[i]]
    # =========================================================================

    # M=0 ← L=0
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

    # M=1 ← L=2
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

    # M=2 ← L=6
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

    # M=3 ← L=3
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

    # M=4 ← L=7
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

    # M=5 ← L=1
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

    # M=6 ← L=5
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

    # M=7 ← L=8
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

    # M=8 ← L=4
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
