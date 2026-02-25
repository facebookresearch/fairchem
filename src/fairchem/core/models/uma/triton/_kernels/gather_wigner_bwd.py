"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Import the Triton forward kernels
from .gather_wigner_fwd import (
    fused_edge_gather_wigner_l2m_lmax2,
    fused_edge_gather_wigner_l2m_lmax2_emit,
)
from .wigner_transform import M_TO_L_GATHER_IDX

# M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
# This is the inverse of L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]


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


def fused_wigner_backward_scatter_add(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """V2 Triton backward: two-phase approach avoiding atomic contention.

    Phase 1: Triton kernel computes M→L + W^T @ grad, writes to [E, 9, 2C]
    Phase 2: PyTorch index_add_ scatters to nodes (uses segment reduction)

    This is ~2x faster than the atomic-based approach for high edge counts.

    Args:
        grad_output: Gradient from downstream [E, 9, 2C] in M-major order
        edge_index: Edge indices [2, E]
        wigner: Per-edge Wigner matrices [E, 81] or [E, 9, 9]
        num_nodes: Number of nodes N

    Returns:
        grad_x: Gradient w.r.t. input x [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    wigner_flat = wigner_flat.contiguous()

    # Phase 1: Compute per-edge gradients (no scatter)
    grad_edge = torch.empty(
        num_edges,
        9,
        2 * sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    BLOCK_C = triton.next_power_of_2(sphere_channels)

    wigner_transform_bwd_kernel[(num_edges,)](
        grad_output,
        wigner_flat,
        grad_edge,
        num_edges,
        sphere_channels,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_edge.stride(0),
        grad_edge.stride(1),
        grad_edge.stride(2),
        BLOCK_C=BLOCK_C,
    )

    # Phase 2: Scatter using PyTorch's optimized index_add_
    # This uses segment reduction instead of atomics
    grad_x = torch.zeros(
        num_nodes,
        9,
        sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    # Flatten for index_add_: [E, 9, C] -> [E, 9*C]
    grad_src = grad_edge[:, :, :sphere_channels].reshape(num_edges, -1)  # [E, 9*C]
    grad_tgt = grad_edge[:, :, sphere_channels:].reshape(num_edges, -1)  # [E, 9*C]
    grad_x_flat = grad_x.view(num_nodes, -1)  # [N, 9*C]

    # Scatter add: much faster than atomics due to segment reduction
    grad_x_flat.index_add_(0, edge_index[0], grad_src)
    grad_x_flat.index_add_(0, edge_index[1], grad_tgt)

    return grad_x


# =============================================================================
# Triton Backward Kernel (original with atomics) - kept for comparison
# =============================================================================


@triton.jit
def fused_wigner_scatter_bwd_kernel(
    grad_out_ptr,  # [E, 9, 2C] gradient from downstream (M-major)
    edge_index_ptr,  # [2, E] edge indices
    wigner_ptr,  # [E, 81] Wigner matrices (flattened 9x9)
    grad_x_ptr,  # [N, 9, C] output gradient (accumulated via atomics)
    num_edges,
    num_nodes,
    sphere_channels,
    grad_stride_e,
    grad_stride_l,
    grad_stride_c,
    edge_stride,
    grad_x_stride_n,
    grad_x_stride_l,
    grad_x_stride_c,
    BLOCK_C: tl.constexpr,
):
    """Triton backward kernel: M→L + W^T @ grad + scatter.

    Each program handles one edge, computes the gradient contribution,
    and atomically adds to the source and target nodes.

    M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    Means: L-pos i gets value from M-pos M_TO_L_GATHER_IDX[i]
      L=0 <- M=0, L=1 <- M=5, L=2 <- M=1, L=3 <- M=3, L=4 <- M=8,
      L=5 <- M=6, L=6 <- M=2, L=7 <- M=4, L=8 <- M=7
    """
    edge_id = tl.program_id(0)
    if edge_id >= num_edges:
        return

    # Load node indices for this edge
    idx0 = tl.load(edge_index_ptr + edge_id).to(tl.int64)
    idx1 = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)

    # Channel vectorization
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    # Wigner and gradient base pointers
    w_base = edge_id * 81
    grad_base = edge_id * grad_stride_e

    # =========================================================================
    # Load gradient (M-major) and apply M→L permutation inline
    # M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    # dy_l[L-pos] = dy_m[M_TO_L[L-pos]]
    # =========================================================================
    # We load from M-major positions and assign to L-major variables

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
    # For transpose: W^T[i,j] = W[j,i], so we swap row/col indices
    # dx_edge = W^T @ dy_l
    # =========================================================================

    # L=0 block: 1x1 scalar at (0,0) - transpose is same
    w00 = tl.load(wigner_ptr + w_base + 0)
    dx0_src = w00 * dy_l0_src
    dx0_tgt = w00 * dy_l0_tgt

    # L=1 block: 3x3 at [1:4, 1:4] - need transpose
    # Original forward: y1 = w11*x1 + w12*x2 + w13*x3
    # Transpose backward: dx1 = w11*dy1 + w21*dy2 + w31*dy3 (column of W becomes row of W^T)
    w11 = tl.load(wigner_ptr + w_base + 1 * 9 + 1)
    w12 = tl.load(wigner_ptr + w_base + 1 * 9 + 2)
    w13 = tl.load(wigner_ptr + w_base + 1 * 9 + 3)
    w21 = tl.load(wigner_ptr + w_base + 2 * 9 + 1)
    w22 = tl.load(wigner_ptr + w_base + 2 * 9 + 2)
    w23 = tl.load(wigner_ptr + w_base + 2 * 9 + 3)
    w31 = tl.load(wigner_ptr + w_base + 3 * 9 + 1)
    w32 = tl.load(wigner_ptr + w_base + 3 * 9 + 2)
    w33 = tl.load(wigner_ptr + w_base + 3 * 9 + 3)

    # W^T @ dy: dx[i] = sum_j W[j,i] * dy[j]
    dx1_src = w11 * dy_l1_src + w21 * dy_l2_src + w31 * dy_l3_src
    dx2_src = w12 * dy_l1_src + w22 * dy_l2_src + w32 * dy_l3_src
    dx3_src = w13 * dy_l1_src + w23 * dy_l2_src + w33 * dy_l3_src

    dx1_tgt = w11 * dy_l1_tgt + w21 * dy_l2_tgt + w31 * dy_l3_tgt
    dx2_tgt = w12 * dy_l1_tgt + w22 * dy_l2_tgt + w32 * dy_l3_tgt
    dx3_tgt = w13 * dy_l1_tgt + w23 * dy_l2_tgt + w33 * dy_l3_tgt

    # L=2 block: 5x5 at [4:9, 4:9] - need transpose
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
    # Atomic scatter add to source node (idx0) and target node (idx1)
    # =========================================================================

    # Source node (idx0) gets first half of channels
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 0 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx0_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 1 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx1_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 2 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx2_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 3 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx3_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 4 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx4_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 5 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx5_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 6 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx6_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 7 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx7_src,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx0 * grad_x_stride_n
        + 8 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx8_src,
        mask=c_mask,
    )

    # Target node (idx1) gets second half of channels
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 0 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx0_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 1 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx1_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 2 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx2_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 3 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx3_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 4 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx4_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 5 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx5_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 6 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx6_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 7 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx7_tgt,
        mask=c_mask,
    )
    tl.atomic_add(
        grad_x_ptr
        + idx1 * grad_x_stride_n
        + 8 * grad_x_stride_l
        + c_range * grad_x_stride_c,
        dx8_tgt,
        mask=c_mask,
    )


def fused_wigner_backward_triton(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Triton backward for fused edge gather + Wigner + L→M.

    Fuses M→L permutation + W^T multiply + scatter add in one kernel.

    Args:
        grad_output: Gradient from downstream [E, 9, 2C] in M-major order
        edge_index: Edge indices [2, E]
        wigner: Per-edge Wigner matrices [E, 81] or [E, 9, 9]
        num_nodes: Number of nodes N

    Returns:
        grad_x: Gradient w.r.t. input x [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    # Flatten wigner if needed
    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    edge_index = edge_index.contiguous()
    wigner_flat = wigner_flat.contiguous()

    # Output buffer (zeros for atomic add)
    grad_x = torch.zeros(
        num_nodes,
        9,
        sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    # Choose BLOCK_C
    BLOCK_C = triton.next_power_of_2(sphere_channels)

    # Launch kernel
    fused_wigner_scatter_bwd_kernel[(num_edges,)](
        grad_output,
        edge_index,
        wigner_flat,
        grad_x,
        num_edges,
        num_nodes,
        sphere_channels,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        edge_index.stride(0),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        BLOCK_C=BLOCK_C,
    )

    return grad_x


# =============================================================================
# PyTorch Reference Backward (for comparison)
# =============================================================================


def _m_to_l_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch M→L permutation for lmax=2."""
    return x[:, M_TO_L_GATHER_IDX, :]


def _scatter_add_edge_grad(
    grad_edge: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Scatter add edge gradients back to nodes."""
    E, num_coeffs, C2 = grad_edge.shape
    C = C2 // 2

    grad_x = torch.zeros(
        num_nodes, num_coeffs, C, device=grad_edge.device, dtype=grad_edge.dtype
    )

    grad_src = grad_edge[:, :, :C]
    grad_tgt = grad_edge[:, :, C:]

    grad_x.index_add_(0, edge_index[0], grad_src)
    grad_x.index_add_(0, edge_index[1], grad_tgt)

    return grad_x


def fused_wigner_backward_pytorch(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """PyTorch reference backward for comparison.

    Implements the same operation as the Triton kernel but using PyTorch ops.
    """
    # Ensure wigner is [E, 9, 9]
    if wigner.ndim == 2:
        wigner_3d = wigner.view(-1, 9, 9)
    else:
        wigner_3d = wigner

    # Step 1: M→L permutation
    grad_output = grad_output.contiguous()
    grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

    # Step 2: W^T @ grad_l
    wigner_t = wigner_3d.transpose(1, 2)  # [E, 9, 9]
    grad_edge = torch.bmm(wigner_t, grad_l)  # [E, 9, 2C]

    # Step 3: Scatter add
    grad_x = _scatter_add_edge_grad(grad_edge, edge_index, num_nodes)

    return grad_x


# =============================================================================
# Autograd Function with Triton Backward
# =============================================================================


class FusedEdgeGatherWignerL2MTritonBwdFunction(torch.autograd.Function):
    """Autograd function with Triton forward AND Triton backward.

    Both forward and backward use Triton kernels for maximum performance.

    IMPORTANT: Now computes grad_wigner to preserve gradient flow through
    positions -> euler angles -> wigner -> output. Without this, forces
    computed via autograd.grad(energy, pos) would be incorrect.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        # Edge gather (before Wigner) - need this for grad_wigner computation
        x_src = x[edge_index[0]]  # [E, 9, C]
        x_tgt = x[edge_index[1]]  # [E, 9, C]
        x_edge = torch.cat([x_src, x_tgt], dim=2)  # [E, 9, 2C]

        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = x.shape[0]

        # Use Triton forward
        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass using Triton kernel.

        Computes:
        1. grad_x via W^T @ grad (scattered back to nodes) using Triton
        2. grad_wigner = M_to_L(grad) @ x_edge^T (block-diagonal outer product)
        """
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: Compute grad_x (scattered back to nodes) using Triton
        grad_x = fused_wigner_backward_triton(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: Compute grad_wigner = M_to_L(grad) @ x_edge^T
        # Using PyTorch for this part (could be Tritonized later for more speed)
        grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

        E, _, C2 = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1): index 0
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3): indices 1,2,3
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :], x_edge[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block (5x5): indices 4,5,6,7,8
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :], x_edge[:, 4:9, :].transpose(1, 2)
        )

        return grad_x, None, grad_wigner


class FusedEdgeGatherWignerL2MPyTorchBwdFunction(torch.autograd.Function):
    """Autograd function with Triton forward and PyTorch backward.

    Forward uses Triton, backward uses PyTorch (legacy behavior).

    IMPORTANT: Now computes grad_wigner to preserve gradient flow through
    positions -> euler angles -> wigner -> output. Without this, forces
    computed via autograd.grad(energy, pos) would be incorrect.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        # Edge gather (before Wigner) - need this for grad_wigner computation
        # x_edge = [x[edge_index[0]], x[edge_index[1]]] concatenated
        x_src = x[edge_index[0]]  # [E, 9, C]
        x_tgt = x[edge_index[1]]  # [E, 9, C]
        x_edge = torch.cat([x_src, x_tgt], dim=2)  # [E, 9, 2C]

        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = x.shape[0]

        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass using PyTorch.

        Computes:
        1. grad_x via W^T @ grad (scattered back to nodes)
        2. grad_wigner = M_to_L(grad) @ x_edge^T (block-diagonal outer product)
        """
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: Compute grad_x (scattered back to nodes)
        grad_x = fused_wigner_backward_pytorch(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: Compute grad_wigner = M_to_L(grad) @ x_edge^T
        # First, M->L permutation on grad_output
        grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

        # Block-diagonal outer product: dW[e] = sum_c grad_l[e,:,c] @ x_edge[e,:,c]^T
        # For block-diagonal structure (blocks: 1x1, 3x3, 5x5):
        # dW = grad_l @ x_edge.transpose(-1,-2), then mask/zero non-block-diagonal
        # But since W is block-diagonal, only block-diagonal parts of dW matter
        E, _, C2 = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1): indices 0
        # dW[0,0] = sum_c grad_l[:,0,c] * x_edge[:,0,c]
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3): indices 1,2,3
        # dW[1:4, 1:4] = grad_l[:,1:4,:] @ x_edge[:,1:4,:].T summed over C
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :],  # [E, 3, 2C]
            x_edge[:, 1:4, :].transpose(1, 2),  # [E, 2C, 3]
        )  # [E, 3, 3]

        # L=2 block (5x5): indices 4,5,6,7,8
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :],  # [E, 5, 2C]
            x_edge[:, 4:9, :].transpose(1, 2),  # [E, 2C, 5]
        )  # [E, 5, 5]

        return grad_x, None, grad_wigner


class FusedEdgeGatherWignerL2MTritonV2BwdFunction(torch.autograd.Function):
    """Autograd function with Triton forward AND Triton V2 backward.

    V2 backward uses two-phase approach:
    - Phase 1: Triton kernel does M→L + W^T transform per-edge (no atomics)
    - Phase 2: PyTorch index_add_ for scatter (segment reduction)

    This avoids atomic contention and should be faster for large edge counts.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        # Edge gather (before Wigner) - need this for grad_wigner computation
        x_src = x[edge_index[0]]  # [E, 9, C]
        x_tgt = x[edge_index[1]]  # [E, 9, C]
        x_edge = torch.cat([x_src, x_tgt], dim=2)  # [E, 9, 2C]

        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = x.shape[0]

        # Use Triton forward
        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass using Triton V2 kernel (two-phase, no atomics)."""
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: Compute grad_x using V2 (two-phase, no atomics)
        grad_x = fused_wigner_backward_scatter_add(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: Compute grad_wigner = M_to_L(grad) @ x_edge^T
        grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

        E, _, _ = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1): index 0
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3): indices 1,2,3
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :], x_edge[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block (5x5): indices 4,5,6,7,8
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :], x_edge[:, 4:9, :].transpose(1, 2)
        )

        return grad_x, None, grad_wigner


class FusedEdgeGatherWignerL2MRecomputeFunction(torch.autograd.Function):
    """Memory-optimized autograd function that recomputes edge features in backward.

    Key optimization: Saves x [N, 9, C] instead of x_edge [E, 9, 2C].

    Memory comparison for typical case (N=2000 nodes, E=74000 edges, C=128):
    - Original (save x_edge): 74000 * 9 * 256 * 4 bytes = 682 MB
    - This version (save x):  2000 * 9 * 128 * 4 bytes = 9 MB
    - Savings: ~670 MB per layer!

    Trade-off: Recomputes x[edge_index[0]] and x[edge_index[1]] in backward.
    This is a memory read (cheap) vs memory write (save_for_backward, expensive).

    Additional optimization: Never materializes the full x_edge tensor.
    Instead, computes grad_wigner as:
        grad_wigner = grad_l_src @ x_src^T + grad_l_tgt @ x_tgt^T
    where grad_l_src = grad_l[:,:,:C] and grad_l_tgt = grad_l[:,:,C:] are views.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel.

        Saves x [N, 9, C] instead of x_edge [E, 9, 2C] for ~98% memory reduction.
        """
        # Save x directly - much smaller than x_edge for typical graphs
        ctx.save_for_backward(x, edge_index, wigner)
        ctx.num_nodes = x.shape[0]

        # Use Triton forward (which does edge gather + wigner + L→M internally)
        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass that recomputes edge features instead of loading saved.

        Computes:
        1. grad_x via W^T @ grad (scattered back to nodes) using Triton
        2. grad_wigner = grad_l_src @ x_src^T + grad_l_tgt @ x_tgt^T
           (never materializes x_edge = cat([x_src, x_tgt], dim=2))
        """
        x, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: Compute grad_x (scattered back to nodes) using Triton
        grad_x = fused_wigner_backward_triton(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: Compute grad_wigner WITHOUT materializing x_edge
        # grad_wigner = M_to_L(grad) @ x_edge^T
        # Since x_edge = [x_src, x_tgt], we can split this as:
        # grad_wigner = grad_l[:,:,:C] @ x_src^T + grad_l[:,:,C:] @ x_tgt^T

        grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

        E = edge_index.shape[1]
        C = x.shape[2]

        # Split grad_l into source and target parts (views, no allocation)
        grad_l_src = grad_l[:, :, :C]  # [E, 9, C] - view
        grad_l_tgt = grad_l[:, :, C:]  # [E, 9, C] - view

        # Recompute edge features by indexing into x
        x_src = x[edge_index[0]]  # [E, 9, C]
        x_tgt = x[edge_index[1]]  # [E, 9, C]

        # Compute grad_wigner block by block
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1): index 0
        # grad_wigner[:,0,0] = sum_c (grad_l_src[:,0,c] * x_src[:,0,c] + grad_l_tgt[:,0,c] * x_tgt[:,0,c])
        grad_wigner[:, 0, 0] = (grad_l_src[:, 0, :] * x_src[:, 0, :]).sum(dim=-1) + (
            grad_l_tgt[:, 0, :] * x_tgt[:, 0, :]
        ).sum(dim=-1)

        # L=1 block (3x3): indices 1,2,3
        # grad_wigner[:,1:4,1:4] = grad_l[:,:,:C] @ x_src^T + grad_l[:,:,C:] @ x_tgt^T
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l_src[:, 1:4, :], x_src[:, 1:4, :].transpose(1, 2)
        ) + torch.bmm(grad_l_tgt[:, 1:4, :], x_tgt[:, 1:4, :].transpose(1, 2))

        # L=2 block (5x5): indices 4,5,6,7,8
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l_src[:, 4:9, :], x_src[:, 4:9, :].transpose(1, 2)
        ) + torch.bmm(grad_l_tgt[:, 4:9, :], x_tgt[:, 4:9, :].transpose(1, 2))

        return grad_x, None, grad_wigner


# =============================================================================
# Fused grad_wigner Triton Kernel — single-launch backward for grad_wigner
# =============================================================================


@triton.jit
def fused_grad_wigner_kernel(
    grad_out_ptr,  # [E, 9, 2C] gradient from downstream (M-major)
    x_ptr,  # [N, 9, C] node features
    edge_index_ptr,  # [2, E] edge indices
    grad_wigner_ptr,  # [E, 35] output (only block-diagonal entries)
    num_edges,
    sphere_channels,
    grad_stride_e,
    grad_stride_l,
    grad_stride_c,
    x_stride_n,
    x_stride_l,
    x_stride_c,
    edge_stride,
    BLOCK_C: tl.constexpr,
):
    """Fused grad_wigner backward kernel with 2D grid.

    Computes grad_wigner = sum_c (grad_l_src * x_src + grad_l_tgt * x_tgt)
    for block-diagonal entries only, with M->L permutation and edge gather
    fused into a single kernel launch.

    Grid: (num_edges, num_c_blocks) -- parallelizes over both edges and
    channel tiles. Each (edge, c_block) program computes partial sums for
    its channel tile and atomically accumulates into the output.

    Output layout: [E, 35] packed as:
        [dw_00, dw_11..dw_33, dw_44..dw_88] (1 + 9 + 25 = 35 entries)
    """
    edge_id = tl.program_id(0)
    c_block_id = tl.program_id(1)

    if edge_id >= num_edges:
        return

    # Load node indices for this edge
    src_node = tl.load(edge_index_ptr + edge_id).to(tl.int64)
    tgt_node = tl.load(edge_index_ptr + edge_stride + edge_id).to(tl.int64)

    # Channel range for this block
    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    grad_base = edge_id * grad_stride_e
    x_src_base = src_node * x_stride_n
    x_tgt_base = tgt_node * x_stride_n

    # =============================================================
    # Load grad_output with M->L permutation (src half: 0..C-1)
    # M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    # =============================================================
    gs0 = tl.load(
        grad_out_ptr + grad_base + 0 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs1 = tl.load(
        grad_out_ptr + grad_base + 5 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs2 = tl.load(
        grad_out_ptr + grad_base + 1 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs3 = tl.load(
        grad_out_ptr + grad_base + 3 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs4 = tl.load(
        grad_out_ptr + grad_base + 8 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs5 = tl.load(
        grad_out_ptr + grad_base + 6 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs6 = tl.load(
        grad_out_ptr + grad_base + 2 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs7 = tl.load(
        grad_out_ptr + grad_base + 4 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gs8 = tl.load(
        grad_out_ptr + grad_base + 7 * grad_stride_l + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    # =============================================================
    # Load grad_output with M->L permutation (tgt half: C..2C-1)
    # =============================================================
    c_tgt = sphere_channels + c_range * grad_stride_c
    gt0 = tl.load(
        grad_out_ptr + grad_base + 0 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt1 = tl.load(
        grad_out_ptr + grad_base + 5 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt2 = tl.load(
        grad_out_ptr + grad_base + 1 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt3 = tl.load(
        grad_out_ptr + grad_base + 3 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt4 = tl.load(
        grad_out_ptr + grad_base + 8 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt5 = tl.load(
        grad_out_ptr + grad_base + 6 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt6 = tl.load(
        grad_out_ptr + grad_base + 2 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt7 = tl.load(
        grad_out_ptr + grad_base + 4 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gt8 = tl.load(
        grad_out_ptr + grad_base + 7 * grad_stride_l + c_tgt,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    # =============================================================
    # Load x[src_node] (L-major, 9 components)
    # =============================================================
    xs0 = tl.load(
        x_ptr + x_src_base + 0 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs1 = tl.load(
        x_ptr + x_src_base + 1 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs2 = tl.load(
        x_ptr + x_src_base + 2 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs3 = tl.load(
        x_ptr + x_src_base + 3 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs4 = tl.load(
        x_ptr + x_src_base + 4 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs5 = tl.load(
        x_ptr + x_src_base + 5 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs6 = tl.load(
        x_ptr + x_src_base + 6 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs7 = tl.load(
        x_ptr + x_src_base + 7 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xs8 = tl.load(
        x_ptr + x_src_base + 8 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    # =============================================================
    # Load x[tgt_node] (L-major, 9 components)
    # =============================================================
    xt0 = tl.load(
        x_ptr + x_tgt_base + 0 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt1 = tl.load(
        x_ptr + x_tgt_base + 1 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt2 = tl.load(
        x_ptr + x_tgt_base + 2 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt3 = tl.load(
        x_ptr + x_tgt_base + 3 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt4 = tl.load(
        x_ptr + x_tgt_base + 4 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt5 = tl.load(
        x_ptr + x_tgt_base + 5 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt6 = tl.load(
        x_ptr + x_tgt_base + 6 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt7 = tl.load(
        x_ptr + x_tgt_base + 7 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    xt8 = tl.load(
        x_ptr + x_tgt_base + 8 * x_stride_l + c_range * x_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    # =============================================================
    # Compute block-diagonal outer products for this channel tile
    # dw[i,j] = sum_c (gs_i * xs_j + gt_i * xt_j)
    # =============================================================

    # L=0 block (1x1): row 0, col 0
    dw_00 = tl.sum(gs0 * xs0) + tl.sum(gt0 * xt0)

    # L=1 block (3x3): rows 1-3, cols 1-3
    dw_11 = tl.sum(gs1 * xs1) + tl.sum(gt1 * xt1)
    dw_12 = tl.sum(gs1 * xs2) + tl.sum(gt1 * xt2)
    dw_13 = tl.sum(gs1 * xs3) + tl.sum(gt1 * xt3)
    dw_21 = tl.sum(gs2 * xs1) + tl.sum(gt2 * xt1)
    dw_22 = tl.sum(gs2 * xs2) + tl.sum(gt2 * xt2)
    dw_23 = tl.sum(gs2 * xs3) + tl.sum(gt2 * xt3)
    dw_31 = tl.sum(gs3 * xs1) + tl.sum(gt3 * xt1)
    dw_32 = tl.sum(gs3 * xs2) + tl.sum(gt3 * xt2)
    dw_33 = tl.sum(gs3 * xs3) + tl.sum(gt3 * xt3)

    # L=2 block (5x5): rows 4-8, cols 4-8
    dw_44 = tl.sum(gs4 * xs4) + tl.sum(gt4 * xt4)
    dw_45 = tl.sum(gs4 * xs5) + tl.sum(gt4 * xt5)
    dw_46 = tl.sum(gs4 * xs6) + tl.sum(gt4 * xt6)
    dw_47 = tl.sum(gs4 * xs7) + tl.sum(gt4 * xt7)
    dw_48 = tl.sum(gs4 * xs8) + tl.sum(gt4 * xt8)
    dw_54 = tl.sum(gs5 * xs4) + tl.sum(gt5 * xt4)
    dw_55 = tl.sum(gs5 * xs5) + tl.sum(gt5 * xt5)
    dw_56 = tl.sum(gs5 * xs6) + tl.sum(gt5 * xt6)
    dw_57 = tl.sum(gs5 * xs7) + tl.sum(gt5 * xt7)
    dw_58 = tl.sum(gs5 * xs8) + tl.sum(gt5 * xt8)
    dw_64 = tl.sum(gs6 * xs4) + tl.sum(gt6 * xt4)
    dw_65 = tl.sum(gs6 * xs5) + tl.sum(gt6 * xt5)
    dw_66 = tl.sum(gs6 * xs6) + tl.sum(gt6 * xt6)
    dw_67 = tl.sum(gs6 * xs7) + tl.sum(gt6 * xt7)
    dw_68 = tl.sum(gs6 * xs8) + tl.sum(gt6 * xt8)
    dw_74 = tl.sum(gs7 * xs4) + tl.sum(gt7 * xt4)
    dw_75 = tl.sum(gs7 * xs5) + tl.sum(gt7 * xt5)
    dw_76 = tl.sum(gs7 * xs6) + tl.sum(gt7 * xt6)
    dw_77 = tl.sum(gs7 * xs7) + tl.sum(gt7 * xt7)
    dw_78 = tl.sum(gs7 * xs8) + tl.sum(gt7 * xt8)
    dw_84 = tl.sum(gs8 * xs4) + tl.sum(gt8 * xt4)
    dw_85 = tl.sum(gs8 * xs5) + tl.sum(gt8 * xt5)
    dw_86 = tl.sum(gs8 * xs6) + tl.sum(gt8 * xt6)
    dw_87 = tl.sum(gs8 * xs7) + tl.sum(gt8 * xt7)
    dw_88 = tl.sum(gs8 * xs8) + tl.sum(gt8 * xt8)

    # =================================================================
    # Atomically accumulate partial sums into [E, 35] output
    # Each edge has at most num_c_blocks writers, contention is minimal.
    # =================================================================
    out_base = edge_id * 35

    # L=0 block
    tl.atomic_add(grad_wigner_ptr + out_base + 0, dw_00)

    # L=1 block (3x3) — packed at offsets 1..9
    tl.atomic_add(grad_wigner_ptr + out_base + 1, dw_11)
    tl.atomic_add(grad_wigner_ptr + out_base + 2, dw_12)
    tl.atomic_add(grad_wigner_ptr + out_base + 3, dw_13)
    tl.atomic_add(grad_wigner_ptr + out_base + 4, dw_21)
    tl.atomic_add(grad_wigner_ptr + out_base + 5, dw_22)
    tl.atomic_add(grad_wigner_ptr + out_base + 6, dw_23)
    tl.atomic_add(grad_wigner_ptr + out_base + 7, dw_31)
    tl.atomic_add(grad_wigner_ptr + out_base + 8, dw_32)
    tl.atomic_add(grad_wigner_ptr + out_base + 9, dw_33)

    # L=2 block (5x5) — packed at offsets 10..34
    tl.atomic_add(grad_wigner_ptr + out_base + 10, dw_44)
    tl.atomic_add(grad_wigner_ptr + out_base + 11, dw_45)
    tl.atomic_add(grad_wigner_ptr + out_base + 12, dw_46)
    tl.atomic_add(grad_wigner_ptr + out_base + 13, dw_47)
    tl.atomic_add(grad_wigner_ptr + out_base + 14, dw_48)
    tl.atomic_add(grad_wigner_ptr + out_base + 15, dw_54)
    tl.atomic_add(grad_wigner_ptr + out_base + 16, dw_55)
    tl.atomic_add(grad_wigner_ptr + out_base + 17, dw_56)
    tl.atomic_add(grad_wigner_ptr + out_base + 18, dw_57)
    tl.atomic_add(grad_wigner_ptr + out_base + 19, dw_58)
    tl.atomic_add(grad_wigner_ptr + out_base + 20, dw_64)
    tl.atomic_add(grad_wigner_ptr + out_base + 21, dw_65)
    tl.atomic_add(grad_wigner_ptr + out_base + 22, dw_66)
    tl.atomic_add(grad_wigner_ptr + out_base + 23, dw_67)
    tl.atomic_add(grad_wigner_ptr + out_base + 24, dw_68)
    tl.atomic_add(grad_wigner_ptr + out_base + 25, dw_74)
    tl.atomic_add(grad_wigner_ptr + out_base + 26, dw_75)
    tl.atomic_add(grad_wigner_ptr + out_base + 27, dw_76)
    tl.atomic_add(grad_wigner_ptr + out_base + 28, dw_77)
    tl.atomic_add(grad_wigner_ptr + out_base + 29, dw_78)
    tl.atomic_add(grad_wigner_ptr + out_base + 30, dw_84)
    tl.atomic_add(grad_wigner_ptr + out_base + 31, dw_85)
    tl.atomic_add(grad_wigner_ptr + out_base + 32, dw_86)
    tl.atomic_add(grad_wigner_ptr + out_base + 33, dw_87)
    tl.atomic_add(grad_wigner_ptr + out_base + 34, dw_88)


# Mapping from packed [35] layout to sparse [9,9] block-diagonal positions.
# Used to unpack kernel output to full Wigner gradient format.
_PACKED_TO_SPARSE_ROW = [
    0,  # L=0
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    3,  # L=1
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,  # L=2
]
_PACKED_TO_SPARSE_COL = [
    0,  # L=0
    1,
    2,
    3,
    1,
    2,
    3,
    1,
    2,
    3,  # L=1
    4,
    5,
    6,
    7,
    8,
    4,
    5,
    6,
    7,
    8,
    4,
    5,
    6,
    7,
    8,
    4,
    5,
    6,
    7,
    8,
    4,
    5,
    6,
    7,
    8,  # L=2
]


def fused_grad_wigner(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute grad_wigner using a fused Triton kernel with 2D grid.

    Fuses M->L permutation + edge gather + block-diagonal outer product.
    Eliminates all intermediate tensors (grad_l, x_src, x_tgt).

    Uses 2D grid (num_edges, num_c_blocks) to parallelize over both
    edges and channel tiles. Each thread block computes partial sums
    for one channel tile and atomically accumulates into a packed [E, 35]
    buffer, which is then unpacked to [E, 81].

    Args:
        grad_output: Gradient from downstream [E, 9, 2C] in M-major order
        x: Node features [N, 9, C]
        edge_index: Edge indices [2, E]

    Returns:
        grad_wigner: [E, 81] (flat), only block-diagonal entries are nonzero
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    grad_output = grad_output.contiguous()
    x = x.contiguous()
    edge_index = edge_index.contiguous()

    # Packed output: only 35 block-diagonal entries per edge
    grad_wigner_packed = torch.zeros(
        num_edges, 35, device=grad_output.device, dtype=torch.float32
    )

    BLOCK_C = 128
    num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C

    fused_grad_wigner_kernel[(num_edges, num_c_blocks)](
        grad_output,
        x,
        edge_index,
        grad_wigner_packed,
        num_edges,
        sphere_channels,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        edge_index.stride(0),
        BLOCK_C=BLOCK_C,
    )

    # Unpack [E, 35] -> [E, 81] sparse format
    grad_wigner_flat = torch.zeros(
        num_edges, 81, device=grad_output.device, dtype=grad_output.dtype
    )
    rows = torch.tensor(_PACKED_TO_SPARSE_ROW, device=grad_output.device)
    cols = torch.tensor(_PACKED_TO_SPARSE_COL, device=grad_output.device)
    sparse_idx = rows * 9 + cols  # [35] indices into flat [81]
    grad_wigner_flat[:, sparse_idx] = grad_wigner_packed.to(grad_output.dtype)

    return grad_wigner_flat


class FusedEdgeGatherWignerL2MAllFusedBwdFunction(torch.autograd.Function):
    """Fully fused autograd function with Triton backward for grad_x and grad_wigner.

    Key optimization: Saves x [N, 9, C] instead of x_edge [E, 9, 2C],
    and computes grad_wigner in a single Triton kernel launch instead of
    ~6 PyTorch kernel launches.

    Memory comparison for typical case (N=2000, E=74000, C=128):
    - Original (save x_edge): 74000 * 9 * 256 * 4 bytes = 682 MB
    - This version (save x):  2000 * 9 * 128 * 4 bytes = 9 MB

    grad_wigner computation:
    - Before: ~6 kernel launches (_m_to_l, 2x gather, 2x bmm, add)
    - After: 1 fused Triton kernel launch, zero intermediates
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel.

        Saves x [N, 9, C] instead of x_edge [E, 9, 2C].
        """
        ctx.save_for_backward(x, edge_index, wigner)
        ctx.num_nodes = x.shape[0]
        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward with fused Triton kernels for both grad_x and grad_wigner.

        Computes:
        1. grad_x via W^T @ grad (scattered to nodes) using Triton
        2. grad_wigner via fused Triton kernel (M->L + gather + outer product)
        """
        x, edge_index, wigner = ctx.saved_tensors

        # Step 1: grad_x using Triton two-phase scatter_add
        grad_x = fused_wigner_backward_scatter_add(
            grad_output, edge_index, wigner, ctx.num_nodes
        )

        # Step 2: grad_wigner using fused Triton kernel
        grad_wigner = fused_grad_wigner(grad_output, x, edge_index)
        grad_wigner = grad_wigner.view_as(wigner)

        return grad_x, None, grad_wigner


class FusedEdgeGatherWignerL2MTritonBwdEmitFunction(torch.autograd.Function):
    """Autograd function using the emit kernel that produces side outputs.

    The forward kernel emits x_edge [E, 9, 2C] (src at [:C], tgt at [C:2C])
    as a side output alongside the main rotated output [E, 9, 2C].
    This eliminates the redundant edge gather and torch.cat that the
    V2 forward does explicitly.

    Backward uses two bmm calls per L-block (K=2C) instead of four
    with K=C. Same total FLOPs, fewer kernel launches.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using emit Triton kernel.

        Single kernel call does gather + Wigner + L→M + side output writes.
        No explicit x[edge_index[0]], x[edge_index[1]], or torch.cat.
        """
        out, x_edge = fused_edge_gather_wigner_l2m_lmax2_emit(x, edge_index, wigner)
        ctx.save_for_backward(x_edge, edge_index, wigner)
        ctx.num_nodes = x.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward using concatenated x_edge [E, 9, 2C].

        Uses two bmm calls per L-block with K=2C instead of four with K=C.
        """
        x_edge, edge_index, wigner = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # Step 1: grad_x via two-phase scatter_add (unchanged)
        grad_x = fused_wigner_backward_scatter_add(
            grad_output, edge_index, wigner, num_nodes
        )

        # Step 2: grad_wigner using concatenated x_edge
        grad_l = _m_to_l_pytorch(grad_output)  # [E, 9, 2C]

        E, _, _ = x_edge.shape
        grad_wigner = torch.zeros(E, 9, 9, device=wigner.device, dtype=wigner.dtype)

        # L=0 block (1x1)
        grad_wigner[:, 0, 0] = (grad_l[:, 0, :] * x_edge[:, 0, :]).sum(dim=-1)

        # L=1 block (3x3)
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            grad_l[:, 1:4, :], x_edge[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block (5x5)
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            grad_l[:, 4:9, :], x_edge[:, 4:9, :].transpose(1, 2)
        )

        return grad_x, None, grad_wigner


class FusedEdgeGatherWignerL2MEmitFwdAllFusedBwdFunction(torch.autograd.Function):
    """Emit forward kernel + all-fused backward.

    Forward: Uses the emit Triton kernel (fused gather+Wigner+L->M with
    x_edge side output). The x_edge side output is discarded -- we save
    x [N,9,C] instead for the backward pass.

    Backward: Uses fused_grad_wigner Triton kernel (same as AllFusedBwd).
    Saves x [N,9,C] (~9 MB) instead of x_edge [E,9,2C] (~920 MB).

    Combines emit's forward speed with all_fused's memory efficiency.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using emit Triton kernel, saving x instead of x_edge."""
        out, _x_edge = fused_edge_gather_wigner_l2m_lmax2_emit(x, edge_index, wigner)
        ctx.save_for_backward(x, edge_index, wigner)
        ctx.num_nodes = x.shape[0]
        return out  # discard _x_edge

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward with fused Triton kernels for both grad_x and grad_wigner."""
        x, edge_index, wigner = ctx.saved_tensors

        grad_x = fused_wigner_backward_scatter_add(
            grad_output, edge_index, wigner, ctx.num_nodes
        )

        grad_wigner = fused_grad_wigner(grad_output, x, edge_index)
        grad_wigner = grad_wigner.view_as(wigner)

        return grad_x, None, grad_wigner


# =============================================================================
# Node-centric backward: CSR utility + Triton kernel + wrapper
# =============================================================================


def _build_csr(edge_index, num_nodes):
    """
    Build CSR row pointers and sorted edge permutation for source
    and target nodes.
    """
    device = edge_index.device
    src = edge_index[0]
    tgt = edge_index[1]

    _, src_perm = torch.sort(src, stable=True)
    src_counts = torch.bincount(src, minlength=num_nodes)
    src_row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    src_row_ptr[1:] = torch.cumsum(src_counts, dim=0)

    _, tgt_perm = torch.sort(tgt, stable=True)
    tgt_counts = torch.bincount(tgt, minlength=num_nodes)
    tgt_row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    tgt_row_ptr[1:] = torch.cumsum(tgt_counts, dim=0)

    return src_perm, src_row_ptr, tgt_perm, tgt_row_ptr


@triton.jit
def _wt_multiply_one_edge(
    grad_out_ptr,
    wigner_ptr,
    edge_id,
    grad_stride_e,
    grad_stride_l,
    grad_stride_c,
    c_range,
    c_mask,
    chan_offset,
):
    """
    Load grad_output with M->L permutation, load Wigner, compute W^T @ dy.
    """
    grad_base = edge_id * grad_stride_e

    # Load grad with M->L: M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    dy0 = tl.load(
        grad_out_ptr
        + grad_base
        + 0 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy1 = tl.load(
        grad_out_ptr
        + grad_base
        + 5 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy2 = tl.load(
        grad_out_ptr
        + grad_base
        + 1 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy3 = tl.load(
        grad_out_ptr
        + grad_base
        + 3 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy4 = tl.load(
        grad_out_ptr
        + grad_base
        + 8 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy5 = tl.load(
        grad_out_ptr
        + grad_base
        + 6 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy6 = tl.load(
        grad_out_ptr
        + grad_base
        + 2 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy7 = tl.load(
        grad_out_ptr
        + grad_base
        + 4 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )
    dy8 = tl.load(
        grad_out_ptr
        + grad_base
        + 7 * grad_stride_l
        + chan_offset
        + c_range * grad_stride_c,
        mask=c_mask,
        other=0.0,
    )

    # Load Wigner block-diagonal entries
    w_base = edge_id * 81
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

    # W^T @ dy: dx[j] = sum_i W[i,j] * dy[i]
    dx0 = w00 * dy0
    dx1 = w11 * dy1 + w21 * dy2 + w31 * dy3
    dx2 = w12 * dy1 + w22 * dy2 + w32 * dy3
    dx3 = w13 * dy1 + w23 * dy2 + w33 * dy3
    dx4 = w44 * dy4 + w54 * dy5 + w64 * dy6 + w74 * dy7 + w84 * dy8
    dx5 = w45 * dy4 + w55 * dy5 + w65 * dy6 + w75 * dy7 + w85 * dy8
    dx6 = w46 * dy4 + w56 * dy5 + w66 * dy6 + w76 * dy7 + w86 * dy8
    dx7 = w47 * dy4 + w57 * dy5 + w67 * dy6 + w77 * dy7 + w87 * dy8
    dx8 = w48 * dy4 + w58 * dy5 + w68 * dy6 + w78 * dy7 + w88 * dy8

    return dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8


@triton.jit
def node_centric_wigner_bwd_kernel(
    grad_out_ptr,
    wigner_ptr,
    grad_x_ptr,
    src_perm_ptr,
    src_row_ptr_ptr,
    tgt_perm_ptr,
    tgt_row_ptr_ptr,
    num_nodes,
    sphere_channels,
    grad_stride_e,
    grad_stride_l,
    grad_stride_c,
    grad_x_stride_n,
    grad_x_stride_l,
    grad_x_stride_c,
    BLOCK_C: tl.constexpr,
):
    """
    Node-centric backward: accumulates W^T @ M_to_L(grad) per node.

    Instead of writing per-edge gradients [E,9,2C] and then scattering,
    iterates over each node's incident edges and accumulates directly
    into grad_x [N,9,C]. Eliminates 682 MB intermediate allocation.

    Grid: (num_nodes, num_c_blocks)
    """
    node_id = tl.program_id(0)
    c_block_id = tl.program_id(1)

    if node_id >= num_nodes:
        return

    c_start = c_block_id * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < sphere_channels

    # Initialize accumulators for 9 L-major coefficients
    acc0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc8 = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Process edges where this node is SOURCE (src half: channels 0..C-1)
    src_start = tl.load(src_row_ptr_ptr + node_id)
    src_end = tl.load(src_row_ptr_ptr + node_id + 1)
    idx = src_start
    while idx < src_end:
        edge_id = tl.load(src_perm_ptr + idx).to(tl.int64)
        dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8 = _wt_multiply_one_edge(
            grad_out_ptr,
            wigner_ptr,
            edge_id,
            grad_stride_e,
            grad_stride_l,
            grad_stride_c,
            c_range,
            c_mask,
            0,
        )
        acc0 += dx0
        acc1 += dx1
        acc2 += dx2
        acc3 += dx3
        acc4 += dx4
        acc5 += dx5
        acc6 += dx6
        acc7 += dx7
        acc8 += dx8
        idx += 1

    # Process edges where this node is TARGET (tgt half: channels C..2C-1)
    tgt_start = tl.load(tgt_row_ptr_ptr + node_id)
    tgt_end = tl.load(tgt_row_ptr_ptr + node_id + 1)
    idx = tgt_start
    while idx < tgt_end:
        edge_id = tl.load(tgt_perm_ptr + idx).to(tl.int64)
        dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8 = _wt_multiply_one_edge(
            grad_out_ptr,
            wigner_ptr,
            edge_id,
            grad_stride_e,
            grad_stride_l,
            grad_stride_c,
            c_range,
            c_mask,
            sphere_channels,
        )
        acc0 += dx0
        acc1 += dx1
        acc2 += dx2
        acc3 += dx3
        acc4 += dx4
        acc5 += dx5
        acc6 += dx6
        acc7 += dx7
        acc8 += dx8
        idx += 1

    # Store accumulated gradient
    out_base = node_id * grad_x_stride_n
    tl.store(
        grad_x_ptr + out_base + 0 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc0,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 1 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc1,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 2 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc2,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 3 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc3,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 4 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc4,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 5 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc5,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 6 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc6,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 7 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc7,
        mask=c_mask,
    )
    tl.store(
        grad_x_ptr + out_base + 8 * grad_x_stride_l + c_range * grad_x_stride_c,
        acc8,
        mask=c_mask,
    )


def node_centric_wigner_backward(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    num_nodes: int,
    src_perm: torch.Tensor,
    src_row_ptr: torch.Tensor,
    tgt_perm: torch.Tensor,
    tgt_row_ptr: torch.Tensor,
) -> torch.Tensor:
    """
    Node-centric backward: no intermediate [E,9,2C] allocation.

    Args:
        grad_output: [E, 9, 2C] gradient (M-major)
        edge_index: [2, E]
        wigner: [E, 81] or [E, 9, 9]
        num_nodes: N
        src_perm: [E] edges sorted by source node
        src_row_ptr: [N+1] CSR row pointers for source
        tgt_perm: [E] edges sorted by target node
        tgt_row_ptr: [N+1] CSR row pointers for target

    Returns:
        grad_x: [N, 9, C]
    """
    num_edges = edge_index.shape[1]
    sphere_channels = grad_output.shape[2] // 2

    wigner_flat = wigner.reshape(num_edges, -1) if wigner.ndim == 3 else wigner
    grad_output = grad_output.contiguous()

    grad_x = torch.zeros(
        num_nodes,
        9,
        sphere_channels,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    BLOCK_C = 128
    num_c_blocks = (sphere_channels + BLOCK_C - 1) // BLOCK_C

    node_centric_wigner_bwd_kernel[(num_nodes, num_c_blocks)](
        grad_output,
        wigner_flat,
        grad_x,
        src_perm,
        src_row_ptr,
        tgt_perm,
        tgt_row_ptr,
        num_nodes,
        sphere_channels,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        BLOCK_C=BLOCK_C,
    )
    return grad_x


class FusedEdgeGatherWignerL2MNodeCentricFunction(torch.autograd.Function):
    """
    Node-centric backward avoids materializing grad_edge [E,9,2C].

    Forward: Uses non-emit Triton kernel (same as AllFusedBwd).
    Backward grad_x: Node-centric Triton kernel iterates over each
        node's incident edges and accumulates W^T @ M_to_L(grad)
        directly into grad_x [N,9,C]. Zero intermediate allocation.
    Backward grad_wigner: Same fused_grad_wigner as AllFusedBwd.

    Memory comparison (N=2000, E=74000, C=128):
    - AllFusedBwd intermediate: 74000 * 9 * 256 * 4 = 682 MB (transient)
    - NodeCentric intermediate: 0 MB
    """

    @staticmethod
    def forward(ctx, x, edge_index, wigner):
        """
        Forward pass using non-emit Triton kernel, building CSR
        for backward.
        """
        # Build CSR for node-centric backward
        src_perm, src_row_ptr, tgt_perm, tgt_row_ptr = _build_csr(
            edge_index, x.shape[0]
        )
        ctx.save_for_backward(
            x,
            edge_index,
            wigner,
            src_perm,
            src_row_ptr,
            tgt_perm,
            tgt_row_ptr,
        )
        ctx.num_nodes = x.shape[0]
        return fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with node-centric grad_x and fused grad_wigner.
        """
        (
            x,
            edge_index,
            wigner,
            src_perm,
            src_row_ptr,
            tgt_perm,
            tgt_row_ptr,
        ) = ctx.saved_tensors

        # grad_x via node-centric kernel (no intermediate allocation)
        grad_x = node_centric_wigner_backward(
            grad_output,
            edge_index,
            wigner,
            ctx.num_nodes,
            src_perm,
            src_row_ptr,
            tgt_perm,
            tgt_row_ptr,
        )

        # grad_wigner via fused Triton kernel (same as AllFusedBwd)
        grad_wigner = fused_grad_wigner(grad_output, x, edge_index)
        grad_wigner = grad_wigner.view_as(wigner)

        return grad_x, None, grad_wigner
