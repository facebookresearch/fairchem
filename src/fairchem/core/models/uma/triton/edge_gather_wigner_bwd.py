"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Import the Triton forward kernel
from .edge_gather_wigner_fwd import fused_edge_gather_wigner_l2m_lmax2
from .wigner_ops import M_TO_L_GATHER_IDX

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
