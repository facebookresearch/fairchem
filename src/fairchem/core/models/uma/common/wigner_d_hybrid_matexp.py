"""
Wigner D matrices via hybrid approach with matrix exponential fallback.

This module provides Wigner D computation using the optimal method for each l:
- l=0: Trivial (identity)
- l=1: Direct quaternion to rotation matrix (fastest for 3x3)
- l=2: Quaternion einsum tensor contraction (~20x faster on GPU)
- l=3,4: Batched quaternion matmul (single kernel dispatch)
- l>=5: torch.linalg.matrix_exp

Entry point:
- axis_angle_wigner_hybrid_matexp: Main function for Wigner D from edge vectors

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    compute_euler_matching_gamma,
    get_so3_generators,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_y_rotation,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_l3l4_batched,
    quaternion_to_wigner_d_matmul,
)

# =============================================================================
# Hybrid Wigner D Computation
# =============================================================================


def wigner_d_from_quaternion_hybrid_matexp(
    q: torch.Tensor,
    lmax: int,
    generators: dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternion using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomial einsum
    - l=3,4: Batched quaternion matmul (single kernel dispatch)
    - l>=5: matrix_exp (converts to axis-angle for these blocks only)

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = q.shape[0]
    device = q.device
    dtype = q.dtype
    size = (lmax + 1) ** 2

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    # l=0: identity
    D[:, 0, 0] = 1.0

    # l=1: direct quaternion to rotation matrix
    if lmax >= 1:
        D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)

    # l=2: einsum tensor contraction
    if lmax >= 2:
        D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)

    # l=3,4: batched matmul (single kernel dispatch for both)
    if lmax >= 4:
        D_l3, D_l4 = quaternion_to_wigner_d_l3l4_batched(q)
        D[:, 9:16, 9:16] = D_l3
        D[:, 16:25, 16:25] = D_l4
    elif lmax >= 3:
        D[:, 9:16, 9:16] = quaternion_to_wigner_d_matmul(q, 3)

    # l>=5: matrix_exp (convert to axis-angle for these blocks only)
    if lmax >= 5:
        axis, angle = quaternion_to_axis_angle(q)

        K_x_list = generators["K_x"]
        K_y_list = generators["K_y"]
        K_z_list = generators["K_z"]

        block_start = 25
        for ell in range(5, lmax + 1):
            block_size = 2 * ell + 1
            block_end = block_start + block_size

            K_x = K_x_list[ell]
            K_y = K_y_list[ell]
            K_z = K_z_list[ell]

            K = (
                axis[:, 0:1, None, None] * K_x
                + axis[:, 1:2, None, None] * K_y
                + axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            D_ell = torch.linalg.matrix_exp(angle[:, None, None] * K)
            D[:, block_start:block_end, block_start:block_end] = D_ell

            block_start = block_end

    return D


# =============================================================================
# Main Entry Point
# =============================================================================


def axis_angle_wigner_hybrid_matexp(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    generators: Optional[dict[str, list[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D from edge vectors using hybrid approach with matexp fallback.

    Uses quaternion kernels for l=1-4 and matrix exponential for l>=5.
    The quaternion is used directly for l=1-4 (no axis-angle conversion),
    and converted to axis-angle only for the l>=5 matrix_exp path.

    Uses Euler-aligned generators that have the Euler basis transformation folded in,
    so the matrix exponential output is directly in Euler basis.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
               Note: this introduces gradient singularity at edge = +Y.
        generators: Optional pre-computed SO(3) generators from get_so3_generators().
               If None, generators are fetched internally (may cause torch.compile
               graph breaks). For optimal torch.compile performance, pre-compute
               generators and pass them here.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
        - wigner_edge_to_y: the edge -> +Y rotation (D @ edge = +Y for l=1)
        - wigner_y_to_edge: the +Y -> edge rotation (D @ +Y = edge for l=1)
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        if use_euler_gamma:
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge -> +Y) using NLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Get Euler-aligned generators (cached or passed in)
    # These have the Euler transform folded in for l >= 2
    if generators is None:
        generators = get_so3_generators(lmax, dtype, device)

    # Step 6: Compute Wigner D using hybrid approach (quaternion for l=1-4, matexp for l>=5)
    D = wigner_d_from_quaternion_hybrid_matexp(
        q_combined, lmax, generators
    )

    # Step 7: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
