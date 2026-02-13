"""
Wigner D matrices via hybrid approach (fastest method per l).

This module provides Wigner D computation using the optimal method for each l:
- l=0: Trivial (identity)
- l=1: Direct quaternion to rotation matrix (fastest for 3x3)
- l=2: Quaternion einsum tensor contraction (~20x faster on GPU)
- l=3,4: Batched quaternion matmul (single kernel dispatch)
- l>=5: Ra/Rb polynomial (faster than matrix_exp on GPU)

Entry point:
- axis_angle_wigner_hybrid: Main function using real arithmetic throughout

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    WignerCoefficients,
    compute_euler_matching_gamma,
    get_ra_rb_coefficients_real,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_to_ra_rb_real,
    quaternion_y_rotation,
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
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


def wigner_d_from_quaternion_hybrid(
    q: torch.Tensor,
    lmax: int,
    coeffs: Optional[WignerCoefficients] = None,
    U_blocks: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternion using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomial einsum
    - l=3: Quaternion matmul (used when lmax=3)
    - l=3,4: Batched quaternion matmul (used when lmax>=4)
    - l>=5: Ra/Rb polynomial

    Uses real-pair arithmetic throughout for torch.compile compatibility.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        coeffs: Optional pre-computed WignerCoefficients. If provided with U_blocks,
                skips the cache lookup for better performance in hot paths.
        U_blocks: Optional pre-computed U transformation blocks.

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

    # l=3,4: batched matmul when both needed, l=3 only matmul otherwise
    if lmax >= 4:
        D_l3, D_l4 = quaternion_to_wigner_d_l3l4_batched(q)
        D[:, 9:16, 9:16] = D_l3
        D[:, 16:25, 16:25] = D_l4
    elif lmax >= 3:
        D[:, 9:16, 9:16] = quaternion_to_wigner_d_matmul(q, 3)

    # l>=5: Ra/Rb polynomial with real-pair arithmetic
    lmin = 5
    if lmax >= lmin:
        if coeffs is None or U_blocks is None:
            coeffs, U_blocks = get_ra_rb_coefficients_real(
                lmax, dtype, device, lmin=lmin
            )
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
        D_range = wigner_d_pair_to_real(D_re, D_im, U_blocks, lmin=lmin, lmax=lmax)

        block_offset = lmin * lmin  # 25
        D[:, block_offset:, block_offset:] = D_range

    return D


# =============================================================================
# Main Entry Point
# =============================================================================


def axis_angle_wigner_hybrid(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    coeffs: Optional[WignerCoefficients] = None,
    U_blocks: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using hybrid approach (optimal method per l).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion einsum tensor contraction
    - l=3,4: Batched quaternion matmul (single kernel dispatch)
    - l>=5: Ra/Rb polynomial

    Combines the edge->Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Uses real-pair arithmetic throughout for torch.compile compatibility.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
        coeffs: Optional pre-computed WignerCoefficients. If provided with U_blocks,
               skips the cache lookup for better performance in hot paths.
        U_blocks: Optional pre-computed U transformation blocks.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
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

    # Step 3: Compute quaternion (edge -> +Y)
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Compute Wigner D using hybrid approach
    D = wigner_d_from_quaternion_hybrid(
        q_combined,
        lmax,
        coeffs=coeffs,
        U_blocks=U_blocks,
    )

    # Step 6: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
