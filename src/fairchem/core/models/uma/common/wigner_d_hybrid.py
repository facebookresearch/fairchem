"""
Wigner D matrices via hybrid approach (fastest method per l).

This module provides Wigner D computation using the optimal method for each l:
- l=0: Trivial (identity)
- l=1: Direct quaternion to rotation matrix (fastest for 3x3)
- l=2: Quaternion einsum tensor contraction (~20x faster on GPU)
- l=3,4: Quaternion matmul
- l>=5: Ra/Rb polynomial (faster than matrix_exp on GPU)

Entry points:
- axis_angle_wigner_hybrid: Main function using complex arithmetic
- axis_angle_wigner_hybrid_real: torch.compile-compatible version using real-pair arithmetic

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
    get_ra_rb_coefficients,
    get_ra_rb_coefficients_real,
    get_so3_generators,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    quaternion_y_rotation,
    wigner_d_complex_to_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
)

from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_l3_matmul,
    quaternion_to_wigner_d_l4_matmul,
)


# =============================================================================
# Hybrid Wigner D Computation
# =============================================================================


def wigner_d_from_axis_angle_hybrid(
    axis: torch.Tensor,
    angle: torch.Tensor,
    q: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
    l3_l4_kernel: bool=False,
) -> torch.Tensor:
    """
    Compute Wigner D matrices using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomials (faster backward pass)
    - l=3,4: if l3_l4_kernel is True Quaternion matmul (faster than Ra/Rb for small l, but only when using torch.compile)
    - l>=5: Ra/Rb polynomial from quaternion (faster than matrix_exp on GPU)

    The caller should pass Euler-aligned generators for l>=2.

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    size = (lmax + 1) ** 2

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    lmin = 5 if l3_l4_kernel else 3

    # Compute l=0, l=1, l=2 using direct quaternion methods (all real arithmetic)
    for ell in range(min(lmax + 1, lmin)):
        if ell == 0:
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # Direct quaternion to rotation matrix (already real)
            D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
        elif ell == 2:
            # Direct quaternion to Wigner D l=2 using einsum tensor contraction
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)
        # Compute l=3,4 using quaternion matmul
        elif l3_l4_kernel and ell == 3:
            D[:, 9:16, 9:16] = quaternion_to_wigner_d_l3_matmul(q)
        elif l3_l4_kernel and ell == 4:
            D[:, 16:25, 16:25] = quaternion_to_wigner_d_l4_matmul(q)

    # Compute l>=5 using Ra/Rb polynomial from quaternion (range version)
    if lmax >= lmin:
        # Get Ra/Rb coefficients for l>=5 only (more efficient)
        coeffs_range, U_blocks_range = get_ra_rb_coefficients(lmax, dtype, device, lmin=lmin)
        Ra, Rb = quaternion_to_ra_rb(q)
        D_complex_range = wigner_d_matrix_complex(Ra, Rb, coeffs_range)
        D_ra_rb_range = wigner_d_complex_to_real(D_complex_range, U_blocks_range, lmin=lmin, lmax=lmax)

        # D_ra_rb_range is already just the l>=3 or 5 blocks
        # Copy l>=3 or 5 blocks directly from the range result
        block_offset = 25 if l3_l4_kernel else 9 # Skip l=0,1,2,3,4 in full matrix (1 + 3 + 5 + 7 + 9 = 25)
        D[:, block_offset:, block_offset:] = D_ra_rb_range

    return D


def wigner_d_from_axis_angle_hybrid_real(
    axis: torch.Tensor,
    angle: torch.Tensor,
    q: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
    l3_l4_kernel: bool=False,
) -> torch.Tensor:
    """
    Compute Wigner D matrices using hybrid approach with real-pair arithmetic.

    This is the torch.compile-compatible version of wigner_d_from_axis_angle_hybrid
    that uses real-pair arithmetic for l>=3 to avoid complex tensor operations.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomials (faster backward pass)
    - l=3,4: if l3_l4_kernel is True Quaternion matmul (faster than Ra/Rb for small l, but only when using torch.compile)
    - l>=5: Ra/Rb polynomial with real-pair arithmetic (torch.compile compatible)

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    size = (lmax + 1) ** 2

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    lmin = 5 if l3_l4_kernel else 3

    # Compute l=0, l=1, l=2 using direct quaternion methods (all real arithmetic)
    for ell in range(min(lmax + 1, lmin)):
        if ell == 0:
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # Direct quaternion to rotation matrix (already real)
            D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
        elif ell == 2:
            # Direct quaternion to Wigner D l=2 using einsum tensor contraction
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)
        # Compute l=3,4 using quaternion matmul
        elif l3_l4_kernel and ell == 3:
            D[:, 9:16, 9:16] = quaternion_to_wigner_d_l3_matmul(q)
        elif l3_l4_kernel and ell == 4:
            D[:, 16:25, 16:25] = quaternion_to_wigner_d_l4_matmul(q)


    # Compute l>=5 using Ra/Rb polynomial with real-pair arithmetic
    if lmax >= lmin:
        # Get Ra/Rb coefficients for l>=5 only with real U blocks
        coeffs_range, U_blocks_range_real = get_ra_rb_coefficients_real(lmax, dtype, device, lmin=lmin)
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re_range, D_im_range = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs_range)
        D_ra_rb_range = wigner_d_pair_to_real(D_re_range, D_im_range, U_blocks_range_real, lmin=lmin, lmax=lmax)

        # Copy l>=3 or 5 blocks directly from the range result
        block_offset = 25 if l3_l4_kernel else 9 # Skip l=0,1,2,3,4 in full matrix (1 + 3 + 5 + 7 + 9 = 25)
        D[:, block_offset:, block_offset:] = D_ra_rb_range

    return D


# =============================================================================
# Main Entry Points
# =============================================================================


def axis_angle_wigner_hybrid(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    generators: Optional[dict[str, list[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using hybrid approach (Rodrigues/Cayley-Hamilton + Ra/Rb).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Cayley-Hamilton from axis-angle with Euler-aligned generators
    - l>=3: Ra/Rb polynomial from quaternion with Euler-aligned U blocks

    Combines the edge->Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Uses Euler-aligned generators (l=2) and U blocks (l>=3) that have the
    Euler basis transformation folded in, eliminating the separate transform step.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
        generators: Optional pre-computed SO(3) generators from get_so3_generators().
               If None, generators are fetched internally (may cause torch.compile
               graph breaks). For optimal torch.compile performance, pre-compute
               generators and pass them here.

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
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    # q_combined = q_gamma * q_edge_to_y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Extract axis-angle from combined quaternion (needed for l=2)
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached or passed in)
    # These have the Euler transform folded in for l=2
    if generators is None:
        generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute Wigner D using hybrid approach
    D = wigner_d_from_axis_angle_hybrid(axis, angle, q_combined, generators, lmax)

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


def axis_angle_wigner_hybrid_real(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    generators: Optional[dict[str, list[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using hybrid approach with real-pair arithmetic (torch.compile compatible).

    This is the torch.compile-compatible version of axis_angle_wigner_hybrid that
    uses real-pair arithmetic for l>=3 to avoid graph breaks from complex operations.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomials
    - l>=3: Ra/Rb polynomial with real-pair arithmetic (no complex tensors)

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
        generators: Optional pre-computed SO(3) generators from get_so3_generators().
               If None, generators are fetched internally.

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

    # Step 5: Extract axis-angle from combined quaternion (needed for generators, but not used here)
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached or passed in)
    if generators is None:
        generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute Wigner D using hybrid approach with real-pair arithmetic
    D = wigner_d_from_axis_angle_hybrid_real(axis, angle, q_combined, generators, lmax)

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
