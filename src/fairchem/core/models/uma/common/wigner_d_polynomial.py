"""
Wigner D matrices via Ra/Rb polynomial formula.

This module provides Wigner D computation using the Ra/Rb Cayley-Klein
decomposition with polynomial evaluation. This is the fastest method on GPU
for large lmax values.

Entry points:
- axis_angle_wigner_polynomial: Main function using complex arithmetic
- axis_angle_wigner_polynomial_real: torch.compile-compatible version using real-pair arithmetic

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    _smooth_step_cinf,
    quaternion_multiply,
    quaternion_nlerp,
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
    wigner_d_complex_to_real,
    wigner_d_pair_to_real,
    precompute_wigner_coefficients,
    precompute_U_blocks_euler_aligned,
    precompute_U_blocks_euler_aligned_real,
)

from fairchem.core.models.uma.common.wigner_d_matexp import (
    quaternion_edge_to_y_stable,
    compute_euler_matching_gamma,
)


# =============================================================================
# Module-Level Caches
# =============================================================================

_RA_RB_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_RA_RB_U_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}
_RA_RB_U_REAL_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}


def clear_memory_caches() -> None:
    """Clear all in-memory caches for this module."""
    _RA_RB_COEFF_CACHE.clear()
    _RA_RB_U_CACHE.clear()
    _RA_RB_U_REAL_CACHE.clear()


# =============================================================================
# Coefficient Caching
# =============================================================================


def _get_ra_rb_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached Ra/Rb polynomial coefficients with Euler-aligned U blocks."""
    key = (lmax, dtype, device)

    if key not in _RA_RB_COEFF_CACHE:
        coeffs = precompute_wigner_coefficients(lmax, dtype=dtype, device=device)
        _RA_RB_COEFF_CACHE[key] = coeffs

    if key not in _RA_RB_U_CACHE:
        U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
        _RA_RB_U_CACHE[key] = U_blocks

    return _RA_RB_COEFF_CACHE[key], _RA_RB_U_CACHE[key]


def _get_ra_rb_coefficients_real(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached Ra/Rb polynomial coefficients with real-pair U blocks."""
    key = (lmax, dtype, device)

    if key not in _RA_RB_COEFF_CACHE:
        coeffs = precompute_wigner_coefficients(lmax, dtype=dtype, device=device)
        _RA_RB_COEFF_CACHE[key] = coeffs

    if key not in _RA_RB_U_REAL_CACHE:
        U_blocks_real = precompute_U_blocks_euler_aligned_real(lmax, dtype=dtype, device=device)
        _RA_RB_U_REAL_CACHE[key] = U_blocks_real

    return _RA_RB_COEFF_CACHE[key], _RA_RB_U_REAL_CACHE[key]


# =============================================================================
# Quaternion Y-Rotation
# =============================================================================


def quaternion_y_rotation(gamma: torch.Tensor) -> torch.Tensor:
    """
    Create quaternion for rotation about Y-axis by angle gamma.

    Args:
        gamma: Rotation angles of shape (N,)

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    half_gamma = gamma / 2
    w = torch.cos(half_gamma)
    x = torch.zeros_like(gamma)
    y = torch.sin(half_gamma)
    z = torch.zeros_like(gamma)
    return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# Polynomial Wigner D Computation
# =============================================================================


def wigner_d_from_quaternion_polynomial(
    q: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternions using Ra/Rb polynomial.

    This is faster than matrix_exp on GPU, especially for higher lmax.
    Uses the same algorithm as wigner_d_quaternion.py but takes the quaternion
    directly (computed by axis_angle's NLERP-blended two-chart approach).

    Output is in Euler-aligned basis with l=1 Cartesian permutation and l>=2
    Euler basis transformation folded into the U blocks.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    dtype = q.dtype
    device = q.device

    coeffs, U_blocks = _get_ra_rb_coefficients(lmax, dtype, device)

    Ra, Rb = quaternion_to_ra_rb(q)
    D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)
    D_real = wigner_d_complex_to_real(D_complex, U_blocks, lmax)

    return D_real


def wigner_d_from_quaternion_polynomial_real(
    q: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternions using real-pair arithmetic.

    This is a torch.compile-compatible version of wigner_d_from_quaternion_polynomial
    that avoids complex tensor operations.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    dtype = q.dtype
    device = q.device

    coeffs, U_blocks_real = _get_ra_rb_coefficients_real(lmax, dtype, device)

    ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
    D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
    D_real = wigner_d_pair_to_real(D_re, D_im, U_blocks_real, lmax)

    return D_real


# =============================================================================
# Main Entry Points
# =============================================================================


def axis_angle_wigner_polynomial(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using Ra/Rb polynomial (GPU-optimized version).

    This is a GPU-optimized alternative to axis_angle_wigner that uses
    the Ra/Rb polynomial formula instead of matrix_exp. It's ~1.5-2x faster
    on GPU for lmax >= 4.

    Uses the same NLERP-blended two-chart quaternion approach as axis_angle_wigner
    to handle singularities correctly. Combines edge->Y and gamma rotations into
    a single quaternion before computing the Wigner D.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype
    N = edge_distance_vec.shape[0]

    # Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Compute gamma if not provided
    if gamma is None:
        gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Compute quaternion (edge -> +Y) using NLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Create Y-rotation quaternion and combine with edge->Y
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Compute Wigner D using Ra/Rb polynomial
    D = wigner_d_from_quaternion_polynomial(q_combined, lmax)

    # Return D and its inverse (transpose for orthogonal matrices)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


def axis_angle_wigner_polynomial_real(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using real-pair Ra/Rb polynomial (torch.compile compatible).

    This is the torch.compile-compatible version of axis_angle_wigner_polynomial
    that uses real-pair arithmetic to avoid graph breaks from complex operations.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype
    N = edge_distance_vec.shape[0]

    # Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Compute gamma if not provided
    if gamma is None:
        gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Compute quaternion (edge -> +Y) using NLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Create Y-rotation quaternion and combine with edge->Y
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Compute Wigner D using real-pair Ra/Rb polynomial
    D = wigner_d_from_quaternion_polynomial_real(q_combined, lmax)

    # Return D and its inverse (transpose for orthogonal matrices)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
