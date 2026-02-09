"""
Wigner D matrices via matrix exponential of SO(3) generators.

This module provides Wigner D computation using torch.linalg.matrix_exp with
SO(3) Lie algebra generators. Uses specialized optimizations:
- l=0: Trivial (identity)
- l=1: Rodrigues formula (~5x faster than matrix_exp)
- l=2: Quaternion einsum (~20x faster on GPU without compile)
- l=3,4: Quaternion matmul
- l>=5: torch.linalg.matrix_exp

Entry point:
- axis_angle_wigner: Main function for Wigner D from edge vectors

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


# =============================================================================
# Module-Level Caches
# =============================================================================

_L2_COEFF_TENSOR_CACHE: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
_L3_MATMUL_CACHE: dict[tuple[torch.dtype, torch.device], tuple] = {}
_L4_MATMUL_CACHE: dict[tuple[torch.dtype, torch.device], tuple] = {}


def clear_memory_caches() -> None:
    """Clear all in-memory caches for this module."""
    _L2_COEFF_TENSOR_CACHE.clear()
    _L3_MATMUL_CACHE.clear()
    _L4_MATMUL_CACHE.clear()


# =============================================================================
# Specialized Quaternion Kernels for l=2, 3, 4
# =============================================================================


def _build_l2_coefficient_tensor() -> torch.Tensor:
    """
    Build the (5, 5, 4, 4, 4, 4) coefficient tensor for einsum-based l=2 computation.

    The tensor C satisfies: D[i,j] = sum_{a,b,c,d} C[i,j,a,b,c,d] * q[a] * q[b] * q[c] * q[d]
    where q = (w, x, y, z) at indices (0, 1, 2, 3).
    """
    from itertools import permutations

    sqrt3 = math.sqrt(3)
    C = torch.zeros(5, 5, 4, 4, 4, 4, dtype=torch.float64)

    def add_term(i, j, coeff, a, b, c, d):
        """Add coefficient for monomial q[a]*q[b]*q[c]*q[d], symmetrized."""
        perms = set(permutations([a, b, c, d]))
        c_per_perm = coeff / len(perms)
        for p in perms:
            C[i, j, p[0], p[1], p[2], p[3]] += c_per_perm

    W, X, Y, Z = 0, 1, 2, 3

    # D[0,0] = w4 + y4 - x4 - z4 - 6*w2y2 + 6*x2z2
    add_term(0, 0, 1, W, W, W, W)
    add_term(0, 0, 1, Y, Y, Y, Y)
    add_term(0, 0, -1, X, X, X, X)
    add_term(0, 0, -1, Z, Z, Z, Z)
    add_term(0, 0, -6, W, W, Y, Y)
    add_term(0, 0, 6, X, X, Z, Z)

    # D[0,1]
    add_term(0, 1, 2, W, W, W, X)
    add_term(0, 1, 2, W, X, X, X)
    add_term(0, 1, -2, Y, Y, Y, Z)
    add_term(0, 1, -2, Y, Z, Z, Z)
    add_term(0, 1, -6, W, X, Y, Y)
    add_term(0, 1, -6, W, X, Z, Z)
    add_term(0, 1, 6, W, W, Y, Z)
    add_term(0, 1, 6, X, X, Y, Z)

    # D[0,2]
    add_term(0, 2, 4*sqrt3, X, Y, Y, Z)
    add_term(0, 2, 4*sqrt3, W, X, X, Y)
    add_term(0, 2, -4*sqrt3, W, Y, Z, Z)
    add_term(0, 2, -4*sqrt3, W, W, X, Z)

    # D[0,3]
    add_term(0, 3, -2, W, W, W, Z)
    add_term(0, 3, -2, W, Z, Z, Z)
    add_term(0, 3, -2, X, X, X, Y)
    add_term(0, 3, -2, X, Y, Y, Y)
    add_term(0, 3, 6, W, W, X, Y)
    add_term(0, 3, 6, W, X, X, Z)
    add_term(0, 3, 6, W, Y, Y, Z)
    add_term(0, 3, 6, X, Y, Z, Z)

    # D[0,4]
    add_term(0, 4, 4, W, W, W, Y)
    add_term(0, 4, -4, W, Y, Y, Y)
    add_term(0, 4, -4, X, X, X, Z)
    add_term(0, 4, 4, X, Z, Z, Z)

    # D[1,0]
    add_term(1, 0, -2, W, W, W, X)
    add_term(1, 0, -2, W, X, X, X)
    add_term(1, 0, -2, Y, Y, Y, Z)
    add_term(1, 0, -2, Y, Z, Z, Z)
    add_term(1, 0, 6, W, X, Y, Y)
    add_term(1, 0, 6, W, X, Z, Z)
    add_term(1, 0, 6, W, W, Y, Z)
    add_term(1, 0, 6, X, X, Y, Z)

    # D[1,1]
    add_term(1, 1, 1, W, W, W, W)
    add_term(1, 1, -1, Y, Y, Y, Y)
    add_term(1, 1, -1, X, X, X, X)
    add_term(1, 1, 1, Z, Z, Z, Z)
    add_term(1, 1, 6, X, X, Y, Y)
    add_term(1, 1, -6, W, W, Z, Z)

    # D[1,2]
    add_term(1, 2, -2*sqrt3, W, W, W, Z)
    add_term(1, 2, 2*sqrt3, W, Z, Z, Z)
    add_term(1, 2, -2*sqrt3, X, X, X, Y)
    add_term(1, 2, 2*sqrt3, X, Y, Y, Y)
    add_term(1, 2, 2*sqrt3, W, W, X, Y)
    add_term(1, 2, 2*sqrt3, W, X, X, Z)
    add_term(1, 2, -2*sqrt3, W, Y, Y, Z)
    add_term(1, 2, -2*sqrt3, X, Y, Z, Z)

    # D[1,3]
    add_term(1, 3, 2, W, W, W, Y)
    add_term(1, 3, 2, W, Y, Y, Y)
    add_term(1, 3, -2, X, X, X, Z)
    add_term(1, 3, -2, X, Z, Z, Z)
    add_term(1, 3, 6, W, W, X, Z)
    add_term(1, 3, -6, W, X, X, Y)
    add_term(1, 3, 6, X, Y, Y, Z)
    add_term(1, 3, -6, W, Y, Z, Z)

    # D[1,4]
    add_term(1, 4, -2, W, W, W, Z)
    add_term(1, 4, 2, W, Z, Z, Z)
    add_term(1, 4, -2, X, X, X, Y)
    add_term(1, 4, 2, X, Y, Y, Y)
    add_term(1, 4, -6, W, W, X, Y)
    add_term(1, 4, -6, W, X, X, Z)
    add_term(1, 4, 6, W, Y, Y, Z)
    add_term(1, 4, 6, X, Y, Z, Z)

    # D[2,0]
    add_term(2, 0, 4*sqrt3, X, Y, Y, Z)
    add_term(2, 0, -4*sqrt3, W, X, X, Y)
    add_term(2, 0, 4*sqrt3, W, Y, Z, Z)
    add_term(2, 0, -4*sqrt3, W, W, X, Z)

    # D[2,1]
    add_term(2, 1, 2*sqrt3, W, W, W, Z)
    add_term(2, 1, -2*sqrt3, W, Z, Z, Z)
    add_term(2, 1, -2*sqrt3, X, X, X, Y)
    add_term(2, 1, 2*sqrt3, X, Y, Y, Y)
    add_term(2, 1, 2*sqrt3, W, W, X, Y)
    add_term(2, 1, -2*sqrt3, W, X, X, Z)
    add_term(2, 1, 2*sqrt3, W, Y, Y, Z)
    add_term(2, 1, -2*sqrt3, X, Y, Z, Z)

    # D[2,2]
    add_term(2, 2, 1, W, W, W, W)
    add_term(2, 2, 1, X, X, X, X)
    add_term(2, 2, 1, Y, Y, Y, Y)
    add_term(2, 2, 1, Z, Z, Z, Z)
    add_term(2, 2, -4, W, W, X, X)
    add_term(2, 2, 2, W, W, Y, Y)
    add_term(2, 2, -4, W, W, Z, Z)
    add_term(2, 2, -4, X, X, Y, Y)
    add_term(2, 2, 2, X, X, Z, Z)
    add_term(2, 2, -4, Y, Y, Z, Z)

    # D[2,3]
    add_term(2, 3, -2*sqrt3, W, W, W, X)
    add_term(2, 3, 2*sqrt3, W, X, X, X)
    add_term(2, 3, 2*sqrt3, Y, Y, Y, Z)
    add_term(2, 3, -2*sqrt3, Y, Z, Z, Z)
    add_term(2, 3, -2*sqrt3, X, X, Y, Z)
    add_term(2, 3, 2*sqrt3, W, W, Y, Z)
    add_term(2, 3, -2*sqrt3, W, X, Y, Y)
    add_term(2, 3, 2*sqrt3, W, X, Z, Z)

    # D[2,4]
    add_term(2, 4, 2*sqrt3, W, W, X, X)
    add_term(2, 4, -2*sqrt3, W, W, Z, Z)
    add_term(2, 4, -2*sqrt3, X, X, Y, Y)
    add_term(2, 4, 2*sqrt3, Y, Y, Z, Z)
    add_term(2, 4, -8*sqrt3, W, X, Y, Z)

    # D[3,0]
    add_term(3, 0, 2, W, W, W, Z)
    add_term(3, 0, 2, W, Z, Z, Z)
    add_term(3, 0, -2, X, X, X, Y)
    add_term(3, 0, -2, X, Y, Y, Y)
    add_term(3, 0, 6, W, W, X, Y)
    add_term(3, 0, -6, W, X, X, Z)
    add_term(3, 0, -6, W, Y, Y, Z)
    add_term(3, 0, 6, X, Y, Z, Z)

    # D[3,1]
    add_term(3, 1, -2, W, W, W, Y)
    add_term(3, 1, -2, W, Y, Y, Y)
    add_term(3, 1, -2, X, X, X, Z)
    add_term(3, 1, -2, X, Z, Z, Z)
    add_term(3, 1, 6, W, W, X, Z)
    add_term(3, 1, 6, W, X, X, Y)
    add_term(3, 1, 6, X, Y, Y, Z)
    add_term(3, 1, 6, W, Y, Z, Z)

    # D[3,2]
    add_term(3, 2, 2*sqrt3, W, W, W, X)
    add_term(3, 2, -2*sqrt3, W, X, X, X)
    add_term(3, 2, 2*sqrt3, Y, Y, Y, Z)
    add_term(3, 2, -2*sqrt3, Y, Z, Z, Z)
    add_term(3, 2, -2*sqrt3, X, X, Y, Z)
    add_term(3, 2, 2*sqrt3, W, W, Y, Z)
    add_term(3, 2, 2*sqrt3, W, X, Y, Y)
    add_term(3, 2, -2*sqrt3, W, X, Z, Z)

    # D[3,3]
    add_term(3, 3, 1, W, W, W, W)
    add_term(3, 3, 1, X, X, X, X)
    add_term(3, 3, -1, Y, Y, Y, Y)
    add_term(3, 3, -1, Z, Z, Z, Z)
    add_term(3, 3, -6, W, W, X, X)
    add_term(3, 3, 6, Y, Y, Z, Z)

    # D[3,4]
    add_term(3, 4, -2, W, W, W, X)
    add_term(3, 4, 2, W, X, X, X)
    add_term(3, 4, -2, Y, Y, Y, Z)
    add_term(3, 4, 2, Y, Z, Z, Z)
    add_term(3, 4, 6, W, X, Y, Y)
    add_term(3, 4, -6, W, X, Z, Z)
    add_term(3, 4, 6, W, W, Y, Z)
    add_term(3, 4, -6, X, X, Y, Z)

    # D[4,0]
    add_term(4, 0, -4, W, W, W, Y)
    add_term(4, 0, 4, W, Y, Y, Y)
    add_term(4, 0, -4, X, X, X, Z)
    add_term(4, 0, 4, X, Z, Z, Z)

    # D[4,1]
    add_term(4, 1, 2, W, W, W, Z)
    add_term(4, 1, -2, W, Z, Z, Z)
    add_term(4, 1, -2, X, X, X, Y)
    add_term(4, 1, 2, X, Y, Y, Y)
    add_term(4, 1, -6, W, W, X, Y)
    add_term(4, 1, -6, W, Y, Y, Z)
    add_term(4, 1, 6, W, X, X, Z)
    add_term(4, 1, 6, X, Y, Z, Z)

    # D[4,2]
    add_term(4, 2, 2*sqrt3, W, W, X, X)
    add_term(4, 2, -2*sqrt3, W, W, Z, Z)
    add_term(4, 2, -2*sqrt3, X, X, Y, Y)
    add_term(4, 2, 2*sqrt3, Y, Y, Z, Z)
    add_term(4, 2, 8*sqrt3, W, X, Y, Z)

    # D[4,3]
    add_term(4, 3, 2, W, W, W, X)
    add_term(4, 3, -2, W, X, X, X)
    add_term(4, 3, -2, Y, Y, Y, Z)
    add_term(4, 3, 2, Y, Z, Z, Z)
    add_term(4, 3, 6, W, X, Z, Z)
    add_term(4, 3, -6, W, X, Y, Y)
    add_term(4, 3, 6, W, W, Y, Z)
    add_term(4, 3, -6, X, X, Y, Z)

    # D[4,4]
    add_term(4, 4, 1, W, W, W, W)
    add_term(4, 4, 1, X, X, X, X)
    add_term(4, 4, 1, Y, Y, Y, Y)
    add_term(4, 4, 1, Z, Z, Z, Z)
    add_term(4, 4, -6, W, W, Y, Y)
    add_term(4, 4, -6, X, X, Z, Z)

    return C


def _get_l2_coefficient_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get cached l=2 coefficient tensor for einsum computation."""
    key = (dtype, device)
    if key not in _L2_COEFF_TENSOR_CACHE:
        _L2_COEFF_TENSOR_CACHE[key] = _build_l2_coefficient_tensor().to(dtype=dtype, device=device)
    return _L2_COEFF_TENSOR_CACHE[key]


def quaternion_to_wigner_d_l2_einsum(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 5x5 l=2 Wigner D matrix using einsum tensor contraction.

    Expresses D as a tensor contraction:
        D[i,j] = C[i,j,a,b,c,d] * q[a] * q[b] * q[c] * q[d]

    where C is a precomputed (5,5,4,4,4,4) coefficient tensor.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 5, 5) for l=2
    """
    C = _get_l2_coefficient_tensor(q.dtype, q.device)

    # Build q x q, then (q x q) x (q x q) = q x q x q x q
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)  # (N, 4, 4)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)  # (N, 4, 4, 4, 4)

    # Contract with coefficient tensor
    D = torch.einsum('nabcd,ijabcd->nij', q4, C)

    return D


def _derive_matmul_coefficients(ell: int) -> tuple[torch.Tensor, list]:
    """
    Derive Wigner D polynomial coefficients numerically for matmul approach.

    Returns:
        C: (size*size, n_monomials) coefficient matrix
        monomials: list of (a, b, c, d) power tuples
    """
    size = 2 * ell + 1
    degree = 2 * ell

    # Generate all monomials of the given degree
    monomials = []

    def generate_monomials(n_vars, total_degree, current=None):
        if current is None:
            current = []
        if len(current) == n_vars - 1:
            current.append(total_degree - sum(current))
            if current[-1] >= 0:
                monomials.append(tuple(current))
            return
        for i in range(total_degree - sum(current) + 1):
            generate_monomials(n_vars, total_degree, current + [i])

    generate_monomials(4, degree)
    n_monomials = len(monomials)
    n_samples = n_monomials + 50

    # Generate sample quaternions
    torch.manual_seed(42)
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Build monomial matrix
    X = torch.zeros(n_samples, n_monomials, dtype=torch.float64)
    for i, (a, b, c, d) in enumerate(monomials):
        X[:, i] = (q[:, 0] ** a) * (q[:, 1] ** b) * (q[:, 2] ** c) * (q[:, 3] ** d)

    # Compute reference D matrices using matrix_exp
    gens = get_so3_generators(ell, torch.float64, torch.device('cpu'))
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)

    D_ref = torch.zeros(n_samples, size, size, dtype=torch.float64)
    for i in range(n_samples):
        n = axis[i]
        K = n[0] * K_x + n[1] * K_y + n[2] * K_z
        D_ref[i] = torch.linalg.matrix_exp(angle[i] * K)

    # Solve for coefficients via least squares
    coefficients = torch.zeros(size, size, n_monomials, dtype=torch.float64)
    for i in range(size):
        for j in range(size):
            y = D_ref[:, i, j]
            c, _, _, _ = torch.linalg.lstsq(X, y.unsqueeze(1))
            coefficients[i, j, :] = c.squeeze()

    # Reshape to (size*size, n_monomials)
    C = coefficients.view(size * size, n_monomials)

    return C, monomials


def _get_l3_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=3."""
    key = (dtype, device)
    if key not in _L3_MATMUL_CACHE:
        C, monomials = _derive_matmul_coefficients(3)
        _L3_MATMUL_CACHE[key] = (C.to(dtype=dtype, device=device), monomials)
    return _L3_MATMUL_CACHE[key]


def _get_l4_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=4."""
    key = (dtype, device)
    if key not in _L4_MATMUL_CACHE:
        C, monomials = _derive_matmul_coefficients(4)
        _L4_MATMUL_CACHE[key] = (C.to(dtype=dtype, device=device), monomials)
    return _L4_MATMUL_CACHE[key]


def quaternion_to_wigner_d_l3_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 7x7 l=3 Wigner D matrix using matmul.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3
    """
    C, monomials = _get_l3_matmul_data(q.dtype, q.device)
    N = q.shape[0]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute powers up to 6
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w4 * w
    w6 = w4 * w2
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x4 * x2
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y4 * y
    y6 = y4 * y2
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z5 = z4 * z
    z6 = z4 * z2

    powers = {
        0: {0: torch.ones_like(w), 1: w, 2: w2, 3: w3, 4: w4, 5: w5, 6: w6},
        1: {0: torch.ones_like(x), 1: x, 2: x2, 3: x3, 4: x4, 5: x5, 6: x6},
        2: {0: torch.ones_like(y), 1: y, 2: y2, 3: y3, 4: y4, 5: y5, 6: y6},
        3: {0: torch.ones_like(z), 1: z, 2: z2, 3: z3, 4: z4, 5: z5, 6: z6},
    }

    # Build monomial matrix M: (N, n_monomials)
    M = torch.stack([
        powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
        for a, b, c, d in monomials
    ], dim=1)

    # D_flat = M @ C^T: (N, 49)
    D_flat = M @ C.T

    return D_flat.view(N, 7, 7)


def quaternion_to_wigner_d_l4_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 9x9 l=4 Wigner D matrix using matmul.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 9, 9) for l=4
    """
    C, monomials = _get_l4_matmul_data(q.dtype, q.device)
    N = q.shape[0]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute powers up to 8
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w4 * w
    w6 = w4 * w2
    w7 = w4 * w3
    w8 = w4 * w4
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x4 * x2
    x7 = x4 * x3
    x8 = x4 * x4
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y4 * y
    y6 = y4 * y2
    y7 = y4 * y3
    y8 = y4 * y4
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z5 = z4 * z
    z6 = z4 * z2
    z7 = z4 * z3
    z8 = z4 * z4

    powers = {
        0: {0: torch.ones_like(w), 1: w, 2: w2, 3: w3, 4: w4, 5: w5, 6: w6, 7: w7, 8: w8},
        1: {0: torch.ones_like(x), 1: x, 2: x2, 3: x3, 4: x4, 5: x5, 6: x6, 7: x7, 8: x8},
        2: {0: torch.ones_like(y), 1: y, 2: y2, 3: y3, 4: y4, 5: y5, 6: y6, 7: y7, 8: y8},
        3: {0: torch.ones_like(z), 1: z, 2: z2, 3: z3, 4: z4, 5: z5, 6: z6, 7: z7, 8: z8},
    }

    # Build monomial matrix M: (N, n_monomials)
    M = torch.stack([
        powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
        for a, b, c, d in monomials
    ], dim=1)

    # D_flat = M @ C^T: (N, 81)
    D_flat = M @ C.T

    return D_flat.view(N, 9, 9)


# =============================================================================
# Wigner D from Axis-Angle (Batched)
# =============================================================================


def wigner_d_from_axis_angle_batched(
    axis: torch.Tensor,
    angle: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from axis-angle representation.

    D^l = exp(angle * (axis . K^l)) for each l block.
    The l=1 block is transformed to Cartesian basis (x,y,z) for compatibility.

    Uses optimizations:
    - l=0: Trivial (identity)
    - l=1: Rodrigues formula (4-5x faster than matrix_exp)
    - l=2: Quaternion einsum (faster than Cayley-Hamilton)
    - l=3,4: Quaternion matmul (faster than matrix_exp)
    - l>=5: matrix_exp

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
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

    K_x_list = generators['K_x']
    K_y_list = generators['K_y']
    K_z_list = generators['K_z']

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    # Convert axis-angle to quaternion for l=2 einsum if needed
    q = None
    if lmax >= 2:
        half_angle = angle * 0.5
        cos_half = torch.cos(half_angle)
        sin_half = torch.sin(half_angle)
        q = torch.stack([
            cos_half,
            sin_half * axis[:, 0],
            sin_half * axis[:, 1],
            sin_half * axis[:, 2],
        ], dim=-1)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        if ell == 0:
            # l=0 is trivial: 1x1 identity
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # l=1: Use Rodrigues formula for ~5x speedup
            # exp(theta*K) = I + sin(theta)*K + (1-cos(theta))*K^2 for 3x3 rotation
            K_x = K_x_list[1]
            K_y = K_y_list[1]
            K_z = K_z_list[1]

            K = (
                axis[:, 0:1, None, None] * K_x +
                axis[:, 1:2, None, None] * K_y +
                axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            I = torch.eye(3, dtype=dtype, device=device)
            sin_t = torch.sin(angle)[:, None, None]
            cos_t = torch.cos(angle)[:, None, None]
            K2 = torch.bmm(K, K)
            D_ell = I + sin_t * K + (1 - cos_t) * K2

            # Transform from m-ordering (y,z,x) to Cartesian (x,y,z) via index reordering
            # P transforms [y,z,x] -> [x,y,z], which is index permutation [2,0,1]
            D[:, 1:4, 1:4] = D_ell[:, [2, 0, 1], :][:, :, [2, 0, 1]]
        elif ell == 2:
            # l=2: Use quaternion einsum (faster than Cayley-Hamilton)
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)
        elif ell == 3:
            # l=3: Use quaternion matmul (faster than matrix_exp)
            D[:, 9:16, 9:16] = quaternion_to_wigner_d_l3_matmul(q)
        elif ell == 4:
            # l=4: Use quaternion matmul (faster than matrix_exp)
            D[:, 16:25, 16:25] = quaternion_to_wigner_d_l4_matmul(q)
        else:
            # l>=5: Use matrix_exp
            K_x = K_x_list[ell]
            K_y = K_y_list[ell]
            K_z = K_z_list[ell]

            K = (
                axis[:, 0:1, None, None] * K_x +
                axis[:, 1:2, None, None] * K_y +
                axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            D_ell = torch.linalg.matrix_exp(angle[:, None, None] * K)
            D[:, block_start:block_end, block_start:block_end] = D_ell

        block_start = block_end

    return D


# =============================================================================
# Main Entry Point
# =============================================================================


def axis_angle_wigner(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    generators: Optional[dict[str, list[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D from edge vectors using axis-angle representation.

    This approach uses two quaternion charts with NLERP blending to map edge -> +Y,
    which eliminates all singularities (the two charts have complementary singular
    points at +Y and -Y respectively). This avoids both the single-chart Rodrigues
    singularity at -Y and the ZYZ Euler angle singularities at +/-Y.

    The output uses the same real spherical harmonic basis as the Euler-based
    implementation (rotation.py), making this a drop-in replacement.

    Combines edge->Y and gamma rotations into a single quaternion before computing
    the Wigner D, avoiding the overhead of computing two separate Wigner D matrices.

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
            # Use atan2-based gamma to exactly match Euler code output
            # Note: has gradient singularity at edge = +Y
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            # Random gamma for SO(2) equivariance (default for training)
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge -> +Y) using NLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Extract axis-angle from combined quaternion
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached or passed in)
    # These have the Euler transform folded in for l >= 2
    if generators is None:
        generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute single Wigner D from combined rotation via matrix_exp
    # Output is directly in Euler basis thanks to Euler-aligned generators
    D = wigner_d_from_axis_angle_batched(axis, angle, generators, lmax)

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
