"""
Custom Wigner D computation kernels for l=1, 2, 3, 4, 5, 6.

This module contains specialized, optimized kernels for computing Wigner D
matrices for small angular momentum values:

Primary kernels (recommended for use):
- l=1: quaternion_to_rotation_matrix - direct quaternion to 3x3 rotation
- l=2: quaternion_to_wigner_d_l2_einsum - tensor contraction (~20x faster on GPU)
- l=3: quaternion_to_wigner_d_l3_matmul - polynomial coefficient approach
- l=4: quaternion_to_wigner_d_l4_matmul - polynomial coefficient approach
- l=5: quaternion_to_wigner_d_l5_matmul - polynomial coefficient approach
- l=6: quaternion_to_wigner_d_l6_matmul - polynomial coefficient approach

Experimental kernels (maintained for reference/inspiration, not recommended):
- l=1: rodrigues_rotation_l1 - Rodrigues formula from axis-angle
       Slower than quaternion method since it requires axis-angle extraction
- l=2: cayley_hamilton_exp_l2 - Cayley-Hamilton matrix exponential
       Slower than einsum since it requires axis-angle and uses bmm

These kernels are used by both wigner_d_matexp.py and wigner_d_hybrid.py
to accelerate the most common angular momentum blocks.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    get_so3_generators,
    quaternion_to_axis_angle,
)

# Precomputed coefficients file path
_COEFFICIENTS_FILE = Path(__file__).parent / "wigner_d_coefficients.pt"

# =============================================================================
# Module-Level Caches
# =============================================================================

# Processed kernel data cache: {(ell, dtype, device): data}
# - l=2: coefficient tensor of shape (5, 5, 4, 4, 4, 4)
# - l=3: (coefficient matrix (49, 84), monomials list)
# - l=4: (coefficient matrix (81, 165), monomials list)
# - l=5: (coefficient matrix (121, 286), monomials list)
# - l=6: (coefficient matrix (169, 455), monomials list)
_KERNEL_CACHE: dict[tuple[int, torch.dtype, torch.device], object] = {}


def clear_memory_caches() -> None:
    """Clear all in-memory caches for this module."""
    _KERNEL_CACHE.clear()
    _load_coefficients.cache_clear()


@functools.lru_cache(maxsize=1)
def _load_coefficients() -> dict:
    """Load precomputed coefficients from file (cached after first load)."""
    return torch.load(_COEFFICIENTS_FILE, map_location="cpu", weights_only=True)


def _generate_monomials(n_vars: int, total_degree: int) -> list[tuple[int, ...]]:
    """Generate all monomials of given degree in n_vars variables.

    Returns a list of tuples (a, b, c, d) representing w^a * x^b * y^c * z^d
    where a + b + c + d = total_degree.
    """
    monomials: list[tuple[int, ...]] = []

    def generate(remaining_vars: int, remaining_deg: int, current: list[int]) -> None:
        if remaining_vars == 1:
            monomials.append(tuple(current + [remaining_deg]))
            return
        for i in range(remaining_deg + 1):
            generate(remaining_vars - 1, remaining_deg - i, current + [i])

    generate(n_vars, total_degree, [])
    return monomials


def _get_kernel_data(ell: int, dtype: torch.dtype, device: torch.device) -> object:
    """Get cached kernel data for l=2, 3, 4, 5, or 6.

    Loads coefficient tensor from precomputed file and (for l>=3) generates
    monomials deterministically. Caches by (ell, dtype, device).

    Args:
        ell: Angular momentum (2, 3, 4, 5, or 6)
        dtype: Data type for the coefficients
        device: Device for the tensors

    Returns:
        - For l=2: coefficient tensor of shape (5, 5, 4, 4, 4, 4)
        - For l=3: tuple of (coefficient matrix (49, 84), monomials list of 84 tuples)
        - For l=4: tuple of (coefficient matrix (81, 165), monomials list of 165 tuples)
        - For l=5: tuple of (coefficient matrix (121, 286), monomials list of 286 tuples)
        - For l=6: tuple of (coefficient matrix (169, 455), monomials list of 455 tuples)
    """
    key = (ell, dtype, device)
    if key not in _KERNEL_CACHE:
        coeffs = _load_coefficients()
        C = coeffs[f"C_l{ell}"].to(dtype=dtype, device=device)
        if ell == 2:
            _KERNEL_CACHE[key] = C
        else:
            monomials = _generate_monomials(4, 2 * ell)
            _KERNEL_CACHE[key] = (C, monomials)
    return _KERNEL_CACHE[key]


def preload_kernel_caches(
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    max_ell: int = 6,
) -> None:
    """Pre-load all kernel coefficient caches for torch.compile compatibility.

    torch.compile cannot trace through torch.load, so this function must be
    called before compiling any code that uses the Wigner D kernels. Typically
    this should be called during model initialization.

    Args:
        dtype: Data type for the coefficients
        device: Device for the tensors (default: CPU)
        max_ell: Maximum l to preload (default: 6)
    """
    if device is None:
        device = torch.device("cpu")
    for ell in range(2, max_ell + 1):
        _get_kernel_data(ell, dtype, device)


# =============================================================================
# l=1 Quaternion to Rotation Matrix (Primary - Recommended)
# =============================================================================


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion directly to 3x3 rotation matrix (l=1 Wigner D).

    This is the recommended method for l=1 as it uses pure polynomial
    arithmetic without requiring axis-angle extraction or matrix operations.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
        ],
        dim=-2,
    )

    return R


# =============================================================================
# l=1 Rodrigues Formula (Experimental - Not Recommended)
# =============================================================================


def rodrigues_rotation_l1(
    axis: torch.Tensor,
    angle: torch.Tensor,
    K_x: torch.Tensor,
    K_y: torch.Tensor,
    K_z: torch.Tensor,
) -> torch.Tensor:
    """
    Compute l=1 Wigner D (3x3 rotation) using Rodrigues formula from axis-angle.

    WARNING: This method is slower than quaternion_to_rotation_matrix because:
    1. It requires extracting axis-angle from quaternion (expensive trig ops)
    2. It uses bmm for K^2 computation
    3. It requires fetching/broadcasting generator matrices

    This is maintained for reference and inspiration only.

    Uses: exp(theta K) = I + sin(theta)K + (1-cos(theta))K^2

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        K_x, K_y, K_z: SO(3) generator matrices for l=1, shape (3, 3)

    Returns:
        Rotation matrices of shape (N, 3, 3) in m-ordering basis (y,z,x)
        Note: Caller must apply index permutation [2,0,1] for Cartesian basis
    """
    device = axis.device
    dtype = axis.dtype

    K = (
        axis[:, 0:1, None, None] * K_x
        + axis[:, 1:2, None, None] * K_y
        + axis[:, 2:3, None, None] * K_z
    ).squeeze(1)

    I = torch.eye(3, dtype=dtype, device=device)
    sin_t = torch.sin(angle)[:, None, None]
    cos_t = torch.cos(angle)[:, None, None]
    K2 = torch.bmm(K, K)

    return I + sin_t * K + (1 - cos_t) * K2


# =============================================================================
# l=2 Quaternion Einsum Kernel
# =============================================================================


def _build_l2_coefficient_tensor() -> torch.Tensor:
    """
    Build the (5, 5, 4, 4, 4, 4) coefficient tensor for einsum-based l=2 computation.

    NOTE: This function is kept for coefficient regeneration/verification purposes.
    At runtime, coefficients are loaded from wigner_d_coefficients.pt instead.

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
    add_term(0, 2, 4 * sqrt3, X, Y, Y, Z)
    add_term(0, 2, 4 * sqrt3, W, X, X, Y)
    add_term(0, 2, -4 * sqrt3, W, Y, Z, Z)
    add_term(0, 2, -4 * sqrt3, W, W, X, Z)

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
    add_term(1, 2, -2 * sqrt3, W, W, W, Z)
    add_term(1, 2, 2 * sqrt3, W, Z, Z, Z)
    add_term(1, 2, -2 * sqrt3, X, X, X, Y)
    add_term(1, 2, 2 * sqrt3, X, Y, Y, Y)
    add_term(1, 2, 2 * sqrt3, W, W, X, Y)
    add_term(1, 2, 2 * sqrt3, W, X, X, Z)
    add_term(1, 2, -2 * sqrt3, W, Y, Y, Z)
    add_term(1, 2, -2 * sqrt3, X, Y, Z, Z)

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
    add_term(2, 0, 4 * sqrt3, X, Y, Y, Z)
    add_term(2, 0, -4 * sqrt3, W, X, X, Y)
    add_term(2, 0, 4 * sqrt3, W, Y, Z, Z)
    add_term(2, 0, -4 * sqrt3, W, W, X, Z)

    # D[2,1]
    add_term(2, 1, 2 * sqrt3, W, W, W, Z)
    add_term(2, 1, -2 * sqrt3, W, Z, Z, Z)
    add_term(2, 1, -2 * sqrt3, X, X, X, Y)
    add_term(2, 1, 2 * sqrt3, X, Y, Y, Y)
    add_term(2, 1, 2 * sqrt3, W, W, X, Y)
    add_term(2, 1, -2 * sqrt3, W, X, X, Z)
    add_term(2, 1, 2 * sqrt3, W, Y, Y, Z)
    add_term(2, 1, -2 * sqrt3, X, Y, Z, Z)

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
    add_term(2, 3, -2 * sqrt3, W, W, W, X)
    add_term(2, 3, 2 * sqrt3, W, X, X, X)
    add_term(2, 3, 2 * sqrt3, Y, Y, Y, Z)
    add_term(2, 3, -2 * sqrt3, Y, Z, Z, Z)
    add_term(2, 3, -2 * sqrt3, X, X, Y, Z)
    add_term(2, 3, 2 * sqrt3, W, W, Y, Z)
    add_term(2, 3, -2 * sqrt3, W, X, Y, Y)
    add_term(2, 3, 2 * sqrt3, W, X, Z, Z)

    # D[2,4]
    add_term(2, 4, 2 * sqrt3, W, W, X, X)
    add_term(2, 4, -2 * sqrt3, W, W, Z, Z)
    add_term(2, 4, -2 * sqrt3, X, X, Y, Y)
    add_term(2, 4, 2 * sqrt3, Y, Y, Z, Z)
    add_term(2, 4, -8 * sqrt3, W, X, Y, Z)

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
    add_term(3, 2, 2 * sqrt3, W, W, W, X)
    add_term(3, 2, -2 * sqrt3, W, X, X, X)
    add_term(3, 2, 2 * sqrt3, Y, Y, Y, Z)
    add_term(3, 2, -2 * sqrt3, Y, Z, Z, Z)
    add_term(3, 2, -2 * sqrt3, X, X, Y, Z)
    add_term(3, 2, 2 * sqrt3, W, W, Y, Z)
    add_term(3, 2, 2 * sqrt3, W, X, Y, Y)
    add_term(3, 2, -2 * sqrt3, W, X, Z, Z)

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
    add_term(4, 2, 2 * sqrt3, W, W, X, X)
    add_term(4, 2, -2 * sqrt3, W, W, Z, Z)
    add_term(4, 2, -2 * sqrt3, X, X, Y, Y)
    add_term(4, 2, 2 * sqrt3, Y, Y, Z, Z)
    add_term(4, 2, 8 * sqrt3, W, X, Y, Z)

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
    C = _get_kernel_data(2, q.dtype, q.device)

    # Build q x q, then (q x q) x (q x q) = q x q x q x q
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)  # (N, 4, 4)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(
        -3
    )  # (N, 4, 4, 4, 4)

    # Contract with coefficient tensor
    D = torch.einsum("nabcd,ijabcd->nij", q4, C)

    return D


# =============================================================================
# l=2 Cayley-Hamilton Kernel (Experimental - Not Recommended)
# =============================================================================


def cayley_hamilton_exp_l2(K: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(theta K) for 5x5 antisymmetric K using Cayley-Hamilton.

    WARNING: This method is slower than quaternion_to_wigner_d_l2_einsum because:
    1. It requires extracting axis-angle from quaternion (expensive trig ops)
    2. It uses multiple bmm operations for K^2, K^3, K^4
    3. It has complex branching for degenerate eigenvalue handling

    This is maintained for reference and inspiration only.

    For antisymmetric K with eigenvalues 0, ±iλ_1, ±iλ_2:
        exp(theta K) = I + c_1K + c_2K^2 + c_3 K^3 + c_4 K^4

    The coefficients are derived via Lagrange interpolation on the eigenvalues.

    Args:
        K: Antisymmetric matrices of shape (N, 5, 5)
        angle: Rotation angles of shape (N,)

    Returns:
        Rotation matrices exp(theta K) of shape (N, 5, 5)
    """
    device = K.device
    dtype = K.dtype

    I = torch.eye(5, device=device, dtype=dtype).unsqueeze(0)

    # Compute powers of K
    K2 = torch.bmm(K, K)
    K3 = torch.bmm(K2, K)
    K4 = torch.bmm(K2, K2)

    # Extract eigenvalue info from traces
    # For antisymmetric K: eigenvalues are 0, ±iλ_1, ±iλ_2
    # tr(K^2) = -2(λ_1^2 + λ_2^2)
    # tr(K^4) = 2(λ_1^4 + λ_2^4)
    tr_K2 = torch.einsum("nii->n", K2)
    tr_K4 = torch.einsum("nii->n", K4)

    s1 = -tr_K2 / 2  # λ_1^2 + λ_2^2
    s2 = tr_K4 / 2  # λ_1^4 + λ_2^4

    # Solve for sigma_1 = λ_1^2, sigma_2 = λ_2^2
    # Product p = sigma_1 * sigma_2 = (s1^2 - s2) / 2
    p = (s1 * s1 - s2) / 2

    # sigma_1, sigma_2 are roots of t^2 - s1*t + p = 0
    discriminant = (s1 * s1 - 4 * p).clamp(min=0)
    sqrt_disc = torch.sqrt(discriminant)

    sigma1 = ((s1 + sqrt_disc) / 2).clamp(min=0)
    sigma2 = ((s1 - sqrt_disc) / 2).clamp(min=0)

    lambda1 = torch.sqrt(sigma1)
    lambda2 = torch.sqrt(sigma2)

    theta = angle
    eps = 1e-12

    # Compute sinc-like functions:
    # A = (cos(theta λ_1) - 1) / sigma_1
    # B = (cos(theta λ_2) - 1) / sigma_2
    # C = sin(theta λ_1) / λ_1
    # D = sin(theta λ_2) / λ_2

    theta_l1 = theta * lambda1
    theta_l2 = theta * lambda2

    cos_l1 = torch.cos(theta_l1)
    cos_l2 = torch.cos(theta_l2)
    sin_l1 = torch.sin(theta_l1)
    sin_l2 = torch.sin(theta_l2)

    # Safe division for A = (cos(theta λ) - 1) / sigma
    sigma1_safe = torch.where(sigma1 < eps, torch.ones_like(sigma1), sigma1)
    sigma2_safe = torch.where(sigma2 < eps, torch.ones_like(sigma2), sigma2)
    A = torch.where(sigma1 < eps, -theta * theta / 2, (cos_l1 - 1) / sigma1_safe)
    B = torch.where(sigma2 < eps, -theta * theta / 2, (cos_l2 - 1) / sigma2_safe)

    # Safe division for C = sin(theta λ) / λ
    lambda1_safe = torch.where(lambda1 < eps, torch.ones_like(lambda1), lambda1)
    lambda2_safe = torch.where(lambda2 < eps, torch.ones_like(lambda2), lambda2)
    C = torch.where(lambda1 < eps, theta, sin_l1 / lambda1_safe)
    D_coeff = torch.where(lambda2 < eps, theta, sin_l2 / lambda2_safe)

    # Coefficients via Lagrange interpolation:
    #   c_1 = (sigma_2 C - sigma_1 D) / (sigma_2 - sigma_1)
    #   c_2 = (sigma_1 B - sigma_2 A) / (sigma_2 - sigma_1)
    #   c_3 = (C - D) / (sigma_2 - sigma_1)
    #   c_4 = (B - A) / (sigma_2 - sigma_1)

    sigma_diff = sigma2 - sigma1
    degenerate = sigma_diff.abs() < eps
    sigma_diff_safe = torch.where(degenerate, torch.ones_like(sigma_diff), sigma_diff)

    # Non-degenerate case
    c1_nondeg = (sigma2 * C - sigma1 * D_coeff) / sigma_diff_safe
    c2_nondeg = (sigma1 * B - sigma2 * A) / sigma_diff_safe
    c3_nondeg = (C - D_coeff) / sigma_diff_safe
    c4_nondeg = (B - A) / sigma_diff_safe

    # Degenerate case (sigma_1 = sigma_2 = sigma): Rodrigues-like formula
    sigma_common = (sigma1 + sigma2) / 2
    lambda_common = torch.sqrt(sigma_common.clamp(min=0))
    theta_lc = theta * lambda_common
    cos_lc = torch.cos(theta_lc)
    sin_lc = torch.sin(theta_lc)

    lambda_common_safe = torch.where(
        lambda_common < eps, torch.ones_like(lambda_common), lambda_common
    )
    sigma_common_safe = torch.where(
        sigma_common < eps, torch.ones_like(sigma_common), sigma_common
    )

    c1_deg = torch.where(lambda_common < eps, theta, sin_lc / lambda_common_safe)
    c2_deg = torch.where(
        sigma_common < eps, -theta * theta / 2, (cos_lc - 1) / sigma_common_safe
    )
    c3_deg = torch.zeros_like(theta)
    c4_deg = torch.zeros_like(theta)

    # Select based on degeneracy
    c1 = torch.where(degenerate, c1_deg, c1_nondeg)
    c2 = torch.where(degenerate, c2_deg, c2_nondeg)
    c3 = torch.where(degenerate, c3_deg, c3_nondeg)
    c4 = torch.where(degenerate, c4_deg, c4_nondeg)

    # Reshape for broadcasting
    c1 = c1[:, None, None]
    c2 = c2[:, None, None]
    c3 = c3[:, None, None]
    c4 = c4[:, None, None]

    # exp(theta K) = I + c_1 K + c_2 K^2 + c_3 K^3 + c_4 K^4
    return I + c1 * K + c2 * K2 + c3 * K3 + c4 * K4


# =============================================================================
# l=3,4 Quaternion Matmul Kernels
# =============================================================================


def _derive_matmul_coefficients(ell: int) -> tuple[torch.Tensor, list]:
    """
    Derive Wigner D polynomial coefficients numerically for matmul approach.

    NOTE: This function is kept for coefficient regeneration/verification purposes.
    At runtime, coefficients are loaded from wigner_d_coefficients.pt instead.

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
    gens = get_so3_generators(ell, torch.float64, torch.device("cpu"))
    K_x = gens["K_x"][ell]
    K_y = gens["K_y"][ell]
    K_z = gens["K_z"][ell]

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


def _precompute_powers(
    w: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    max_power: int,
) -> dict[int, dict[int, torch.Tensor]]:
    """Precompute powers 0..max_power for quaternion components.

    Uses optimal multiplication tree: p[i] = p[i//2] * p[(i+1)//2]
    which minimizes the number of multiplications.

    Args:
        w, x, y, z: Quaternion components, each of shape (N,)
        max_power: Maximum power to compute

    Returns:
        Dictionary mapping variable index (0=w, 1=x, 2=y, 3=z) to
        a dictionary mapping power to the precomputed tensor.
    """

    def powers_for_var(var: torch.Tensor) -> dict[int, torch.Tensor]:
        p: dict[int, torch.Tensor] = {0: torch.ones_like(var), 1: var}
        for i in range(2, max_power + 1):
            p[i] = p[i // 2] * p[(i + 1) // 2]
        return p

    return {
        0: powers_for_var(w),
        1: powers_for_var(x),
        2: powers_for_var(y),
        3: powers_for_var(z),
    }


def quaternion_to_wigner_d_matmul(q: torch.Tensor, ell: int) -> torch.Tensor:
    """Matmul-based Wigner D computation for l=3, 4, 5, or 6.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        ell: Angular momentum (3, 4, 5, or 6)

    Returns:
        Wigner D matrices of shape (N, 2*ell+1, 2*ell+1)
    """
    C, monomials = _get_kernel_data(ell, q.dtype, q.device)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    powers = _precompute_powers(w, x, y, z, 2 * ell)

    # Build monomial matrix M: (N, n_monomials)
    M = torch.stack(
        [
            powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
            for a, b, c, d in monomials
        ],
        dim=1,
    )

    # D_flat = M @ C^T
    D_flat = M @ C.T
    size = 2 * ell + 1

    return D_flat.view(q.shape[0], size, size)


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
    return quaternion_to_wigner_d_matmul(q, 3)


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
    return quaternion_to_wigner_d_matmul(q, 4)


def quaternion_to_wigner_d_l5_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 11x11 l=5 Wigner D matrix using matmul.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 11, 11) for l=5
    """
    return quaternion_to_wigner_d_matmul(q, 5)


def quaternion_to_wigner_d_l6_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 13x13 l=6 Wigner D matrix using matmul.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 13, 13) for l=6
    """
    return quaternion_to_wigner_d_matmul(q, 6)
