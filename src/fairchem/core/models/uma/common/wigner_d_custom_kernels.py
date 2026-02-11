"""
Custom Wigner D computation kernels for l=1, 2, 3, 4.

This module contains specialized, optimized kernels for computing Wigner D
matrices for small angular momentum values:

Primary kernels (recommended for use):
- l=1: quaternion_to_rotation_matrix - direct quaternion to 3x3 rotation
- l=2: quaternion_to_wigner_d_l2_einsum - tensor contraction (~20x faster on GPU)
- l=3: quaternion_to_wigner_d_l3_matmul - polynomial coefficient approach
- l=4: quaternion_to_wigner_d_l4_matmul - polynomial coefficient approach

These kernels are used by both wigner_d_matexp.py and wigner_d_hybrid.py
to accelerate the most common angular momentum blocks.

Coefficient matrices are loaded from wigner_d_coefficients.pt at runtime.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# Precomputed coefficients file path
_COEFFICIENTS_FILE = Path(__file__).parent / "wigner_d_coefficients.pt"

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
    _load_coefficients.cache_clear()


def preload_kernel_caches(
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> None:
    """Pre-load all kernel coefficient caches for torch.compile compatibility.

    torch.compile cannot trace through torch.load, so this function must be
    called before compiling any code that uses the Wigner D kernels. Typically
    this should be called during model initialization.

    Args:
        dtype: Data type for the coefficients
        device: Device for the tensors (default: CPU)
    """
    if device is None:
        device = torch.device("cpu")
    _get_l2_coefficient_tensor(dtype, device)
    _get_l3_matmul_data(dtype, device)
    _get_l4_matmul_data(dtype, device)


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
# l=2 Quaternion Einsum Kernel
# =============================================================================


def _get_l2_coefficient_tensor(
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get cached l=2 coefficient tensor for einsum computation.

    Loads from precomputed file (wigner_d_coefficients.pt) and caches by dtype/device.
    """
    key = (dtype, device)
    if key not in _L2_COEFF_TENSOR_CACHE:
        coeffs = _load_coefficients()
        _L2_COEFF_TENSOR_CACHE[key] = coeffs["C_l2"].to(dtype=dtype, device=device)
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
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(
        -3
    )  # (N, 4, 4, 4, 4)

    # Contract with coefficient tensor
    D = torch.einsum("nabcd,ijabcd->nij", q4, C)

    return D


# =============================================================================
# l=3,4 Quaternion Matmul Kernels
# =============================================================================


def _get_l3_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=3.

    Loads coefficient matrix from precomputed file and generates monomials
    deterministically. Caches by dtype/device.
    """
    key = (dtype, device)
    if key not in _L3_MATMUL_CACHE:
        coeffs = _load_coefficients()
        C = coeffs["C_l3"].to(dtype=dtype, device=device)
        monomials = _generate_monomials(4, 6)  # degree 2*ell = 6 for ell=3
        _L3_MATMUL_CACHE[key] = (C, monomials)
    return _L3_MATMUL_CACHE[key]


def _get_l4_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=4.

    Loads coefficient matrix from precomputed file and generates monomials
    deterministically. Caches by dtype/device.
    """
    key = (dtype, device)
    if key not in _L4_MATMUL_CACHE:
        coeffs = _load_coefficients()
        C = coeffs["C_l4"].to(dtype=dtype, device=device)
        monomials = _generate_monomials(4, 8)  # degree 2*ell = 8 for ell=4
        _L4_MATMUL_CACHE[key] = (C, monomials)
    return _L4_MATMUL_CACHE[key]


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


def _quaternion_to_wigner_d_matmul(q: torch.Tensor, ell: int) -> torch.Tensor:
    """Generic matmul-based Wigner D computation for l=3 or l=4.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        ell: Angular momentum (3 or 4)

    Returns:
        Wigner D matrices of shape (N, 2*ell+1, 2*ell+1)
    """
    cache_fn = _get_l3_matmul_data if ell == 3 else _get_l4_matmul_data
    C, monomials = cache_fn(q.dtype, q.device)

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
    return _quaternion_to_wigner_d_matmul(q, 3)


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
    return _quaternion_to_wigner_d_matmul(q, 4)
