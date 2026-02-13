"""
Custom Wigner D computation kernels for l=1, 2, 3, 4.

This module contains specialized, optimized kernels for computing Wigner D
matrices for small angular momentum values:

Primary kernels (recommended for use):
- l=1: quaternion_to_rotation_matrix - direct quaternion to 3x3 rotation
- l=2: quaternion_to_wigner_d_l2_einsum - tensor contraction (~20x faster on GPU)
- l=3,4: quaternion_to_wigner_d_matmul - polynomial coefficient approach

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

# Processed kernel data cache: {(ell, dtype, device): data}
# - l=2: coefficient tensor of shape (5, 5, 4, 4, 4, 4)
# - l=3: (coefficient matrix (49, 84), monomials list)
# - l=4: (coefficient matrix (81, 165), monomials list)
_KERNEL_CACHE: dict[tuple[int, torch.dtype, torch.device], object] = {}

# Batched l=3,4 combined coefficient cache: {(dtype, device): tensor}
_BATCHED_L3L4_CACHE: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}


@functools.lru_cache(maxsize=1)
def _load_coefficients() -> dict:
    """Load precomputed coefficients from file (cached after first load).

    Coefficients are stored in palette-compressed format for smaller file size.
    Each matrix is stored as (palette, indices, shape) and decompressed here.
    """
    raw = torch.load(_COEFFICIENTS_FILE, map_location="cpu", weights_only=True)

    # Decompress palette format
    result = {}
    for ell in [2, 3, 4]:
        key = f"C_l{ell}"
        palette = raw[f"{key}_palette"]
        indices = raw[f"{key}_indices"]
        shape = tuple(raw[f"{key}_shape"].tolist())
        result[key] = palette[indices.long()].reshape(shape)

    return result


def clear_memory_caches() -> None:
    """Clear all in-memory caches for this module."""
    _KERNEL_CACHE.clear()
    _BATCHED_L3L4_CACHE.clear()
    _load_coefficients.cache_clear()


def _get_kernel_data(ell: int, dtype: torch.dtype, device: torch.device) -> object:
    """Get cached kernel data for l=2, 3, or 4.

    Loads coefficient tensor from precomputed file and (for l>=3) generates
    monomials deterministically. Caches by (ell, dtype, device).

    Args:
        ell: Angular momentum (2, 3, or 4)
        dtype: Data type for the coefficients
        device: Device for the tensors

    Returns:
        - For l=2: coefficient tensor of shape (5, 5, 4, 4, 4, 4)
        - For l=3: tuple of (coefficient matrix (49, 84), monomials list of 84 tuples)
        - For l=4: tuple of (coefficient matrix (81, 165), monomials list of 165 tuples)
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
    for ell in (2, 3, 4):
        _get_kernel_data(ell, dtype, device)
    _get_batched_l3l4_kernel_data(dtype, device)


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
# l=3,4 Quaternion Matmul Kernels
# =============================================================================


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
    """Matmul-based Wigner D computation for l=3 or l=4.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        ell: Angular momentum (3 or 4)

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3 or (N, 9, 9) for l=4
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


def _get_batched_l3l4_kernel_data(
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get cached combined coefficient matrix for batched l=3,4 computation.

    Lifts the l=3 degree-6 coefficients to degree-8 by multiplying each
    monomial by |q|^2 = w^2 + x^2 + y^2 + z^2 = 1, then stacks with l=4
    coefficients for a single matmul.

    Args:
        dtype: Data type for the coefficients
        device: Device for the tensors

    Returns:
        Combined coefficient matrix of shape (130, 165) where the first 49
        rows correspond to l=3 and the remaining 81 rows to l=4.
    """
    key = (dtype, device)
    if key not in _BATCHED_L3L4_CACHE:
        # Get existing kernel data
        C_l3, monomials_l3 = _get_kernel_data(3, dtype, device)  # (49, 84), 84 tuples
        C_l4, monomials_l4 = _get_kernel_data(4, dtype, device)  # (81, 165), 165 tuples

        # Build lookup from degree-8 monomial tuple to column index
        mono8_to_idx = {m: i for i, m in enumerate(monomials_l4)}

        # Lift l=3 coefficients from degree-6 to degree-8
        # Each degree-6 monomial (a,b,c,d) maps to four degree-8 monomials:
        #   (a+2,b,c,d), (a,b+2,c,d), (a,b,c+2,d), (a,b,c,d+2)
        # since |q|^2 = w^2 + x^2 + y^2 + z^2 = 1
        n_mono8 = len(monomials_l4)  # 165
        C_l3_lifted = torch.zeros(49, n_mono8, dtype=dtype, device=device)

        for j, (a, b, c, d) in enumerate(monomials_l3):
            # For each degree-6 monomial, distribute its coefficients
            # to the four degree-8 monomials
            lifted = [
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ]
            for mono8 in lifted:
                idx = mono8_to_idx[mono8]
                C_l3_lifted[:, idx] += C_l3[:, j]

        # Stack: first 49 rows = l=3, next 81 rows = l=4
        C_combined = torch.cat([C_l3_lifted, C_l4], dim=0)  # (130, 165)
        _BATCHED_L3L4_CACHE[key] = C_combined

    return _BATCHED_L3L4_CACHE[key]


def quaternion_to_wigner_d_l3l4_batched(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute l=3 and l=4 Wigner D matrices in a single matmul.

    Builds degree-8 monomials once and multiplies by the combined (130, 165)
    coefficient matrix to get both D_l3 and D_l4 from one kernel dispatch.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Tuple of (D_l3, D_l4) with shapes (N, 7, 7) and (N, 9, 9)
    """
    C_combined = _get_batched_l3l4_kernel_data(q.dtype, q.device)
    _, monomials_l4 = _get_kernel_data(4, q.dtype, q.device)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    powers = _precompute_powers(w, x, y, z, 8)

    # Build degree-8 monomial vector: (N, 165)
    M = torch.stack(
        [
            powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
            for a, b, c, d in monomials_l4
        ],
        dim=1,
    )

    # Single matmul: (N, 165) @ (165, 130) -> (N, 130)
    D_flat = M @ C_combined.T

    N = q.shape[0]
    D_l3 = D_flat[:, :49].reshape(N, 7, 7)
    D_l4 = D_flat[:, 49:].reshape(N, 9, 9)

    return D_l3, D_l4
