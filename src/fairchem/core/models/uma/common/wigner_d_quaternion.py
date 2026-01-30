"""
Quaternion-based Wigner D Matrix Construction (Clean Implementation)

This module implements Wigner D matrix computation using quaternions, following
the spherical_functions package by Mike Boyle (https://github.com/moble/spherical_functions).

The algorithm computes complex Wigner D matrices directly from the quaternion's
Ra/Rb decomposition, then transforms to real spherical harmonics.

Key properties:
- NO arccos or atan2 on edge vector components
- NO Euler angle computation
- Numerically stable for all edge orientations including y-aligned
- Correct gradients for backpropagation

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# =============================================================================
# Constants
# =============================================================================

# Threshold for detecting near-zero magnitudes
EPSILON = 1e-14


# =============================================================================
# Precomputation of Wigner Coefficients
# =============================================================================


def _factorial_table(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Compute factorial table [0!, 1!, 2!, ..., n!]."""
    table = torch.zeros(n + 1, dtype=dtype, device=device)
    table[0] = 1.0
    for i in range(1, n + 1):
        table[i] = table[i - 1] * i
    return table


def _binomial(n: int, k: int, factorial: torch.Tensor) -> float:
    """Compute binomial coefficient C(n, k) using precomputed factorials."""
    if k < 0 or k > n:
        return 0.0
    return float(factorial[n] / (factorial[k] * factorial[n - k]))


def precompute_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Precompute Wigner D coefficient tables.

    The Wigner D element formula involves:
        coeff(ℓ, m', m) = sqrt[(ℓ+m)!(ℓ-m)! / ((ℓ+m')!(ℓ-m')!)] × C(ℓ+m', ρ_min) × C(ℓ-m', ℓ-m-ρ_min)

    We precompute these for both Case 1 (|Ra| ≥ |Rb|) and Case 2 (|Ra| < |Rb|).

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors

    Returns:
        Dictionary with coefficient tables and metadata
    """
    factorial = _factorial_table(2 * lmax + 1, dtype, device)

    # Storage: [ℓ, m' + lmax, m + lmax]
    size = 2 * lmax + 1
    case1_coeffs = torch.zeros((lmax + 1, size, size), dtype=dtype, device=device)
    case2_coeffs = torch.zeros((lmax + 1, size, size), dtype=dtype, device=device)
    rho_min_case1 = torch.zeros((lmax + 1, size, size), dtype=torch.int64, device=device)
    rho_max_case1 = torch.zeros((lmax + 1, size, size), dtype=torch.int64, device=device)
    rho_min_case2 = torch.zeros((lmax + 1, size, size), dtype=torch.int64, device=device)
    rho_max_case2 = torch.zeros((lmax + 1, size, size), dtype=torch.int64, device=device)

    for ell in range(lmax + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                mp_idx = mp + lmax
                m_idx = m + lmax

                # Sqrt prefactor: sqrt[(ℓ+m)!(ℓ-m)! / ((ℓ+m')!(ℓ-m')!)]
                sqrt_factor = math.sqrt(
                    float(factorial[ell + m] * factorial[ell - m])
                    / float(factorial[ell + mp] * factorial[ell - mp])
                )

                # Case 1: |Ra| ≥ |Rb|
                # ρ_min = max(0, m' - m), ρ_max = min(ℓ + m', ℓ - m)
                rho_min_1 = max(0, mp - m)
                rho_max_1 = min(ell + mp, ell - m)
                rho_min_case1[ell, mp_idx, m_idx] = rho_min_1
                rho_max_case1[ell, mp_idx, m_idx] = rho_max_1

                if rho_min_1 <= rho_max_1:
                    binom1 = _binomial(ell + mp, rho_min_1, factorial)
                    binom2 = _binomial(ell - mp, ell - m - rho_min_1, factorial)
                    case1_coeffs[ell, mp_idx, m_idx] = sqrt_factor * binom1 * binom2

                # Case 2: |Ra| < |Rb|
                # ρ_min = max(0, -(m' + m)), ρ_max = min(ℓ - m, ℓ - m')
                rho_min_2 = max(0, -(mp + m))
                rho_max_2 = min(ell - m, ell - mp)
                rho_min_case2[ell, mp_idx, m_idx] = rho_min_2
                rho_max_case2[ell, mp_idx, m_idx] = rho_max_2

                if rho_min_2 <= rho_max_2:
                    binom1 = _binomial(ell + mp, ell - m - rho_min_2, factorial)
                    binom2 = _binomial(ell - mp, rho_min_2, factorial)
                    case2_coeffs[ell, mp_idx, m_idx] = sqrt_factor * binom1 * binom2

    return {
        "case1_coeffs": case1_coeffs,
        "case2_coeffs": case2_coeffs,
        "rho_min_case1": rho_min_case1,
        "rho_max_case1": rho_max_case1,
        "rho_min_case2": rho_min_case2,
        "rho_max_case2": rho_max_case2,
        "lmax": lmax,
    }


# =============================================================================
# Edge Vector to Quaternion
# =============================================================================


def edge_to_quaternion(
    edge_vec: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert edge vectors to quaternions that rotate +y to the edge direction.

    Uses the half-angle formula which is singularity-free except at edge = -y.

    For edge = (x, y, z), we want quaternion q such that q⊗(0,1,0)⊗q* = edge.
    Using q = normalize(1 + a·b, a×b) with a = +y:
        a·b = y
        a×b = (z, 0, -x)
    So q = normalize(1+y, z, 0, -x)

    Args:
        edge_vec: Edge vectors of shape (N, 3), need not be normalized
        gamma: Optional rotation angles around y-axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    # Normalize edge vectors
    edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    # Generate random gamma if not provided
    if gamma is None:
        gamma = torch.rand_like(ex) * 2 * math.pi

    # Half-angle formula: q = normalize(1+ey, ez, 0, -ex)
    one_plus_ey = 1.0 + ey

    # Handle singularity at ey = -1 (edge pointing to -y)
    singular = one_plus_ey < 1e-7

    # For non-singular case:
    # norm = sqrt((1+ey)² + ez² + ex²) = sqrt(2(1+ey))
    safe_one_plus_ey = torch.where(singular, torch.ones_like(one_plus_ey), one_plus_ey)
    inv_norm = 1.0 / torch.sqrt(2.0 * safe_one_plus_ey)

    w_base = torch.sqrt(safe_one_plus_ey / 2.0)
    x_base = ez * inv_norm
    y_base = torch.zeros_like(ex)
    z_base = -ex * inv_norm

    # Fallback for singular case: 180° rotation around x-axis → q = (0, 1, 0, 0)
    w_base = torch.where(singular, torch.zeros_like(w_base), w_base)
    x_base = torch.where(singular, torch.ones_like(x_base), x_base)
    y_base = torch.where(singular, torch.zeros_like(y_base), y_base)
    z_base = torch.where(singular, torch.zeros_like(z_base), z_base)

    # Apply gamma rotation around y-axis: q_final = q_base ⊗ q_gamma
    # q_gamma = (cos(γ/2), 0, sin(γ/2), 0)
    #
    # This applies the gamma rotation AFTER aligning the edge with y,
    # which means rotating around the aligned axis (the edge direction).
    cos_hg = torch.cos(gamma / 2.0)
    sin_hg = torch.sin(gamma / 2.0)

    # Quaternion multiplication: q_base ⊗ q_gamma
    # (w1,x1,y1,z1) ⊗ (w2,x2,y2,z2) =
    #   (w1w2 - x1x2 - y1y2 - z1z2,
    #    w1x2 + x1w2 + y1z2 - z1y2,
    #    w1y2 - x1z2 + y1w2 + z1x2,
    #    w1z2 + x1y2 - y1x2 + z1w2)
    #
    # With q_gamma = (cos_hg, 0, sin_hg, 0):
    w = w_base * cos_hg - y_base * sin_hg
    x = x_base * cos_hg + z_base * sin_hg
    y = y_base * cos_hg + w_base * sin_hg
    z = z_base * cos_hg - x_base * sin_hg

    return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# Quaternion to Ra/Rb Decomposition
# =============================================================================


def quaternion_to_ra_rb(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into complex numbers Ra and Rb.

    For q = (w, x, y, z):
        Ra = w + i·z
        Rb = y + i·x

    These encode the rotation as:
        |Ra| = cos(β/2), |Rb| = sin(β/2)
        arg(Ra) = (α+γ)/2, arg(Rb) = (γ-α)/2

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) convention

    Returns:
        Tuple (Ra, Rb) of complex tensors with shape (...)
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    Ra = torch.complex(w, z)
    Rb = torch.complex(y, x)

    return Ra, Rb


# =============================================================================
# Complex Wigner D Matrix Computation
# =============================================================================


def wigner_d_element_complex(
    ell: int,
    mp: int,
    m: int,
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    coeffs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute a single complex Wigner D matrix element D^ℓ_{m',m}(R).

    Implements the four-case algorithm from spherical_functions.

    Args:
        ell: Angular momentum quantum number
        mp: m' index
        m: m index
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        coeffs: Precomputed coefficient dictionary

    Returns:
        Complex tensor of D^ℓ_{m',m} values, shape (N,)
    """
    ra = torch.abs(Ra)
    rb = torch.abs(Rb)

    lmax = coeffs["lmax"]
    mp_idx = mp + lmax
    m_idx = m + lmax

    # Get precomputed values
    coeff1 = coeffs["case1_coeffs"][ell, mp_idx, m_idx]
    coeff2 = coeffs["case2_coeffs"][ell, mp_idx, m_idx]
    rho_min_1 = int(coeffs["rho_min_case1"][ell, mp_idx, m_idx].item())
    rho_max_1 = int(coeffs["rho_max_case1"][ell, mp_idx, m_idx].item())
    rho_min_2 = int(coeffs["rho_min_case2"][ell, mp_idx, m_idx].item())
    rho_max_2 = int(coeffs["rho_max_case2"][ell, mp_idx, m_idx].item())

    N = Ra.shape[0]
    complex_dtype = Ra.dtype
    device = Ra.device

    result = torch.zeros(N, dtype=complex_dtype, device=device)

    # Case masks
    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    use_case1 = (ra >= rb) & ~ra_small
    use_case2 = (ra < rb) & ~rb_small

    # ==========================================================================
    # Case 1: |Ra| ≈ 0 (β ≈ π)
    # Only anti-diagonal elements: D^ℓ_{-m,m} = (-1)^{ℓ-m} Rb^{2m}
    # ==========================================================================
    if mp == -m:
        sign = (-1) ** (ell - m)
        # Rb^{2m} = |Rb|^{2m} · exp(i·2m·arg(Rb))
        val = sign * torch.pow(Rb, 2 * m)
        result = torch.where(ra_small, val, result)

    # ==========================================================================
    # Case 2: |Rb| ≈ 0 (β ≈ 0)
    # Only diagonal elements: D^ℓ_{m,m} = Ra^{2m}
    # ==========================================================================
    if mp == m:
        val = torch.pow(Ra, 2 * m)
        result = torch.where(rb_small & ~ra_small, val, result)

    # ==========================================================================
    # Case 3: |Ra| ≥ |Rb| (general case, branch 1)
    # D = coeff · |Ra|^{2ℓ-2m} · Ra^{m'+m} · Rb^{m-m'} · Σ_ρ[...]
    # ==========================================================================
    if use_case1.any() and rho_min_1 <= rho_max_1:
        # Ratio for Horner evaluation: r = -(|Rb|/|Ra|)²
        safe_ra = torch.clamp(ra, min=EPSILON)
        ratio = -(rb * rb) / (safe_ra * safe_ra)

        # Horner evaluation of polynomial
        horner_sum = torch.ones(N, dtype=Ra.real.dtype, device=device)
        for rho in range(rho_max_1, rho_min_1, -1):
            n1 = ell + mp - rho + 1
            n2 = ell - m - rho + 1
            d1 = rho
            d2 = m - mp + rho
            if d1 != 0 and d2 != 0:
                factor = ratio * (n1 * n2) / (d1 * d2)
                horner_sum = 1.0 + horner_sum * factor

        # Magnitude: coeff · |Ra|^{2ℓ+m'-m-2ρ_min} · |Rb|^{m-m'+2ρ_min}
        ra_exp = 2 * ell + mp - m - 2 * rho_min_1
        rb_exp = m - mp + 2 * rho_min_1
        magnitude = coeff1 * torch.pow(safe_ra, ra_exp) * torch.pow(rb, rb_exp)

        # Sign factor: (-1)^{ρ_min}
        sign = (-1) ** rho_min_1
        magnitude = sign * magnitude

        # Phase: (m'+m)·arg(Ra) + (m-m')·arg(Rb)
        phia = torch.angle(Ra)
        phib = torch.angle(Rb)
        phase = (mp + m) * phia + (m - mp) * phib

        # Combine: D = magnitude · Σ · exp(i·phase)
        val = magnitude * horner_sum * torch.exp(1j * phase)
        result = torch.where(use_case1, val, result)

    # ==========================================================================
    # Case 4: |Ra| < |Rb| (general case, branch 2)
    # D = (-1)^{ℓ-m} · coeff · Ra^{m'+m} · Rb^{m-m'} · |Rb|^{2ℓ-2m} · Σ_ρ[...]
    # ==========================================================================
    if use_case2.any() and rho_min_2 <= rho_max_2:
        # Ratio for Horner evaluation: r = -(|Ra|/|Rb|)²
        safe_rb = torch.clamp(rb, min=EPSILON)
        ratio = -(ra * ra) / (safe_rb * safe_rb)

        # Horner evaluation
        horner_sum = torch.ones(N, dtype=Ra.real.dtype, device=device)
        for rho in range(rho_max_2, rho_min_2, -1):
            n1 = ell - m - rho + 1
            n2 = ell - mp - rho + 1
            d1 = rho
            d2 = mp + m + rho
            if d1 != 0 and d2 != 0:
                factor = ratio * (n1 * n2) / (d1 * d2)
                horner_sum = 1.0 + horner_sum * factor

        # Magnitude: coeff · |Ra|^{m'+m+2ρ_min} · |Rb|^{2ℓ-m'-m-2ρ_min}
        ra_exp = mp + m + 2 * rho_min_2
        rb_exp = 2 * ell - mp - m - 2 * rho_min_2
        magnitude = coeff2 * torch.pow(ra, ra_exp) * torch.pow(safe_rb, rb_exp)

        # Sign factor: (-1)^{ℓ-m} · (-1)^{ρ_min}
        sign = ((-1) ** (ell - m)) * ((-1) ** rho_min_2)
        magnitude = sign * magnitude

        # Phase: (m'+m)·arg(Ra) + (m-m')·arg(Rb)
        phia = torch.angle(Ra)
        phib = torch.angle(Rb)
        phase = (mp + m) * phia + (m - mp) * phib

        # Combine
        val = magnitude * horner_sum * torch.exp(1j * phase)
        result = torch.where(use_case2, val, result)

    return result


def wigner_d_matrix_complex(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    lmax: int,
    coeffs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute full complex Wigner D matrices for ℓ = 0, 1, ..., lmax.

    Returns block-diagonal matrix where each block is the (2ℓ+1)×(2ℓ+1)
    Wigner D matrix for that ℓ.

    Args:
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        lmax: Maximum angular momentum
        coeffs: Precomputed coefficient dictionary

    Returns:
        Complex block-diagonal matrices of shape (N, size, size)
        where size = (lmax+1)²
    """
    N = Ra.shape[0]
    device = Ra.device
    complex_dtype = Ra.dtype

    size = (lmax + 1) ** 2
    D = torch.zeros(N, size, size, dtype=complex_dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1

        # Compute each element of the (2ℓ+1)×(2ℓ+1) block
        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell

                D[:, block_start + mp_local, block_start + m_local] = \
                    wigner_d_element_complex(ell, mp, m, Ra, Rb, coeffs)

        block_start += block_size

    return D


# =============================================================================
# Complex to Real Spherical Harmonics Transformation
# =============================================================================


def precompute_complex_to_real_matrix(
    lmax: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute the unitary matrix U that transforms complex → real spherical harmonics.

    Real spherical harmonics convention (matching e3nn):
        m > 0: Y^m_ℓ(real) = (1/√2) · [(-1)^m · Y^m_ℓ(complex) + Y^{-m}_ℓ(complex)]
        m = 0: Y^0_ℓ(real) = Y^0_ℓ(complex)
        m < 0: Y^m_ℓ(real) = (i/√2) · [Y^{-|m|}_ℓ(complex) - (-1)^{|m|} · Y^{|m|}_ℓ(complex)]

    Note: Different sources use different conventions. This follows e3nn.

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type
        device: Device for output

    Returns:
        Block-diagonal unitary matrix of shape (size, size) where size = (lmax+1)²
    """
    size = (lmax + 1) ** 2
    U = torch.zeros(size, size, dtype=dtype, device=device)

    sqrt2_inv = 1.0 / math.sqrt(2.0)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1

        for m in range(-ell, ell + 1):
            row = block_start + (m + ell)  # Real harmonic index

            if m > 0:
                # Y^m(real) = (1/√2) · [(-1)^m · Y^m(complex) + Y^{-m}(complex)]
                col_pos = block_start + (m + ell)   # Y^m(complex)
                col_neg = block_start + (-m + ell)  # Y^{-m}(complex)
                sign = (-1) ** m
                U[row, col_pos] = sign * sqrt2_inv
                U[row, col_neg] = sqrt2_inv

            elif m == 0:
                # Y^0(real) = Y^0(complex)
                col = block_start + ell
                U[row, col] = 1.0

            else:  # m < 0
                # Y^m(real) = (i/√2) · [Y^{-|m|}(complex) - (-1)^{|m|} · Y^{|m|}(complex)]
                abs_m = abs(m)
                col_pos = block_start + (abs_m + ell)   # Y^{|m|}(complex)
                col_neg = block_start + (-abs_m + ell)  # Y^{-|m|}(complex)
                sign = (-1) ** abs_m
                U[row, col_neg] = 1j * sqrt2_inv
                U[row, col_pos] = -sign * 1j * sqrt2_inv

        block_start += block_size

    return U


def wigner_d_complex_to_real(
    D_complex: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from complex to real spherical harmonics basis.

    If Y_real = U @ Y_complex (column vectors of spherical harmonics), then
    the representation transforms as:
        D_real = U · D_complex · U†

    Args:
        D_complex: Complex Wigner D matrices of shape (N, size, size)
        U: Unitary transformation matrix of shape (size, size)

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    # D_real = U @ D_complex @ U^H
    U_H = U.conj().T
    D_real = torch.einsum("ij,njk,lk->nil", U, D_complex, U.conj())

    # The result should be real for proper rotations
    return D_real.real


# =============================================================================
# Main Entry Point
# =============================================================================


def compute_wigner_d_from_quaternion(
    q: torch.Tensor,
    lmax: int,
    coeffs: dict[str, torch.Tensor],
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Compute real Wigner D matrices from quaternions.

    This is the main entry point that:
    1. Decomposes quaternion into Ra/Rb
    2. Computes complex Wigner D matrices
    3. Transforms to real spherical harmonics basis

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        coeffs: Precomputed Wigner coefficients
        U: Precomputed complex-to-real transformation matrix

    Returns:
        Real block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)²
    """
    Ra, Rb = quaternion_to_ra_rb(q)
    D_complex = wigner_d_matrix_complex(Ra, Rb, lmax, coeffs)
    D_real = wigner_d_complex_to_real(D_complex, U)
    return D_real


def get_wigner_from_edge_vectors(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    coeffs: dict[str, torch.Tensor],
    U: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete pipeline: edge vectors → real Wigner D matrices.

    This is the drop-in replacement for the Euler angle-based pipeline.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        lmax: Maximum angular momentum
        coeffs: Precomputed Wigner coefficients
        U: Precomputed complex-to-real transformation matrix

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
        and size = (lmax+1)²
    """
    q = edge_to_quaternion(edge_distance_vec, gamma=None)
    wigner = compute_wigner_d_from_quaternion(q, lmax, coeffs, U)

    # Inverse is transpose (Wigner D matrices are orthogonal)
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

    return wigner, wigner_inv
