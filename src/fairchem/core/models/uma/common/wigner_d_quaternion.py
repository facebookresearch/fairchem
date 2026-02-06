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
from pathlib import Path
from typing import Optional

import torch


# =============================================================================
# Constants
# =============================================================================

# Threshold for detecting near-zero magnitudes
EPSILON = 1e-14

# Default cache directory for precomputed coefficients
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fairchem" / "wigner_coeffs"


# =============================================================================
# Core Helper Functions
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


def _vectorized_horner(
    ratio: torch.Tensor,
    horner_coeffs: torch.Tensor,
    poly_len: torch.Tensor,
    max_poly_len: int,
) -> torch.Tensor:
    """
    Vectorized Horner polynomial evaluation for all elements simultaneously.

    Evaluates polynomials of varying lengths using masking.

    Args:
        ratio: The ratio term -(rb/ra)² or -(ra/rb)², shape (N,)
        horner_coeffs: Horner factors, shape (n_elements, max_poly_len)
        poly_len: Actual polynomial length per element, shape (n_elements,)
        max_poly_len: Maximum polynomial length

    Returns:
        Polynomial values of shape (N, n_elements)
    """
    N = ratio.shape[0]
    n_elements = horner_coeffs.shape[0]
    device = ratio.device
    dtype = ratio.dtype

    # Initialize result: all elements start with 1.0
    # Shape: (N, n_elements)
    result = torch.ones(N, n_elements, dtype=dtype, device=device)

    # ratio broadcasted to (N, n_elements)
    ratio_expanded = ratio.unsqueeze(1).expand(N, n_elements)

    # Iterate through Horner steps (from highest term down)
    # poly_len-1 is the number of Horner steps needed
    for i in range(max_poly_len - 1):
        # Get coefficient for this step: (n_elements,)
        coeff = horner_coeffs[:, i]

        # Mask: only apply if i < poly_len - 1 (i.e., this step is needed)
        # poly_len = rho_max - rho_min + 1, so we need poly_len - 1 Horner steps
        mask = i < (poly_len - 1)  # (n_elements,)

        # factor = ratio * coeff
        factor = ratio_expanded * coeff.unsqueeze(0)  # (N, n_elements)

        # result = 1.0 + result * factor, but only where mask is True
        new_result = 1.0 + result * factor
        result = torch.where(mask.unsqueeze(0), new_result, result)

    return result


def _smooth_step_cinf(t: torch.Tensor) -> torch.Tensor:
    """
    C-infinity smooth step function based on the classic bump function.

    Uses f(x) = exp(-1/x) for x > 0 (0 otherwise), then:
    step(t) = f(t) / (f(t) + f(1-t)) = sigmoid((2t-1)/(t*(1-t)))

    Properties:
    - C-infinity smooth everywhere
    - All derivatives are exactly zero at t=0 and t=1
    - Values: f(0)=0, f(1)=1
    - Symmetric: f(t) + f(1-t) = 1

    This provides true C-infinity continuity at the blend region boundaries,
    unlike polynomial smoothstep functions which only have finite-order continuity.

    Args:
        t: Input tensor, will be clamped to [0, 1]

    Returns:
        Smooth step values in [0, 1]
    """
    t_clamped = t.clamp(0, 1)

    # Use a safe epsilon that works for both float32 and float64
    eps = 1e-7

    # Compute (2t-1)/(t*(1-t)) in a numerically stable way
    # Near t=0: this goes to -infinity -> sigmoid = 0
    # Near t=1: this goes to +infinity -> sigmoid = 1
    # At t=0.5: this is 0 -> sigmoid = 0.5
    numerator = 2.0 * t_clamped - 1.0
    denominator = t_clamped * (1.0 - t_clamped)

    # Clamp denominator to avoid division by zero
    denom_safe = denominator.clamp(min=eps)

    # Compute the argument for sigmoid
    arg = numerator / denom_safe

    # Apply sigmoid
    result = torch.sigmoid(arg)

    # Near boundaries, sigmoid will saturate naturally, but we ensure
    # exact 0 and 1 for very small/large t to avoid numerical issues
    result = torch.where(t_clamped < eps, torch.zeros_like(result), result)
    result = torch.where(t_clamped > 1 - eps, torch.ones_like(result), result)

    return result


# =============================================================================
# Edge Vector to Quaternion
# =============================================================================


def _quat_y_to_edge_standard(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Standard quaternion formula: rotate +Y to edge. Singular at edge = -Y.

    Uses q = normalize(1+ey, ez, 0, -ex).
    """
    one_plus_ey = 1.0 + ey

    # Handle singularity at ey = -1
    singular = one_plus_ey < 1e-7
    safe_one_plus_ey = torch.where(singular, torch.ones_like(one_plus_ey), one_plus_ey)
    inv_norm = 1.0 / torch.sqrt(2.0 * safe_one_plus_ey)

    w = torch.sqrt(safe_one_plus_ey / 2.0)
    x = ez * inv_norm
    y = torch.zeros_like(ex)
    z = -ex * inv_norm

    # Fallback for singular case: 180° rotation around x-axis
    w = torch.where(singular, torch.zeros_like(w), w)
    x = torch.where(singular, torch.ones_like(x), x)
    y = torch.where(singular, torch.zeros_like(y), y)
    z = torch.where(singular, torch.zeros_like(z), z)

    return torch.stack([w, x, y, z], dim=-1)


def _quat_y_to_edge_via_z(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Alternative quaternion formula: rotate +Y to edge via +Z. Singular at edge = -Z.

    First rotates +Y to +Z, then +Z to edge.
    """
    one_plus_ez = 1.0 + ez

    # Handle singularity at ez = -1
    singular = one_plus_ez < 1e-7
    safe_one_plus_ez = torch.where(singular, torch.ones_like(one_plus_ez), one_plus_ez)
    inv_norm = 1.0 / torch.sqrt(2.0 * safe_one_plus_ez)

    # Quaternion for +Z to edge: q = normalize(1+ez, -ey, ex, 0)
    w2 = torch.sqrt(safe_one_plus_ez / 2.0)
    x2 = -ey * inv_norm
    y2 = ex * inv_norm
    z2 = torch.zeros_like(ex)

    # Fallback for singular case
    w2 = torch.where(singular, torch.zeros_like(w2), w2)
    x2 = torch.where(singular, torch.ones_like(x2), x2)
    y2 = torch.where(singular, torch.zeros_like(y2), y2)

    # Compose with +Y to +Z rotation: q_yz = (√2/2, √2/2, 0, 0)
    # q_total = q_ze ⊗ q_yz (apply q_yz first, then q_ze)
    s2 = math.sqrt(0.5)

    w = s2 * w2 - s2 * x2
    x = s2 * w2 + s2 * x2
    y = s2 * y2 + s2 * z2
    z = s2 * z2 - s2 * y2

    return torch.stack([w, x, y, z], dim=-1)


def edge_to_quaternion(
    edge_vec: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert edge vectors to quaternions that rotate +y to the edge direction.

    Uses an adaptive approach with two formulas to avoid singularities:
    - Standard formula (singular at -Y): used when ey > -0.7
    - Via-Z formula (singular at -Z): used when ey < -0.9
    - C-infinity smooth blend in between

    This ensures bounded gradients for all edge directions with continuous
    derivatives of all orders across the blend region boundaries.

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

    # Compute both quaternion formulas
    q_std = _quat_y_to_edge_standard(ex, ey, ez)
    q_via_z = _quat_y_to_edge_via_z(ex, ey, ez)

    # Smooth blend: use standard when ey > -0.7, via-Z when ey < -0.9
    # C-infinity smooth step based on the classic bump function:
    # f(x) = exp(-1/x) for x > 0, 0 otherwise
    # step(t) = f(t) / (f(t) + f(1-t)) = sigmoid((2t-1)/(t*(1-t)))
    # All derivatives are exactly zero at t=0 and t=1, giving C-infinity
    # continuity across the blend region boundaries.
    blend_start = -0.9
    blend_width = 0.2
    t = (ey - blend_start) / blend_width  # 0 at ey=-0.9, 1 at ey=-0.7
    blend = _smooth_step_cinf(t)

    # Align quaternions (q and -q represent same rotation)
    dot = (q_std * q_via_z).sum(dim=-1, keepdim=True)
    q_via_z_aligned = torch.where(dot < 0, -q_via_z, q_via_z)

    # Blend quaternions and renormalize
    q_base = blend.unsqueeze(-1) * q_std + (1 - blend.unsqueeze(-1)) * q_via_z_aligned
    q_base = torch.nn.functional.normalize(q_base, dim=-1)

    w_base = q_base[..., 0]
    x_base = q_base[..., 1]
    y_base = q_base[..., 2]
    z_base = q_base[..., 3]

    # Apply gamma rotation around y-axis: q_final = q_gamma ⊗ q_base
    # Note: We negate gamma because the Wigner D is transposed to get edge→+y
    cos_hg = torch.cos(-gamma / 2.0)
    sin_hg = torch.sin(-gamma / 2.0)

    # Quaternion multiplication: q_gamma ⊗ q_base
    # q_gamma = (cos_hg, 0, sin_hg, 0)
    w = cos_hg * w_base - sin_hg * y_base
    x = cos_hg * x_base - sin_hg * z_base
    y = cos_hg * y_base + sin_hg * w_base
    z = cos_hg * z_base + sin_hg * x_base

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
# Precomputation of Wigner Coefficients (Symmetric Version)
# =============================================================================


def precompute_wigner_coefficients_symmetric(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Precompute Wigner D coefficients exploiting conjugate symmetry.

    Uses the symmetry D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m}) to compute
    only ~half the elements ("primary") and derive the rest ("derived").

    Primary elements: m' + m > 0, OR (m' + m = 0 AND m' >= 0)

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors

    Returns:
        Dictionary with symmetric coefficient tables
    """
    factorial = _factorial_table(2 * lmax + 1, dtype, device)

    # Count elements: primary vs derived
    # For each (ell, mp, m), it's primary if mp + m > 0 or (mp + m == 0 and mp >= 0)
    n_total = sum((2 * ell + 1) ** 2 for ell in range(lmax + 1))
    n_primary = 0
    for ell in range(lmax + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                if mp + m > 0 or (mp + m == 0 and mp >= 0):
                    n_primary += 1

    max_poly_len = lmax + 1

    # Storage for PRIMARY elements only
    primary_row_indices = torch.zeros(n_primary, dtype=torch.int64, device=device)
    primary_col_indices = torch.zeros(n_primary, dtype=torch.int64, device=device)

    case1_coeff = torch.zeros(n_primary, dtype=dtype, device=device)
    case2_coeff = torch.zeros(n_primary, dtype=dtype, device=device)
    case1_horner = torch.zeros(n_primary, max_poly_len, dtype=dtype, device=device)
    case2_horner = torch.zeros(n_primary, max_poly_len, dtype=dtype, device=device)
    case1_poly_len = torch.zeros(n_primary, dtype=torch.int64, device=device)
    case2_poly_len = torch.zeros(n_primary, dtype=torch.int64, device=device)
    case1_ra_exp = torch.zeros(n_primary, dtype=dtype, device=device)
    case1_rb_exp = torch.zeros(n_primary, dtype=dtype, device=device)
    case2_ra_exp = torch.zeros(n_primary, dtype=dtype, device=device)
    case2_rb_exp = torch.zeros(n_primary, dtype=dtype, device=device)
    case1_sign = torch.zeros(n_primary, dtype=dtype, device=device)
    case2_sign = torch.zeros(n_primary, dtype=dtype, device=device)
    mp_plus_m = torch.zeros(n_primary, dtype=dtype, device=device)
    m_minus_mp = torch.zeros(n_primary, dtype=dtype, device=device)

    # Special case info
    diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    anti_diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    special_2m = torch.zeros(n_primary, dtype=dtype, device=device)
    anti_diag_sign = torch.zeros(n_primary, dtype=dtype, device=device)

    # Derived element info: for each derived element, store:
    # - its (row, col) position
    # - the index of its corresponding primary element
    # - the sign factor (-1)^{m'-m}
    n_derived = n_total - n_primary
    derived_row_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_col_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_primary_idx = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_sign = torch.zeros(n_derived, dtype=dtype, device=device)

    # Build mapping from (row, col) to primary index for lookup
    primary_map = {}  # (row, col) -> primary_idx

    primary_idx = 0
    derived_idx = 0
    block_start = 0

    for ell in range(lmax + 1):
        block_size = 2 * ell + 1

        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell

                row = block_start + mp_local
                col = block_start + m_local

                is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)

                if is_primary:
                    primary_map[(row, col)] = primary_idx

                    primary_row_indices[primary_idx] = row
                    primary_col_indices[primary_idx] = col

                    mp_plus_m[primary_idx] = mp + m
                    m_minus_mp[primary_idx] = m - mp

                    diagonal_mask[primary_idx] = mp == m
                    anti_diagonal_mask[primary_idx] = mp == -m
                    special_2m[primary_idx] = 2 * m
                    anti_diag_sign[primary_idx] = (-1) ** (ell - m)

                    # Sqrt prefactor
                    sqrt_factor = math.sqrt(
                        float(factorial[ell + m] * factorial[ell - m])
                        / float(factorial[ell + mp] * factorial[ell - mp])
                    )

                    # Case 1: |Ra| >= |Rb|
                    rho_min_1 = max(0, mp - m)
                    rho_max_1 = min(ell + mp, ell - m)

                    if rho_min_1 <= rho_max_1:
                        binom1 = _binomial(ell + mp, rho_min_1, factorial)
                        binom2 = _binomial(ell - mp, ell - m - rho_min_1, factorial)
                        case1_coeff[primary_idx] = sqrt_factor * binom1 * binom2

                        poly_len = rho_max_1 - rho_min_1 + 1
                        case1_poly_len[primary_idx] = poly_len

                        for i, rho in enumerate(range(rho_max_1, rho_min_1, -1)):
                            n1 = ell + mp - rho + 1
                            n2 = ell - m - rho + 1
                            d1 = rho
                            d2 = m - mp + rho
                            if d1 != 0 and d2 != 0:
                                case1_horner[primary_idx, i] = (n1 * n2) / (d1 * d2)

                        case1_ra_exp[primary_idx] = 2 * ell + mp - m - 2 * rho_min_1
                        case1_rb_exp[primary_idx] = m - mp + 2 * rho_min_1
                        case1_sign[primary_idx] = (-1) ** rho_min_1

                    # Case 2: |Ra| < |Rb|
                    rho_min_2 = max(0, -(mp + m))
                    rho_max_2 = min(ell - m, ell - mp)

                    if rho_min_2 <= rho_max_2:
                        binom1 = _binomial(ell + mp, ell - m - rho_min_2, factorial)
                        binom2 = _binomial(ell - mp, rho_min_2, factorial)
                        case2_coeff[primary_idx] = sqrt_factor * binom1 * binom2

                        poly_len = rho_max_2 - rho_min_2 + 1
                        case2_poly_len[primary_idx] = poly_len

                        for i, rho in enumerate(range(rho_max_2, rho_min_2, -1)):
                            n1 = ell - m - rho + 1
                            n2 = ell - mp - rho + 1
                            d1 = rho
                            d2 = mp + m + rho
                            if d1 != 0 and d2 != 0:
                                case2_horner[primary_idx, i] = (n1 * n2) / (d1 * d2)

                        case2_ra_exp[primary_idx] = mp + m + 2 * rho_min_2
                        case2_rb_exp[primary_idx] = 2 * ell - mp - m - 2 * rho_min_2
                        case2_sign[primary_idx] = ((-1) ** (ell - m)) * ((-1) ** rho_min_2)

                    primary_idx += 1

        block_start += block_size

    # Second pass: fill derived element info
    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1

        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell

                row = block_start + mp_local
                col = block_start + m_local

                is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)

                if not is_primary:
                    # This element (-mp, -m) maps to primary element (mp, m)
                    # Find the corresponding primary element
                    neg_mp = -mp
                    neg_m = -m
                    neg_mp_local = neg_mp + ell
                    neg_m_local = neg_m + ell
                    primary_row = block_start + neg_mp_local
                    primary_col = block_start + neg_m_local

                    derived_row_indices[derived_idx] = row
                    derived_col_indices[derived_idx] = col
                    derived_primary_idx[derived_idx] = primary_map[(primary_row, primary_col)]
                    derived_sign[derived_idx] = (-1) ** (mp - m)

                    derived_idx += 1

        block_start += block_size

    return {
        "n_primary": n_primary,
        "n_derived": n_derived,
        "n_total": n_total,
        "max_poly_len": max_poly_len,
        "lmax": lmax,
        # Primary element indices
        "primary_row_indices": primary_row_indices,
        "primary_col_indices": primary_col_indices,
        # Case 1
        "case1_coeff": case1_coeff,
        "case1_horner": case1_horner,
        "case1_poly_len": case1_poly_len,
        "case1_ra_exp": case1_ra_exp,
        "case1_rb_exp": case1_rb_exp,
        "case1_sign": case1_sign,
        # Case 2
        "case2_coeff": case2_coeff,
        "case2_horner": case2_horner,
        "case2_poly_len": case2_poly_len,
        "case2_ra_exp": case2_ra_exp,
        "case2_rb_exp": case2_rb_exp,
        "case2_sign": case2_sign,
        # Phase
        "mp_plus_m": mp_plus_m,
        "m_minus_mp": m_minus_mp,
        # Special cases
        "diagonal_mask": diagonal_mask,
        "anti_diagonal_mask": anti_diagonal_mask,
        "special_2m": special_2m,
        "anti_diag_sign": anti_diag_sign,
        # Derived element mapping
        "derived_row_indices": derived_row_indices,
        "derived_col_indices": derived_col_indices,
        "derived_primary_idx": derived_primary_idx,
        "derived_sign": derived_sign,
    }


# =============================================================================
# Complex Wigner D Matrix Computation
# =============================================================================


def wigner_d_matrix_complex(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    coeffs_sym: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute complex Wigner D matrices exploiting conjugate symmetry.

    Computes only primary elements (~half) and derives the rest via:
        D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m})

    Uses optimized vectorized operations with torch.outer and log-exp trick.

    Args:
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        coeffs_sym: Precomputed symmetric coefficient dictionary

    Returns:
        Complex block-diagonal matrices of shape (N, size, size)
    """
    N = Ra.shape[0]
    device = Ra.device
    complex_dtype = Ra.dtype
    real_dtype = Ra.real.dtype

    lmax = coeffs_sym["lmax"]
    n_primary = coeffs_sym["n_primary"]
    max_poly_len = coeffs_sym["max_poly_len"]
    size = (lmax + 1) ** 2

    # Extract precomputed arrays
    primary_row_indices = coeffs_sym["primary_row_indices"]
    primary_col_indices = coeffs_sym["primary_col_indices"]

    case1_coeff = coeffs_sym["case1_coeff"]
    case1_horner = coeffs_sym["case1_horner"]
    case1_poly_len = coeffs_sym["case1_poly_len"]
    case1_ra_exp = coeffs_sym["case1_ra_exp"]
    case1_rb_exp = coeffs_sym["case1_rb_exp"]
    case1_sign = coeffs_sym["case1_sign"]

    case2_coeff = coeffs_sym["case2_coeff"]
    case2_horner = coeffs_sym["case2_horner"]
    case2_poly_len = coeffs_sym["case2_poly_len"]
    case2_ra_exp = coeffs_sym["case2_ra_exp"]
    case2_rb_exp = coeffs_sym["case2_rb_exp"]
    case2_sign = coeffs_sym["case2_sign"]

    mp_plus_m = coeffs_sym["mp_plus_m"]
    m_minus_mp = coeffs_sym["m_minus_mp"]

    diagonal_mask = coeffs_sym["diagonal_mask"]
    anti_diagonal_mask = coeffs_sym["anti_diagonal_mask"]
    special_2m = coeffs_sym["special_2m"]
    anti_diag_sign = coeffs_sym["anti_diag_sign"]

    derived_row_indices = coeffs_sym["derived_row_indices"]
    derived_col_indices = coeffs_sym["derived_col_indices"]
    derived_primary_idx = coeffs_sym["derived_primary_idx"]
    derived_sign = coeffs_sym["derived_sign"]

    # Compute magnitudes
    ra = torch.abs(Ra)
    rb = torch.abs(Rb)

    # Case masks
    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    # Phase - compute once using torch.outer
    phia = torch.angle(Ra)
    phib = torch.angle(Rb)
    phase = torch.outer(phia, mp_plus_m) + torch.outer(phib, m_minus_mp)
    exp_phase = torch.exp(1j * phase)

    # Safe magnitudes and their logs for power computation
    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    # Initialize primary results: (N, n_primary)
    result = torch.zeros(N, n_primary, dtype=complex_dtype, device=device)

    # ==========================================================================
    # Special Case 1: |Ra| ≈ 0 - anti-diagonal elements
    # ==========================================================================
    safe_Rb = torch.where(rb < EPSILON, torch.ones_like(Rb), Rb)
    # Replace torch.pow with log-exp trick for complex powers: z^n = exp(n*log|z| + i*n*arg(z))
    log_abs_Rb = torch.log(torch.abs(safe_Rb))  # log|Rb|, shape (N,)
    arg_Rb = torch.angle(safe_Rb)  # arg(Rb), shape (N,)
    # Compute n*log|Rb| + i*n*arg(Rb) using outer products
    exponent = torch.outer(log_abs_Rb, special_2m) + 1j * torch.outer(arg_Rb, special_2m)
    rb_power = torch.exp(exponent.to(dtype=complex_dtype))
    special_val_antidiag = anti_diag_sign.unsqueeze(0).to(complex_dtype) * rb_power
    mask_antidiag = ra_small.unsqueeze(1) & anti_diagonal_mask.unsqueeze(0)
    result = torch.where(mask_antidiag, special_val_antidiag, result)

    # ==========================================================================
    # Special Case 2: |Rb| ≈ 0 - diagonal elements
    # ==========================================================================
    safe_Ra = torch.where(ra < EPSILON, torch.ones_like(Ra), Ra)
    # Replace torch.pow with log-exp trick for complex powers: z^n = exp(n*log|z| + i*n*arg(z))
    log_abs_Ra = torch.log(torch.abs(safe_Ra))  # log|Ra|, shape (N,)
    arg_Ra = torch.angle(safe_Ra)  # arg(Ra), shape (N,)
    # Compute n*log|Ra| + i*n*arg(Ra) using outer products
    exponent = torch.outer(log_abs_Ra, special_2m) + 1j * torch.outer(arg_Ra, special_2m)
    ra_power = torch.exp(exponent.to(dtype=complex_dtype))
    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & diagonal_mask.unsqueeze(0)
    result = torch.where(mask_diag, ra_power, result)

    # ==========================================================================
    # General Case 1: |Ra| >= |Rb|
    # ==========================================================================
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    horner_sum1 = _vectorized_horner(ratio1, case1_horner, case1_poly_len, max_poly_len)

    # Power computation using log-exp trick with torch.outer
    ra_powers1 = torch.exp(torch.outer(log_ra, case1_ra_exp))
    rb_powers1 = torch.exp(torch.outer(log_rb, case1_rb_exp))

    magnitude1 = (case1_sign * case1_coeff) * ra_powers1 * rb_powers1
    val1 = magnitude1 * horner_sum1 * exp_phase

    valid_case1 = case1_poly_len > 0
    mask1 = use_case1.unsqueeze(1) & valid_case1.unsqueeze(0)
    result = torch.where(mask1, val1.to(dtype=complex_dtype), result)

    # ==========================================================================
    # General Case 2: |Ra| < |Rb|
    # ==========================================================================
    ratio2 = -(ra * ra) / (safe_rb * safe_rb)
    horner_sum2 = _vectorized_horner(ratio2, case2_horner, case2_poly_len, max_poly_len)

    ra_powers2 = torch.exp(torch.outer(log_ra, case2_ra_exp))
    rb_powers2 = torch.exp(torch.outer(log_rb, case2_rb_exp))

    magnitude2 = (case2_sign * case2_coeff) * ra_powers2 * rb_powers2
    val2 = magnitude2 * horner_sum2 * exp_phase

    valid_case2 = case2_poly_len > 0
    mask2 = use_case2.unsqueeze(1) & valid_case2.unsqueeze(0)
    result = torch.where(mask2, val2.to(dtype=complex_dtype), result)

    # ==========================================================================
    # Scatter primary results into output matrix
    # ==========================================================================
    D = torch.zeros(N, size, size, dtype=complex_dtype, device=device)

    batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, n_primary)
    row_expanded = primary_row_indices.unsqueeze(0).expand(N, n_primary)
    col_expanded = primary_col_indices.unsqueeze(0).expand(N, n_primary)

    D[batch_indices, row_expanded, col_expanded] = result

    # ==========================================================================
    # Fill derived elements using symmetry
    # D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m})
    # ==========================================================================
    n_derived = coeffs_sym["n_derived"]
    if n_derived > 0:
        # Get the primary values for each derived element
        primary_vals = result[:, derived_primary_idx]  # (N, n_derived)

        # Apply symmetry: conjugate and multiply by sign
        derived_vals = derived_sign.unsqueeze(0).to(complex_dtype) * primary_vals.conj()

        # Scatter derived values
        batch_indices_d = torch.arange(N, device=device).unsqueeze(1).expand(N, n_derived)
        row_expanded_d = derived_row_indices.unsqueeze(0).expand(N, n_derived)
        col_expanded_d = derived_col_indices.unsqueeze(0).expand(N, n_derived)

        D[batch_indices_d, row_expanded_d, col_expanded_d] = derived_vals

    return D


# =============================================================================
# Complex to Real Spherical Harmonics Transformation
# =============================================================================


def _build_u_block(
    ell: int,
    dtype: torch.dtype,
    device: torch.device,
    convention: str = "fairchem",
) -> torch.Tensor:
    """
    Build U transformation matrix for a single ell block.

    Args:
        ell: Angular momentum quantum number
        dtype: Complex data type for the matrix
        device: Device for the tensor
        convention: Either "e3nn" (m = -l to +l) or "fairchem" (m = +1, -1, 0 for l=1)

    Returns:
        U matrix of shape (2*ell+1, 2*ell+1)
    """
    block_size = 2 * ell + 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    U_ell = torch.zeros(block_size, block_size, dtype=dtype, device=device)

    # fairchem l=1 uses different m-ordering: (m=+1, m=-1, m=0) -> (x, y, z)
    if convention == "fairchem" and ell == 1:
        # Row 0: m=+1 (x-direction)
        # Y^{+1}(real) = (1/√2)[(-1)^1 Y^{+1} + Y^{-1}] = (1/√2)[-Y^{+1} + Y^{-1}]
        U_ell[0, 2] = -sqrt2_inv  # Y^{+1}(complex) at col 2
        U_ell[0, 0] = sqrt2_inv   # Y^{-1}(complex) at col 0

        # Row 1: m=-1 (y-direction, the polar axis)
        # Y^{-1}(real) = (i/√2)[Y^{+1} - (-1)^1 Y^{-1}] = (i/√2)[Y^{+1} + Y^{-1}]
        U_ell[1, 0] = 1j * sqrt2_inv   # Y^{-1}(complex)
        U_ell[1, 2] = 1j * sqrt2_inv   # Y^{+1}(complex)

        # Row 2: m=0 (z-direction)
        U_ell[2, 1] = 1.0
    else:
        # e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell)
        for m in range(-ell, ell + 1):
            row = m + ell  # Real harmonic index within block

            if m > 0:
                col_pos = m + ell
                col_neg = -m + ell
                sign = (-1) ** m
                U_ell[row, col_pos] = sign * sqrt2_inv
                U_ell[row, col_neg] = sqrt2_inv
            elif m == 0:
                col = ell
                U_ell[row, col] = 1.0
            else:  # m < 0
                abs_m = abs(m)
                col_pos = abs_m + ell
                col_neg = -abs_m + ell
                sign = (-1) ** abs_m
                U_ell[row, col_neg] = 1j * sqrt2_inv
                U_ell[row, col_pos] = -sign * 1j * sqrt2_inv

    return U_ell


def precompute_U_blocks(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    convention: str = "fairchem",
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices for each ell block.

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64) - will use corresponding complex type
        device: Torch device
        convention: "fairchem" (default) or "e3nn" for m-ordering convention

    Returns:
        List of U matrices where U_blocks[ell] has shape (2*ell+1, 2*ell+1)
    """
    # Convert real dtype to complex dtype
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    elif dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        # If already complex, use as-is
        complex_dtype = dtype

    return [_build_u_block(ell, complex_dtype, device, convention) for ell in range(lmax + 1)]


def precompute_complex_to_real_matrix(
    lmax: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = torch.device("cpu"),
    convention: str = "e3nn",
) -> torch.Tensor:
    """
    Compute the unitary matrix U that transforms complex → real spherical harmonics.

    Real spherical harmonics convention (matching e3nn):
        m > 0: Y^m_ℓ(real) = (1/√2) · [(-1)^m · Y^m_ℓ(complex) + Y^{-m}_ℓ(complex)]
        m = 0: Y^0_ℓ(real) = Y^0_ℓ(complex)
        m < 0: Y^m_ℓ(real) = (i/√2) · [Y^{-|m|}_ℓ(complex) - (-1)^{|m|} · Y^{|m|}_ℓ(complex)]

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type
        device: Device for output
        convention: Either "e3nn" (m = -l to +l) or "fairchem" (m = +1, -1, 0 for l=1)
                   fairchem uses y as polar axis with ordering (x, y, z) at indices (0, 1, 2)

    Returns:
        Block-diagonal unitary matrix of shape (size, size) where size = (lmax+1)²
    """
    blocks = precompute_U_blocks(lmax, dtype, device, convention)
    return torch.block_diag(*blocks)


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
    # Use explicit matmul which is faster than einsum for this pattern
    U_H = U.conj().T
    # D_complex @ U_H: (N, size, size) @ (size, size) -> (N, size, size)
    temp = torch.matmul(D_complex, U_H)
    # U @ temp: (size, size) @ (N, size, size) -> broadcasting
    D_real = torch.matmul(U, temp)

    # The result should be real for proper rotations
    return D_real.real


def wigner_d_complex_to_real_blockwise(
    D_complex: torch.Tensor,
    U_blocks: list[torch.Tensor],
    lmax: int,
    use_einsum: bool = False,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from complex to real, exploiting block structure.

    Since both D and U are block-diagonal, we can transform each block independently,
    which is O(sum of block_size³) instead of O(total_size³).

    Args:
        D_complex: Complex Wigner D matrices of shape (N, size, size)
        U_blocks: List of U matrices for each ell, U_blocks[ell] has shape (2*ell+1, 2*ell+1)
        lmax: Maximum angular momentum
        use_einsum: If True, use einsum instead of two matmuls (may be faster on CPU for large batches)

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    N = D_complex.shape[0]
    size = D_complex.shape[1]
    device = D_complex.device
    complex_dtype = D_complex.dtype
    real_dtype = D_complex.real.dtype

    D_real = torch.zeros(N, size, size, dtype=real_dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        # Extract block: (N, block_size, block_size)
        D_block = D_complex[:, block_start:block_end, block_start:block_end]

        # Get U block for this ell, cast to match input dtype
        U_ell = U_blocks[ell].to(dtype=complex_dtype, device=device)
        U_ell_H = U_ell.conj().T

        # Transform: U @ D @ U^H
        if use_einsum:
            # Single einsum operation: may be faster on CPU for large batches
            D_block_real = torch.einsum("ij,njk,kl->nil", U_ell, D_block, U_ell_H)
        else:
            # Two separate matmuls: generally faster on GPU
            temp = torch.matmul(D_block, U_ell_H)
            D_block_real = torch.matmul(U_ell, temp)

        # Store result
        D_real[:, block_start:block_end, block_start:block_end] = D_block_real.real

        block_start = block_end

    return D_real


# =============================================================================
# Main Pipeline Functions
# =============================================================================


def compute_wigner_d_from_quaternion(
    q: torch.Tensor,
    coeffs_sym: dict[str, torch.Tensor],
    U_blocks: list[torch.Tensor],
) -> torch.Tensor:
    """
    Compute real Wigner D matrices from quaternions.

    Uses optimized symmetric algorithm with blockwise complex-to-real transformation.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        coeffs_sym: Precomputed symmetric Wigner coefficients
        U_blocks: Precomputed U blocks for each ell

    Returns:
        Real block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)²
    """
    lmax = coeffs_sym["lmax"]
    Ra, Rb = quaternion_to_ra_rb(q)
    D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs_sym)
    D_real = wigner_d_complex_to_real_blockwise(D_complex, U_blocks, lmax)
    return D_real


def get_wigner_from_edge_vectors(
    edge_distance_vec: torch.Tensor,
    coeffs_sym: dict[str, torch.Tensor],
    U_blocks: list[torch.Tensor],
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete pipeline: edge vectors → real Wigner D matrices.

    This is the optimized quaternion-based implementation, using:
    - Symmetric coefficients (compute only ~50% of elements)
    - Blockwise complex-to-real transformation

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        coeffs_sym: Precomputed symmetric Wigner coefficients
        U_blocks: Precomputed U blocks for each ell
        gamma: Optional rotation angles around edge axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
        and size = (lmax+1)². wigner rotates from edge frame to lab frame,
        wigner_inv rotates from lab frame to edge frame.
    """
    q = edge_to_quaternion(edge_distance_vec, gamma=gamma)
    wigner = compute_wigner_d_from_quaternion(q, coeffs_sym, U_blocks)
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
    return wigner_inv, wigner


# =============================================================================
# Disk Caching for Precomputed Coefficients
# =============================================================================


def _get_cache_path(
    lmax: int,
    variant: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Get the cache file path for a given lmax and variant."""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    # Create a hash of the parameters to ensure cache invalidation on code changes
    # Include a version string that should be bumped when the computation changes
    version = "v1"
    return cache_dir / f"wigner_{variant}_lmax{lmax}_{version}.pt"


def _save_coefficients(coeffs: dict[str, torch.Tensor], path: Path) -> None:
    """Save coefficients to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move tensors to CPU for storage
    coeffs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in coeffs.items()}
    torch.save(coeffs_cpu, path)


def _load_coefficients(
    path: Path,
    device: torch.device,
) -> Optional[dict[str, torch.Tensor]]:
    """Load coefficients from disk, returning None if not found or invalid."""
    if not path.exists():
        return None
    try:
        coeffs = torch.load(path, map_location=device, weights_only=True)
        return coeffs
    except Exception:
        # Cache is corrupted or incompatible, return None to recompute
        return None


def get_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Get precomputed Wigner D coefficients, loading from cache if available.

    This is the recommended way to get coefficients - it will:
    1. Check if cached coefficients exist on disk
    2. If yes, load and return them
    3. If no, compute them, save to disk, and return

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        cache_dir: Directory for cache files (default: ~/.cache/fairchem/wigner_coeffs)
        use_cache: Whether to use disk caching (default: True)

    Returns:
        Dictionary with precomputed coefficient tensors
    """
    cache_path = _get_cache_path(lmax, "symmetric", cache_dir)

    # Try to load from cache
    if use_cache:
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None:
            # Verify lmax matches
            if coeffs.get("lmax") == lmax:
                # Convert dtype if needed (cache stores in float64)
                if dtype != torch.float64:
                    coeffs = {
                        k: v.to(dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                        for k, v in coeffs.items()
                    }
                return coeffs

    # Compute coefficients
    coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype, device)

    # Save to cache
    if use_cache:
        try:
            _save_coefficients(coeffs, cache_path)
        except Exception:
            # Failed to save cache, but we can still return the coefficients
            pass

    return coeffs


def get_complex_to_real_matrix(
    lmax: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = torch.device("cpu"),
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    convention: str = "fairchem",
) -> torch.Tensor:
    """
    Get precomputed complex-to-real transformation matrix, loading from cache if available.

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type for the matrix
        device: Device for the tensor
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching
        convention: Either "e3nn" or "fairchem" (default). Use "fairchem" to match
                   the rotation.py convention used in the rest of fairchem.

    Returns:
        Unitary transformation matrix of shape (size, size) where size = (lmax+1)²
    """
    cache_path = _get_cache_path(lmax, f"U_matrix_{convention}", cache_dir)

    if use_cache:
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None and "U" in coeffs and coeffs.get("lmax") == lmax:
            U = coeffs["U"]
            if U.dtype != dtype:
                U = U.to(dtype=dtype)
            return U

    # Compute
    U = precompute_complex_to_real_matrix(lmax, dtype, device, convention=convention)

    # Save to cache
    if use_cache:
        try:
            _save_coefficients({"U": U, "lmax": lmax}, cache_path)
        except Exception:
            pass

    return U


def precompute_all_wigner_tables(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    convention: str = "fairchem",
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """
    Convenience function to get both coefficient tables and U blocks.

    This is the recommended single entry point for initialization.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for real coefficients
        device: Device for tensors
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching
        convention: Either "e3nn" or "fairchem" (default). Use "fairchem" to match
                   the rotation.py convention used in the rest of fairchem.

    Returns:
        Tuple of (coeffs, U_blocks) ready for use with get_wigner_from_edge_vectors

    Example:
        >>> coeffs, U_blocks = precompute_all_wigner_tables(lmax=6)
        >>> wigner, wigner_inv = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
    """
    coeffs = get_wigner_coefficients(lmax, dtype, device, cache_dir, use_cache)
    U_blocks = precompute_U_blocks(lmax, dtype=dtype, device=device, convention=convention)

    return coeffs, U_blocks


def clear_wigner_cache(cache_dir: Optional[Path] = None) -> int:
    """
    Clear all cached Wigner coefficient files.

    Args:
        cache_dir: Directory to clear (default: ~/.cache/fairchem/wigner_coeffs)

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return 0

    count = 0
    for f in cache_dir.glob("wigner_*.pt"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass

    return count


# =============================================================================
# Simple API with automatic caching
# =============================================================================

# Module-level cache for precomputed coefficients
_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_U_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}


def _get_cached_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached coefficients, computing if necessary."""
    key = (lmax, dtype, device)

    if key not in _COEFF_CACHE:
        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)
        _COEFF_CACHE[key] = coeffs

    if key not in _U_CACHE:
        U_blocks = precompute_U_blocks(lmax, dtype=dtype, device=device)
        _U_CACHE[key] = U_blocks

    return _COEFF_CACHE[key], _U_CACHE[key]


def quaternion_wigner(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices from edge vectors using quaternions.

    This is the main entry point for quaternion-based Wigner D computation.
    Precomputed coefficients are cached automatically.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional rotation angles around edge axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
        and size = (lmax+1)². wigner rotates from edge frame to lab frame,
        wigner_inv rotates from lab frame to edge frame.
    """
    dtype = edge_distance_vec.dtype
    device = edge_distance_vec.device

    coeffs, U_blocks = _get_cached_coefficients(lmax, dtype, device)

    return get_wigner_from_edge_vectors(
        edge_distance_vec, coeffs, U_blocks, gamma=gamma
    )
