"""
Quaternion-based Wigner D Matrix Construction

This module implements Wigner D matrix computation using quaternions, following
the spherical_functions package by Mike Boyle (https://github.com/moble/spherical_functions).

Convention
----------
Throughout this module, the fundamental operation is computing a Wigner D matrix
that rotates spherical harmonics so that a given edge direction aligns with
the +Y axis. This is denoted as the "edge → +Y" rotation.

For l=1 spherical harmonics in Cartesian basis (x, y, z), this Wigner D matrix
acts like a 3x3 rotation matrix R such that R @ edge_direction = [0, 1, 0].
For higher l values, the Wigner D extends this rotation to the full spherical
harmonic basis.

The internal functions (_quat_edge_to_y_euler, _quat_edge_to_y_euler_alt,
edge_to_quaternion, get_wigner_from_edge_vectors) all follow this edge → +Y
convention consistently.

The public function `quaternion_wigner` returns (wigner_edge_to_y, wigner_y_to_edge)
matching the convention used by the Euler-angle-based rotation.py module in fairchem.

Key properties:
- NO arccos or atan2 on edge vector components
- NO Euler angle computation
- Numerically stable for all edge orientations including ±Y aligned
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
        gamma: Optional rotation angles for Z-rotation component, shape (N,).
               If None, random angles in [0, 2π) are generated.
               To match Euler code with gamma_orig, pass gamma = -gamma_orig.

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention for edge → +Y
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
# Real-Pair Arithmetic Helpers (torch.compile compatible)
# =============================================================================


def quaternion_to_ra_rb_real(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into real/imaginary parts of Ra and Rb.

    This is a torch.compile-compatible alternative to quaternion_to_ra_rb
    that avoids creating complex tensors.

    For q = (w, x, y, z):
        Ra = w + i·z  →  (ra_re=w, ra_im=z)
        Rb = y + i·x  →  (rb_re=y, rb_im=x)

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) convention

    Returns:
        Tuple (ra_re, ra_im, rb_re, rb_im) of real tensors with shape (...)
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    return w, z, y, x  # ra_re=w, ra_im=z, rb_re=y, rb_im=x


def _complex_mul_real(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multiply two complex numbers in real arithmetic.

    (a + ib)(c + id) = (ac - bd) + i(ad + bc)

    Args:
        a_re, a_im: Real and imaginary parts of first complex number
        b_re, b_im: Real and imaginary parts of second complex number

    Returns:
        Tuple (result_re, result_im)
    """
    return (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)


def _exp_i_theta(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exp(i*theta) = cos(theta) + i*sin(theta) in real arithmetic.

    Args:
        theta: Angle tensor

    Returns:
        Tuple (cos(theta), sin(theta))
    """
    return torch.cos(theta), torch.sin(theta)


# =============================================================================
# Precomputation of Wigner Coefficients (Symmetric Version)
# =============================================================================


def precompute_wigner_coefficients_range(
    lmin: int,
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Precompute Wigner D coefficients for l in [lmin, lmax] only.

    This is a memory-efficient variant that computes only the requested l range,
    useful for hybrid approaches where l=0,1,2 are computed via other methods.

    Uses the symmetry D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m}) to compute
    only ~half the elements ("primary") and derive the rest ("derived").

    Args:
        lmin: Minimum angular momentum to include
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors

    Returns:
        Dictionary with symmetric coefficient tables for l in [lmin, lmax]
    """
    factorial = _factorial_table(2 * lmax + 1, dtype, device)

    # Count elements only for l >= lmin
    n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))

    # Compute block_offset: number of rows/cols in l < lmin
    # This is lmin^2 (sum of (2*ell+1) for ell in [0, lmin-1])
    block_offset = lmin * lmin

    # Count primary elements for l >= lmin
    n_primary = 0
    for ell in range(lmin, lmax + 1):
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

    # Derived element info
    n_derived = n_total - n_primary
    derived_row_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_col_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_primary_idx = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_sign = torch.zeros(n_derived, dtype=dtype, device=device)

    # Build mapping from (row, col) to primary index for lookup
    primary_map = {}  # (row, col) -> primary_idx

    primary_idx = 0
    derived_idx = 0

    # block_start is now relative to the reduced matrix (starting at lmin)
    block_start = 0

    for ell in range(lmin, lmax + 1):
        block_size = 2 * ell + 1

        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell

                # Row/col relative to the reduced matrix
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
    for ell in range(lmin, lmax + 1):
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

    # Size of the reduced matrix
    size = (lmax + 1) ** 2 - lmin ** 2

    return {
        "lmin": lmin,
        "lmax": lmax,
        "block_offset": block_offset,
        "size": size,
        "n_primary": n_primary,
        "n_derived": n_derived,
        "n_total": n_total,
        "max_poly_len": max_poly_len,
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


def wigner_d_matrix_complex_range(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    coeffs_range: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute complex Wigner D matrices for l in [lmin, lmax] only.

    This is an optimized variant that computes only the requested l range,
    outputting a reduced-size matrix.

    Args:
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        coeffs_range: Precomputed coefficient dictionary from precompute_wigner_coefficients_range

    Returns:
        Complex block-diagonal matrices of shape (N, size, size)
        where size = (lmax+1)² - lmin²
    """
    N = Ra.shape[0]
    device = Ra.device
    complex_dtype = Ra.dtype
    real_dtype = Ra.real.dtype

    lmin = coeffs_range["lmin"]
    lmax = coeffs_range["lmax"]
    size = coeffs_range["size"]
    n_primary = coeffs_range["n_primary"]
    max_poly_len = coeffs_range["max_poly_len"]

    # Extract precomputed arrays
    primary_row_indices = coeffs_range["primary_row_indices"]
    primary_col_indices = coeffs_range["primary_col_indices"]

    case1_coeff = coeffs_range["case1_coeff"]
    case1_horner = coeffs_range["case1_horner"]
    case1_poly_len = coeffs_range["case1_poly_len"]
    case1_ra_exp = coeffs_range["case1_ra_exp"]
    case1_rb_exp = coeffs_range["case1_rb_exp"]
    case1_sign = coeffs_range["case1_sign"]

    case2_coeff = coeffs_range["case2_coeff"]
    case2_horner = coeffs_range["case2_horner"]
    case2_poly_len = coeffs_range["case2_poly_len"]
    case2_ra_exp = coeffs_range["case2_ra_exp"]
    case2_rb_exp = coeffs_range["case2_rb_exp"]
    case2_sign = coeffs_range["case2_sign"]

    mp_plus_m = coeffs_range["mp_plus_m"]
    m_minus_mp = coeffs_range["m_minus_mp"]

    diagonal_mask = coeffs_range["diagonal_mask"]
    anti_diagonal_mask = coeffs_range["anti_diagonal_mask"]
    special_2m = coeffs_range["special_2m"]
    anti_diag_sign = coeffs_range["anti_diag_sign"]

    derived_row_indices = coeffs_range["derived_row_indices"]
    derived_col_indices = coeffs_range["derived_col_indices"]
    derived_primary_idx = coeffs_range["derived_primary_idx"]
    derived_sign = coeffs_range["derived_sign"]

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
    log_abs_Rb = torch.log(torch.abs(safe_Rb))
    arg_Rb = torch.angle(safe_Rb)
    exponent = torch.outer(log_abs_Rb, special_2m) + 1j * torch.outer(arg_Rb, special_2m)
    rb_power = torch.exp(exponent.to(dtype=complex_dtype))
    special_val_antidiag = anti_diag_sign.unsqueeze(0).to(complex_dtype) * rb_power
    mask_antidiag = ra_small.unsqueeze(1) & anti_diagonal_mask.unsqueeze(0)
    result = torch.where(mask_antidiag, special_val_antidiag, result)

    # ==========================================================================
    # Special Case 2: |Rb| ≈ 0 - diagonal elements
    # ==========================================================================
    safe_Ra = torch.where(ra < EPSILON, torch.ones_like(Ra), Ra)
    log_abs_Ra = torch.log(torch.abs(safe_Ra))
    arg_Ra = torch.angle(safe_Ra)
    exponent = torch.outer(log_abs_Ra, special_2m) + 1j * torch.outer(arg_Ra, special_2m)
    ra_power = torch.exp(exponent.to(dtype=complex_dtype))
    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & diagonal_mask.unsqueeze(0)
    result = torch.where(mask_diag, ra_power, result)

    # ==========================================================================
    # General Case 1: |Ra| >= |Rb|
    # ==========================================================================
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    horner_sum1 = _vectorized_horner(ratio1, case1_horner, case1_poly_len, max_poly_len)

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
    # ==========================================================================
    n_derived = coeffs_range["n_derived"]
    if n_derived > 0:
        primary_vals = result[:, derived_primary_idx]
        derived_vals = derived_sign.unsqueeze(0).to(complex_dtype) * primary_vals.conj()

        batch_indices_d = torch.arange(N, device=device).unsqueeze(1).expand(N, n_derived)
        row_expanded_d = derived_row_indices.unsqueeze(0).expand(N, n_derived)
        col_expanded_d = derived_col_indices.unsqueeze(0).expand(N, n_derived)

        D[batch_indices_d, row_expanded_d, col_expanded_d] = derived_vals

    return D


# =============================================================================
# Real-Pair Wigner D Matrix Computation (torch.compile compatible)
# =============================================================================


def wigner_d_matrix_real(
    ra_re: torch.Tensor,
    ra_im: torch.Tensor,
    rb_re: torch.Tensor,
    rb_im: torch.Tensor,
    coeffs_sym: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices using real arithmetic only.

    This is a torch.compile-compatible alternative to wigner_d_matrix_complex
    that avoids creating complex tensors. All complex operations are replaced
    with their real-pair equivalents.

    Computes only primary elements (~half) and derives the rest via:
        D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m})

    Args:
        ra_re, ra_im: Real and imaginary parts of Ra, shape (N,)
        rb_re, rb_im: Real and imaginary parts of Rb, shape (N,)
        coeffs_sym: Precomputed symmetric coefficient dictionary

    Returns:
        Tuple (D_re, D_im) - real and imaginary parts of the complex
        block-diagonal matrices, each of shape (N, size, size)
    """
    N = ra_re.shape[0]
    device = ra_re.device
    dtype = ra_re.dtype

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

    # Compute magnitudes: |Ra|, |Rb|
    ra = torch.sqrt(ra_re * ra_re + ra_im * ra_im)
    rb = torch.sqrt(rb_re * rb_re + rb_im * rb_im)

    # Case masks
    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    # Phase angles: arg(Ra), arg(Rb)
    phia = torch.atan2(ra_im, ra_re)
    phib = torch.atan2(rb_im, rb_re)

    # Phase = phia * (m' + m) + phib * (m - m')
    # exp(i * phase) = (cos(phase), sin(phase))
    phase = torch.outer(phia, mp_plus_m) + torch.outer(phib, m_minus_mp)
    exp_phase_re = torch.cos(phase)
    exp_phase_im = torch.sin(phase)

    # Safe magnitudes and their logs for power computation
    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    # Initialize primary results as real pairs: (N, n_primary)
    result_re = torch.zeros(N, n_primary, dtype=dtype, device=device)
    result_im = torch.zeros(N, n_primary, dtype=dtype, device=device)

    # ==========================================================================
    # Special Case 1: |Ra| ≈ 0 - anti-diagonal elements
    # Rb^(2m) where 2m is special_2m
    # z^n = |z|^n * exp(i*n*arg(z)) = |z|^n * (cos(n*arg), sin(n*arg))
    # ==========================================================================
    safe_rb_mag = torch.where(rb < EPSILON, torch.ones_like(rb), rb)
    log_safe_rb = torch.log(safe_rb_mag)
    arg_rb = torch.atan2(rb_im, rb_re)

    # |Rb|^(2m) * exp(i * 2m * arg(Rb))
    log_mag_rb_power = torch.outer(log_safe_rb, special_2m)  # n * log|z|
    rb_power_mag = torch.exp(log_mag_rb_power)
    rb_power_phase = torch.outer(arg_rb, special_2m)  # n * arg(z)
    rb_power_re = rb_power_mag * torch.cos(rb_power_phase)
    rb_power_im = rb_power_mag * torch.sin(rb_power_phase)

    # anti_diag_sign * rb_power (sign is real)
    special_val_antidiag_re = anti_diag_sign.unsqueeze(0) * rb_power_re
    special_val_antidiag_im = anti_diag_sign.unsqueeze(0) * rb_power_im

    mask_antidiag = ra_small.unsqueeze(1) & anti_diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_antidiag, special_val_antidiag_re, result_re)
    result_im = torch.where(mask_antidiag, special_val_antidiag_im, result_im)

    # ==========================================================================
    # Special Case 2: |Rb| ≈ 0 - diagonal elements
    # Ra^(2m)
    # ==========================================================================
    safe_ra_mag = torch.where(ra < EPSILON, torch.ones_like(ra), ra)
    log_safe_ra = torch.log(safe_ra_mag)
    arg_ra = torch.atan2(ra_im, ra_re)

    # |Ra|^(2m) * exp(i * 2m * arg(Ra))
    log_mag_ra_power = torch.outer(log_safe_ra, special_2m)
    ra_power_mag = torch.exp(log_mag_ra_power)
    ra_power_phase = torch.outer(arg_ra, special_2m)
    ra_power_re = ra_power_mag * torch.cos(ra_power_phase)
    ra_power_im = ra_power_mag * torch.sin(ra_power_phase)

    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_diag, ra_power_re, result_re)
    result_im = torch.where(mask_diag, ra_power_im, result_im)

    # ==========================================================================
    # General Case 1: |Ra| >= |Rb|
    # ==========================================================================
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    horner_sum1 = _vectorized_horner(ratio1, case1_horner, case1_poly_len, max_poly_len)

    # Power computation using log-exp trick with torch.outer
    ra_powers1 = torch.exp(torch.outer(log_ra, case1_ra_exp))
    rb_powers1 = torch.exp(torch.outer(log_rb, case1_rb_exp))

    # magnitude1 is real: (sign * coeff) * |ra|^exp * |rb|^exp
    magnitude1 = (case1_sign * case1_coeff) * ra_powers1 * rb_powers1

    # val1 = magnitude1 * horner_sum1 * exp(i*phase)
    # magnitude1 * horner_sum1 is real, multiply by exp(i*phase)
    real_factor1 = magnitude1 * horner_sum1
    val1_re = real_factor1 * exp_phase_re
    val1_im = real_factor1 * exp_phase_im

    valid_case1 = case1_poly_len > 0
    mask1 = use_case1.unsqueeze(1) & valid_case1.unsqueeze(0)
    result_re = torch.where(mask1, val1_re, result_re)
    result_im = torch.where(mask1, val1_im, result_im)

    # ==========================================================================
    # General Case 2: |Ra| < |Rb|
    # ==========================================================================
    ratio2 = -(ra * ra) / (safe_rb * safe_rb)
    horner_sum2 = _vectorized_horner(ratio2, case2_horner, case2_poly_len, max_poly_len)

    ra_powers2 = torch.exp(torch.outer(log_ra, case2_ra_exp))
    rb_powers2 = torch.exp(torch.outer(log_rb, case2_rb_exp))

    magnitude2 = (case2_sign * case2_coeff) * ra_powers2 * rb_powers2
    real_factor2 = magnitude2 * horner_sum2
    val2_re = real_factor2 * exp_phase_re
    val2_im = real_factor2 * exp_phase_im

    valid_case2 = case2_poly_len > 0
    mask2 = use_case2.unsqueeze(1) & valid_case2.unsqueeze(0)
    result_re = torch.where(mask2, val2_re, result_re)
    result_im = torch.where(mask2, val2_im, result_im)

    # ==========================================================================
    # Scatter primary results into output matrix
    # ==========================================================================
    D_re = torch.zeros(N, size, size, dtype=dtype, device=device)
    D_im = torch.zeros(N, size, size, dtype=dtype, device=device)

    batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, n_primary)
    row_expanded = primary_row_indices.unsqueeze(0).expand(N, n_primary)
    col_expanded = primary_col_indices.unsqueeze(0).expand(N, n_primary)

    D_re[batch_indices, row_expanded, col_expanded] = result_re
    D_im[batch_indices, row_expanded, col_expanded] = result_im

    # ==========================================================================
    # Fill derived elements using symmetry
    # D^ℓ_{-m',-m} = (-1)^{m'-m} × conj(D^ℓ_{m',m})
    # conj(a + ib) = a - ib, so derived_re = sign * primary_re, derived_im = -sign * primary_im
    # ==========================================================================
    n_derived = coeffs_sym["n_derived"]
    if n_derived > 0:
        # Get the primary values for each derived element
        primary_re = result_re[:, derived_primary_idx]  # (N, n_derived)
        primary_im = result_im[:, derived_primary_idx]  # (N, n_derived)

        # Apply symmetry: conjugate and multiply by sign
        # derived = sign * conj(primary) = sign * (re - i*im) = (sign*re, -sign*im)
        derived_sign_expanded = derived_sign.unsqueeze(0)
        derived_re = derived_sign_expanded * primary_re
        derived_im = -derived_sign_expanded * primary_im

        # Scatter derived values
        batch_indices_d = torch.arange(N, device=device).unsqueeze(1).expand(N, n_derived)
        row_expanded_d = derived_row_indices.unsqueeze(0).expand(N, n_derived)
        col_expanded_d = derived_col_indices.unsqueeze(0).expand(N, n_derived)

        D_re[batch_indices_d, row_expanded_d, col_expanded_d] = derived_re
        D_im[batch_indices_d, row_expanded_d, col_expanded_d] = derived_im

    return D_re, D_im


def wigner_d_matrix_real_range(
    ra_re: torch.Tensor,
    ra_im: torch.Tensor,
    rb_re: torch.Tensor,
    rb_im: torch.Tensor,
    coeffs_range: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices for l in [lmin, lmax] using real arithmetic only.

    This is the range version of wigner_d_matrix_real for computing only
    a subset of l values (e.g., l>=3 for hybrid methods).

    Args:
        ra_re, ra_im: Real and imaginary parts of Ra, shape (N,)
        rb_re, rb_im: Real and imaginary parts of Rb, shape (N,)
        coeffs_range: Precomputed range coefficient dictionary

    Returns:
        Tuple (D_re, D_im) - real and imaginary parts of the complex
        block-diagonal matrices, each of shape (N, size, size)
        where size = (lmax+1)² - lmin²
    """
    N = ra_re.shape[0]
    device = ra_re.device
    dtype = ra_re.dtype

    lmin = coeffs_range["lmin"]
    lmax = coeffs_range["lmax"]
    size = coeffs_range["size"]
    n_primary = coeffs_range["n_primary"]
    max_poly_len = coeffs_range["max_poly_len"]

    # Extract precomputed arrays
    primary_row_indices = coeffs_range["primary_row_indices"]
    primary_col_indices = coeffs_range["primary_col_indices"]

    case1_coeff = coeffs_range["case1_coeff"]
    case1_horner = coeffs_range["case1_horner"]
    case1_poly_len = coeffs_range["case1_poly_len"]
    case1_ra_exp = coeffs_range["case1_ra_exp"]
    case1_rb_exp = coeffs_range["case1_rb_exp"]
    case1_sign = coeffs_range["case1_sign"]

    case2_coeff = coeffs_range["case2_coeff"]
    case2_horner = coeffs_range["case2_horner"]
    case2_poly_len = coeffs_range["case2_poly_len"]
    case2_ra_exp = coeffs_range["case2_ra_exp"]
    case2_rb_exp = coeffs_range["case2_rb_exp"]
    case2_sign = coeffs_range["case2_sign"]

    mp_plus_m = coeffs_range["mp_plus_m"]
    m_minus_mp = coeffs_range["m_minus_mp"]

    diagonal_mask = coeffs_range["diagonal_mask"]
    anti_diagonal_mask = coeffs_range["anti_diagonal_mask"]
    special_2m = coeffs_range["special_2m"]
    anti_diag_sign = coeffs_range["anti_diag_sign"]

    derived_row_indices = coeffs_range["derived_row_indices"]
    derived_col_indices = coeffs_range["derived_col_indices"]
    derived_primary_idx = coeffs_range["derived_primary_idx"]
    derived_sign = coeffs_range["derived_sign"]

    # Compute magnitudes: |Ra|, |Rb|
    ra = torch.sqrt(ra_re * ra_re + ra_im * ra_im)
    rb = torch.sqrt(rb_re * rb_re + rb_im * rb_im)

    # Case masks
    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    # Phase angles: arg(Ra), arg(Rb)
    phia = torch.atan2(ra_im, ra_re)
    phib = torch.atan2(rb_im, rb_re)

    # Phase = phia * (m' + m) + phib * (m - m')
    phase = torch.outer(phia, mp_plus_m) + torch.outer(phib, m_minus_mp)
    exp_phase_re = torch.cos(phase)
    exp_phase_im = torch.sin(phase)

    # Safe magnitudes and their logs for power computation
    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    # Initialize primary results as real pairs: (N, n_primary)
    result_re = torch.zeros(N, n_primary, dtype=dtype, device=device)
    result_im = torch.zeros(N, n_primary, dtype=dtype, device=device)

    # Special Case 1: |Ra| ≈ 0 - anti-diagonal elements
    safe_rb_mag = torch.where(rb < EPSILON, torch.ones_like(rb), rb)
    log_safe_rb = torch.log(safe_rb_mag)
    arg_rb = torch.atan2(rb_im, rb_re)

    log_mag_rb_power = torch.outer(log_safe_rb, special_2m)
    rb_power_mag = torch.exp(log_mag_rb_power)
    rb_power_phase = torch.outer(arg_rb, special_2m)
    rb_power_re = rb_power_mag * torch.cos(rb_power_phase)
    rb_power_im = rb_power_mag * torch.sin(rb_power_phase)

    special_val_antidiag_re = anti_diag_sign.unsqueeze(0) * rb_power_re
    special_val_antidiag_im = anti_diag_sign.unsqueeze(0) * rb_power_im

    mask_antidiag = ra_small.unsqueeze(1) & anti_diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_antidiag, special_val_antidiag_re, result_re)
    result_im = torch.where(mask_antidiag, special_val_antidiag_im, result_im)

    # Special Case 2: |Rb| ≈ 0 - diagonal elements
    safe_ra_mag = torch.where(ra < EPSILON, torch.ones_like(ra), ra)
    log_safe_ra = torch.log(safe_ra_mag)
    arg_ra = torch.atan2(ra_im, ra_re)

    log_mag_ra_power = torch.outer(log_safe_ra, special_2m)
    ra_power_mag = torch.exp(log_mag_ra_power)
    ra_power_phase = torch.outer(arg_ra, special_2m)
    ra_power_re = ra_power_mag * torch.cos(ra_power_phase)
    ra_power_im = ra_power_mag * torch.sin(ra_power_phase)

    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_diag, ra_power_re, result_re)
    result_im = torch.where(mask_diag, ra_power_im, result_im)

    # General Case 1: |Ra| >= |Rb|
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    horner_sum1 = _vectorized_horner(ratio1, case1_horner, case1_poly_len, max_poly_len)

    ra_powers1 = torch.exp(torch.outer(log_ra, case1_ra_exp))
    rb_powers1 = torch.exp(torch.outer(log_rb, case1_rb_exp))

    magnitude1 = (case1_sign * case1_coeff) * ra_powers1 * rb_powers1
    real_factor1 = magnitude1 * horner_sum1
    val1_re = real_factor1 * exp_phase_re
    val1_im = real_factor1 * exp_phase_im

    valid_case1 = case1_poly_len > 0
    mask1 = use_case1.unsqueeze(1) & valid_case1.unsqueeze(0)
    result_re = torch.where(mask1, val1_re, result_re)
    result_im = torch.where(mask1, val1_im, result_im)

    # General Case 2: |Ra| < |Rb|
    ratio2 = -(ra * ra) / (safe_rb * safe_rb)
    horner_sum2 = _vectorized_horner(ratio2, case2_horner, case2_poly_len, max_poly_len)

    ra_powers2 = torch.exp(torch.outer(log_ra, case2_ra_exp))
    rb_powers2 = torch.exp(torch.outer(log_rb, case2_rb_exp))

    magnitude2 = (case2_sign * case2_coeff) * ra_powers2 * rb_powers2
    real_factor2 = magnitude2 * horner_sum2
    val2_re = real_factor2 * exp_phase_re
    val2_im = real_factor2 * exp_phase_im

    valid_case2 = case2_poly_len > 0
    mask2 = use_case2.unsqueeze(1) & valid_case2.unsqueeze(0)
    result_re = torch.where(mask2, val2_re, result_re)
    result_im = torch.where(mask2, val2_im, result_im)

    # Scatter primary results into output matrix
    D_re = torch.zeros(N, size, size, dtype=dtype, device=device)
    D_im = torch.zeros(N, size, size, dtype=dtype, device=device)

    batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, n_primary)
    row_expanded = primary_row_indices.unsqueeze(0).expand(N, n_primary)
    col_expanded = primary_col_indices.unsqueeze(0).expand(N, n_primary)

    D_re[batch_indices, row_expanded, col_expanded] = result_re
    D_im[batch_indices, row_expanded, col_expanded] = result_im

    # Fill derived elements using symmetry
    n_derived = coeffs_range["n_derived"]
    if n_derived > 0:
        primary_re = result_re[:, derived_primary_idx]
        primary_im = result_im[:, derived_primary_idx]

        derived_sign_expanded = derived_sign.unsqueeze(0)
        derived_re = derived_sign_expanded * primary_re
        derived_im = -derived_sign_expanded * primary_im

        batch_indices_d = torch.arange(N, device=device).unsqueeze(1).expand(N, n_derived)
        row_expanded_d = derived_row_indices.unsqueeze(0).expand(N, n_derived)
        col_expanded_d = derived_col_indices.unsqueeze(0).expand(N, n_derived)

        D_re[batch_indices_d, row_expanded_d, col_expanded_d] = derived_re
        D_im[batch_indices_d, row_expanded_d, col_expanded_d] = derived_im

    return D_re, D_im


def wigner_d_pair_to_real_range(
    D_re: torch.Tensor,
    D_im: torch.Tensor,
    U_blocks_range_real: list[tuple[torch.Tensor, torch.Tensor]],
    lmin: int,
    lmax: int,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from real-pair to real basis for l in [lmin, lmax].

    This is a torch.compile-compatible alternative to wigner_d_complex_to_real_range.

    Args:
        D_re: Real part of complex Wigner D matrices, shape (N, size, size)
        D_im: Imaginary part of complex Wigner D matrices, shape (N, size, size)
        U_blocks_range_real: List of (U_re, U_im) tuples for l in [lmin, lmax]
        lmin: Minimum angular momentum
        lmax: Maximum angular momentum

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    N = D_re.shape[0]
    size = D_re.shape[1]
    device = D_re.device
    dtype = D_re.dtype

    D_real = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for idx, ell in enumerate(range(lmin, lmax + 1)):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        # Extract blocks
        D_block_re = D_re[:, block_start:block_end, block_start:block_end]
        D_block_im = D_im[:, block_start:block_end, block_start:block_end]

        # Get U block for this ell (indexed by idx, not ell)
        U_re, U_im = U_blocks_range_real[idx]
        U_re = U_re.to(dtype=dtype, device=device)
        U_im = U_im.to(dtype=dtype, device=device)

        U_re_T = U_re.T
        U_im_T = U_im.T

        # Compute U @ D @ U^H using real arithmetic
        temp_re = torch.matmul(D_block_re, U_re_T) + torch.matmul(D_block_im, U_im_T)
        temp_im = torch.matmul(D_block_im, U_re_T) - torch.matmul(D_block_re, U_im_T)

        result_re = torch.matmul(U_re, temp_re) - torch.matmul(U_im, temp_im)

        D_real[:, block_start:block_end, block_start:block_end] = result_re

        block_start = block_end

    return D_real


# =============================================================================
# Complex to Real Spherical Harmonics Transformation
# =============================================================================


def _build_u_block(
    ell: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build U transformation matrix for a single ell block.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        ell: Angular momentum quantum number
        dtype: Complex data type for the matrix
        device: Device for the tensor

    Returns:
        U matrix of shape (2*ell+1, 2*ell+1)
    """
    block_size = 2 * ell + 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    U_ell = torch.zeros(block_size, block_size, dtype=dtype, device=device)

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
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices for each ell block.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64) - will use corresponding complex type
        device: Torch device

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

    return [_build_u_block(ell, complex_dtype, device) for ell in range(lmax + 1)]


def precompute_U_blocks_range(
    lmin: int,
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices for l in [lmin, lmax] only.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        lmin: Minimum angular momentum
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64) - will use corresponding complex type
        device: Torch device

    Returns:
        List of U matrices where U_blocks[i] corresponds to l=lmin+i
    """
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    elif dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = dtype

    return [_build_u_block(ell, complex_dtype, device) for ell in range(lmin, lmax + 1)]


def precompute_U_blocks_euler_aligned(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices with Euler basis alignment folded in.

    This combines the complex→real transformation with:
    - For l=1: The Cartesian permutation P (m-ordering → x,y,z)
    - For l>=2: The Euler-matching basis transformation

    Using these combined U blocks eliminates the need for separate
    apply_euler_transform and l=1 permutation steps, reducing computation.

    The combined transformation is:
        D_euler_real = U_combined @ D_complex @ U_combined^H

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device

    Returns:
        List of combined U matrices where U_blocks[ell] has shape (2*ell+1, 2*ell+1)
    """
    # Get standard complex→real U blocks
    U_blocks = precompute_U_blocks(lmax, dtype, device)

    # Get complex dtype for combining
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        complex_dtype = torch.complex128

    # Build l=1 permutation matrix P: (y,z,x) -> (x,y,z)
    # This is real, but we treat it as complex for composition
    P = torch.tensor([
        [0., 0., 1.],  # x from position 2
        [1., 0., 0.],  # y from position 0
        [0., 1., 0.]   # z from position 1
    ], dtype=complex_dtype, device=device)

    # Load Jd matrices for Euler transforms (l >= 2)
    jd_path = Path(__file__).parent.parent / "Jd.pt"
    Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

    U_combined = []
    for ell in range(lmax + 1):
        U_ell = U_blocks[ell].to(dtype=complex_dtype, device=device)

        if ell == 0:
            # l=0: no transformation needed
            U_combined.append(U_ell)
        elif ell == 1:
            # l=1: fold in Cartesian permutation P
            # D_cartesian = P @ D_m_ordered @ P.T
            # D_m_ordered = U @ D_complex @ U^H
            # D_cartesian = P @ U @ D_complex @ U^H @ P.T = (P @ U) @ D_complex @ (P @ U)^H
            U_combined.append(P @ U_ell)
        else:
            # l>=2: fold in Euler-matching transformation
            # D_euler = U_euler @ D_axis @ U_euler.T
            # Since U_euler is real orthogonal: U_euler.T = U_euler^H
            # D_euler = U_euler @ U @ D_complex @ U^H @ U_euler^H = (U_euler @ U) @ D_complex @ (U_euler @ U)^H
            from fairchem.core.models.uma.common.wigner_d_axis_angle import _build_euler_transform
            Jd = Jd_list[ell].to(dtype=dtype, device=device)
            U_euler = _build_euler_transform(ell, Jd).to(dtype=complex_dtype, device=device)
            U_combined.append(U_euler @ U_ell)

    return U_combined


def precompute_U_blocks_euler_aligned_range(
    lmin: int,
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """
    Precompute Euler-aligned U transformation matrices for l in [lmin, lmax] only.

    This is the range version of precompute_U_blocks_euler_aligned, for use
    in the hybrid Wigner D computation where only l >= lmin blocks are needed.

    Args:
        lmin: Minimum angular momentum
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device

    Returns:
        List of combined U matrices where U_blocks[i] corresponds to l=lmin+i
    """
    # Get the full Euler-aligned U blocks and return the range subset
    full_U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype, device)
    return full_U_blocks[lmin:]


def precompute_U_blocks_real(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute U transformation matrices as real/imag pairs.

    This is a torch.compile-compatible version that stores U blocks
    as (U_re, U_im) pairs instead of complex tensors.

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device

    Returns:
        List of (U_re, U_im) tuples where each has shape (2*ell+1, 2*ell+1)
    """
    U_blocks_complex = precompute_U_blocks(lmax, dtype=dtype, device=device)
    return [(U.real, U.imag) for U in U_blocks_complex]


def precompute_U_blocks_euler_aligned_real(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute Euler-aligned U transformation matrices as real/imag pairs.

    This is a torch.compile-compatible version that stores U blocks
    as (U_re, U_im) pairs instead of complex tensors.

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device

    Returns:
        List of (U_re, U_im) tuples where each has shape (2*ell+1, 2*ell+1)
    """
    U_blocks_complex = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
    return [(U.real.to(dtype=dtype), U.imag.to(dtype=dtype)) for U in U_blocks_complex]


def precompute_complex_to_real_matrix(
    lmax: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute the unitary matrix U that transforms complex → real spherical harmonics.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Real spherical harmonics convention (matching e3nn):
        m > 0: Y^m_ℓ(real) = (1/√2) · [(-1)^m · Y^m_ℓ(complex) + Y^{-m}_ℓ(complex)]
        m = 0: Y^0_ℓ(real) = Y^0_ℓ(complex)
        m < 0: Y^m_ℓ(real) = (i/√2) · [Y^{-|m|}_ℓ(complex) - (-1)^{|m|} · Y^{|m|}_ℓ(complex)]

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type
        device: Device for output

    Returns:
        Block-diagonal unitary matrix of shape (size, size) where size = (lmax+1)²
    """
    blocks = precompute_U_blocks(lmax, dtype, device)
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


def wigner_d_complex_to_real_range(
    D_complex: torch.Tensor,
    U_blocks_range: list[torch.Tensor],
    lmin: int,
    lmax: int,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from complex to real for l in [lmin, lmax].

    Since both D and U are block-diagonal, we can transform each block independently.

    Args:
        D_complex: Complex Wigner D matrices of shape (N, size, size)
                   where size = (lmax+1)² - lmin²
        U_blocks_range: List of U matrices for l in [lmin, lmax],
                        where U_blocks_range[i] corresponds to l=lmin+i
        lmin: Minimum angular momentum
        lmax: Maximum angular momentum

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
    for idx, ell in enumerate(range(lmin, lmax + 1)):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        # Extract block: (N, block_size, block_size)
        D_block = D_complex[:, block_start:block_end, block_start:block_end]

        # Get U block for this ell (indexed by idx, not ell)
        U_ell = U_blocks_range[idx].to(dtype=complex_dtype, device=device)
        U_ell_H = U_ell.conj().T

        # Transform: U @ D @ U^H
        temp = torch.matmul(D_block, U_ell_H)
        D_block_real = torch.matmul(U_ell, temp)

        # Store result
        D_real[:, block_start:block_end, block_start:block_end] = D_block_real.real

        block_start = block_end

    return D_real


def wigner_d_pair_to_real_blockwise(
    D_re: torch.Tensor,
    D_im: torch.Tensor,
    U_blocks_real: list[tuple[torch.Tensor, torch.Tensor]],
    lmax: int,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from real-pair to real basis using real arithmetic.

    This is a torch.compile-compatible alternative to wigner_d_complex_to_real_blockwise
    that uses real-pair U blocks and avoids complex tensor operations.

    Computes D_real = U @ D @ U^H using real arithmetic:
        result_re = U_re @ D_re @ U_re.T + U_re @ D_im @ U_im.T
                    + U_im @ D_re @ U_im.T - U_im @ D_im @ U_re.T
        result_im = U_re @ D_re @ U_im.T - U_re @ D_im @ U_re.T
                    - U_im @ D_re @ U_re.T - U_im @ D_im @ U_im.T

    For proper rotations, result_im ≈ 0 and we return result_re.

    Args:
        D_re: Real part of complex Wigner D matrices, shape (N, size, size)
        D_im: Imaginary part of complex Wigner D matrices, shape (N, size, size)
        U_blocks_real: List of (U_re, U_im) tuples for each ell
        lmax: Maximum angular momentum

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    N = D_re.shape[0]
    size = D_re.shape[1]
    device = D_re.device
    dtype = D_re.dtype

    D_real = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        # Extract blocks: (N, block_size, block_size)
        D_block_re = D_re[:, block_start:block_end, block_start:block_end]
        D_block_im = D_im[:, block_start:block_end, block_start:block_end]

        # Get U block for this ell as (U_re, U_im)
        U_re, U_im = U_blocks_real[ell]
        U_re = U_re.to(dtype=dtype, device=device)
        U_im = U_im.to(dtype=dtype, device=device)

        # U^H = (U_re - i*U_im)^T = U_re.T - i*U_im.T
        U_re_T = U_re.T
        U_im_T = U_im.T

        # Compute U @ D @ U^H using real arithmetic
        # (U_re + i*U_im) @ (D_re + i*D_im) @ (U_re.T - i*U_im.T)
        #
        # Step 1: temp = D @ U^H = (D_re + i*D_im) @ (U_re.T - i*U_im.T)
        # temp_re = D_re @ U_re.T + D_im @ U_im.T
        # temp_im = D_im @ U_re.T - D_re @ U_im.T
        temp_re = torch.matmul(D_block_re, U_re_T) + torch.matmul(D_block_im, U_im_T)
        temp_im = torch.matmul(D_block_im, U_re_T) - torch.matmul(D_block_re, U_im_T)

        # Step 2: result = U @ temp = (U_re + i*U_im) @ (temp_re + i*temp_im)
        # result_re = U_re @ temp_re - U_im @ temp_im
        # result_im = U_re @ temp_im + U_im @ temp_re
        result_re = torch.matmul(U_re, temp_re) - torch.matmul(U_im, temp_im)
        # result_im = torch.matmul(U_re, temp_im) + torch.matmul(U_im, temp_re)
        # For proper rotations, result_im should be ~0

        # Store result
        D_real[:, block_start:block_end, block_start:block_end] = result_re

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
    Complete pipeline: edge vectors → real Wigner D matrices (edge → +Y).

    Computes the Wigner D matrix that rotates spherical harmonics so that the
    edge direction aligns with +Y. For l=1, this is equivalent to a
    rotation matrix R where R @ edge_direction = [0, 1, 0].

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        coeffs_sym: Precomputed symmetric Wigner coefficients
        U_blocks: Precomputed U blocks for each ell
        gamma: Optional rotation angles around the initial Y axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
        - wigner_edge_to_y: the edge → +Y rotation (R @ edge = +Y for l=1)
        - wigner_y_to_edge: the +Y → edge rotation (R @ +Y = edge for l=1)
    """
    q = edge_to_quaternion(edge_distance_vec, gamma=gamma)
    # edge_to_quaternion returns quaternion for edge → +Y rotation
    wigner_edge_to_y = compute_wigner_d_from_quaternion(q, coeffs_sym, U_blocks)
    wigner_y_to_edge = torch.transpose(wigner_edge_to_y, 1, 2).contiguous()
    return wigner_edge_to_y, wigner_y_to_edge



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
) -> torch.Tensor:
    """
    Get precomputed complex-to-real transformation matrix, loading from cache if available.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type for the matrix
        device: Device for the tensor
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching

    Returns:
        Unitary transformation matrix of shape (size, size) where size = (lmax+1)²
    """
    cache_path = _get_cache_path(lmax, "U_matrix_e3nn", cache_dir)

    if use_cache:
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None and "U" in coeffs and coeffs.get("lmax") == lmax:
            U = coeffs["U"]
            if U.dtype != dtype:
                U = U.to(dtype=dtype)
            return U

    # Compute
    U = precompute_complex_to_real_matrix(lmax, dtype, device)

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
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """
    Convenience function to get both coefficient tables and U blocks.

    This is the recommended single entry point for initialization.
    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for real coefficients
        device: Device for tensors
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching

    Returns:
        Tuple of (coeffs, U_blocks) ready for use with get_wigner_from_edge_vectors

    Example:
        >>> coeffs, U_blocks = precompute_all_wigner_tables(lmax=6)
        >>> wigner, wigner_inv = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
    """
    coeffs = get_wigner_coefficients(lmax, dtype, device, cache_dir, use_cache)
    U_blocks = precompute_U_blocks(lmax, dtype=dtype, device=device)

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


def clear_memory_caches() -> None:
    """
    Clear all in-memory caches for Wigner coefficients.

    This clears the module-level dictionaries that cache precomputed
    coefficients and U blocks. Useful for testing or reducing memory usage.
    """
    global _COEFF_CACHE, _U_CACHE, _U_REAL_CACHE
    _COEFF_CACHE.clear()
    _U_CACHE.clear()
    _U_REAL_CACHE.clear()


# =============================================================================
# Simple API with automatic caching
# =============================================================================

# Module-level cache for precomputed coefficients
_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_U_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}
_U_REAL_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}


def _get_cached_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached coefficients with Euler-aligned U blocks."""
    key = (lmax, dtype, device)

    if key not in _COEFF_CACHE:
        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)
        _COEFF_CACHE[key] = coeffs

    if key not in _U_CACHE:
        U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
        _U_CACHE[key] = U_blocks

    return _COEFF_CACHE[key], _U_CACHE[key]


def _get_cached_coefficients_real(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached coefficients with Euler-aligned U blocks in real-pair form."""
    key = (lmax, dtype, device)

    if key not in _COEFF_CACHE:
        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)
        _COEFF_CACHE[key] = coeffs

    if key not in _U_REAL_CACHE:
        U_blocks_real = precompute_U_blocks_euler_aligned_real(lmax, dtype=dtype, device=device)
        _U_REAL_CACHE[key] = U_blocks_real

    return _COEFF_CACHE[key], _U_REAL_CACHE[key]


def quaternion_wigner(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices from edge vectors using quaternions.

    This is the main entry point for quaternion-based Wigner D computation.
    Matches the convention of axis_angle_wigner and Euler-based rotation.py.
    Precomputed coefficients are cached automatically.

    The gamma rotation is combined with the edge→Y quaternion before computing
    the Wigner D, avoiding the overhead of computing two separate Wigner D
    matrices and multiplying them.

    Uses Euler-aligned U blocks that fold in the l=1 Cartesian permutation
    and l>=2 Euler basis transformation, eliminating separate transformation steps.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional rotation angles around the initial Y axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
        - wigner_edge_to_y: rotates edge → +Y (R @ edge = +Y for l=1)
        - wigner_y_to_edge: rotates +Y → edge (R @ +Y = edge for l=1)

        This matches the return convention of the Euler-based rotation.py.
    """
    # Import axis_angle helpers (they provide the correct quaternion computation)
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        quaternion_edge_to_y_stable,
        quaternion_multiply,
        quaternion_y_rotation,
    )

    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    dtype = edge_distance_vec.dtype
    device = edge_distance_vec.device

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge → +Y) using SLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge→Y
    # Combined rotation: first edge→Y, then rotate about Y by gamma
    # q_combined = q_gamma * q_edge_to_y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Get cached coefficients
    # U blocks have l=1 Cartesian permutation and l>=2 Euler transform folded in
    coeffs, U_blocks = _get_cached_coefficients(lmax, dtype, device)

    # Step 6: Compute Wigner D from combined quaternion using Ra/Rb polynomial
    D = compute_wigner_d_from_quaternion(q_combined, coeffs, U_blocks)

    # Step 7: Return D and its inverse (transpose for orthogonal matrices)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


def compute_wigner_d_from_quaternion_real(
    q: torch.Tensor,
    coeffs_sym: dict[str, torch.Tensor],
    U_blocks_real: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Compute real Wigner D matrices from quaternions using real arithmetic only.

    This is a torch.compile-compatible version that avoids all complex tensor
    operations by using real-pair arithmetic throughout.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        coeffs_sym: Precomputed symmetric Wigner coefficients
        U_blocks_real: Precomputed U blocks as (U_re, U_im) pairs

    Returns:
        Real block-diagonal Wigner D matrices of shape (N, size, size)
    """
    lmax = coeffs_sym["lmax"]
    ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
    D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs_sym)
    D_real = wigner_d_pair_to_real_blockwise(D_re, D_im, U_blocks_real, lmax)
    return D_real


def quaternion_wigner_real(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices using real arithmetic only (torch.compile compatible).

    This is the main entry point for torch.compile-compatible Wigner D computation.
    It uses real-pair arithmetic throughout to avoid graph breaks from complex
    tensor operations.

    The gamma rotation is combined with the edge→Y quaternion before computing
    the Wigner D, avoiding the overhead of computing two separate Wigner D
    matrices and multiplying them.

    Uses Euler-aligned U blocks (stored as real/imag pairs) that fold in the
    l=1 Cartesian permutation and l>=2 Euler basis transformation.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional rotation angles around the initial Y axis, shape (N,).
               If None, random angles in [0, 2π) are generated.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
        - wigner_edge_to_y: rotates edge → +Y (R @ edge = +Y for l=1)
        - wigner_y_to_edge: rotates +Y → edge (R @ +Y = edge for l=1)

        This matches the return convention of the Euler-based rotation.py.
    """
    # Import axis_angle helpers (they provide the correct quaternion computation)
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        quaternion_edge_to_y_stable,
        quaternion_multiply,
        quaternion_y_rotation,
    )

    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    dtype = edge_distance_vec.dtype
    device = edge_distance_vec.device

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge → +Y) using SLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge→Y
    # Combined rotation: first edge→Y, then rotate about Y by gamma
    # q_combined = q_gamma * q_edge_to_y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Get cached coefficients with real-pair U blocks
    coeffs, U_blocks_real = _get_cached_coefficients_real(lmax, dtype, device)

    # Step 6: Compute Wigner D from combined quaternion using real arithmetic
    D = compute_wigner_d_from_quaternion_real(q_combined, coeffs, U_blocks_real)

    # Step 7: Return D and its inverse (transpose for orthogonal matrices)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
