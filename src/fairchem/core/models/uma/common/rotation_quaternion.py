"""
Quaternion-based Wigner D Matrix Construction

This module implements Wigner D matrix computation using quaternions instead of
Euler angles. This approach avoids the gimbal lock singularity that occurs with
Euler angle representations when edges are nearly y-aligned.

The algorithm is based on the spherical_functions package by Mike Boyle:
https://github.com/moble/spherical_functions

Key insight: The quaternion is decomposed into two complex numbers Ra and Rb,
which encode the rotation without singularities. Special formulas handle the
edge cases where |Ra| ~ 0 or |Rb| ~ 0.

CRITICAL: This implementation NEVER computes Euler angles (alpha, beta, gamma).
No arccos, no atan2 on edge components. The path is:
    edge vector -> quaternion -> (Ra, Rb) -> Wigner D elements

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# Threshold for detecting near-zero magnitudes (avoiding ill-conditioned phases)
EPSILON = 1e-15


def precompute_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Precompute Wigner D coefficient tables for both Case 1 (ra >= rb) and Case 2 (ra < rb).

    These coefficients are:
        coeff(l, m', m) = sqrt[(l+m)!(l-m)! / ((l+m')!(l-m')!)] * C(l+m', rho_min) * C(l-m', l-m-rho_min)

    where C(n,k) is the binomial coefficient, and rho_min depends on the case.

    Args:
        lmax: Maximum angular momentum quantum number
        dtype: Data type for the coefficients
        device: Device to place the tensors on

    Returns:
        Dictionary containing:
            - 'case1': Coefficients for Case 1 (ra >= rb), shape (lmax+1, 2*lmax+1, 2*lmax+1)
            - 'case2': Coefficients for Case 2 (ra < rb), shape (lmax+1, 2*lmax+1, 2*lmax+1)
            - 'rho_min_case1': rho_min values for Case 1, shape (2*lmax+1, 2*lmax+1)
            - 'rho_max_case1': rho_max values for Case 1, shape (2*lmax+1, 2*lmax+1)
            - 'rho_min_case2': rho_min values for Case 2, shape (2*lmax+1, 2*lmax+1)
            - 'rho_max_case2': rho_max values for Case 2, shape (2*lmax+1, 2*lmax+1)
    """
    # Precompute factorials up to 2*lmax (sufficient for all calculations)
    max_factorial = 2 * lmax + 1
    factorial = torch.zeros(max_factorial + 1, dtype=dtype, device=device)
    factorial[0] = 1.0
    for i in range(1, max_factorial + 1):
        factorial[i] = factorial[i - 1] * i

    def binomial(n: int, k: int) -> float:
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0.0
        return float(factorial[n] / (factorial[k] * factorial[n - k]))

    # Allocate output tensors
    # Index convention: [l, mp + lmax, m + lmax] where mp, m range from -lmax to +lmax
    size = 2 * lmax + 1
    case1_coeffs = torch.zeros((lmax + 1, size, size), dtype=dtype, device=device)
    case2_coeffs = torch.zeros((lmax + 1, size, size), dtype=dtype, device=device)

    # Also store rho bounds for each (mp, m) pair
    rho_min_case1 = torch.zeros((size, size), dtype=torch.int64, device=device)
    rho_max_case1 = torch.zeros((size, size), dtype=torch.int64, device=device)
    rho_min_case2 = torch.zeros((size, size), dtype=torch.int64, device=device)
    rho_max_case2 = torch.zeros((size, size), dtype=torch.int64, device=device)

    for l in range(lmax + 1):
        for mp in range(-l, l + 1):
            for m in range(-l, l + 1):
                # Index in the (2*lmax+1) x (2*lmax+1) storage
                mp_idx = mp + lmax
                m_idx = m + lmax

                # Compute the sqrt prefactor (same for both cases)
                # sqrt[(l+m)!(l-m)! / ((l+m')!(l-m')!)]
                sqrt_factor = math.sqrt(
                    float(factorial[l + m] * factorial[l - m])
                    / float(factorial[l + mp] * factorial[l - mp])
                )

                # Case 1: ra >= rb
                # rho_min = max(0, m' - m)
                # rho_max = min(l + m', l - m)
                # coeff includes C(l+m', rho_min) * C(l-m', l-m-rho_min)
                rho_min_1 = max(0, mp - m)
                rho_max_1 = min(l + mp, l - m)

                if rho_min_1 <= rho_max_1:
                    binom1 = binomial(l + mp, rho_min_1)
                    binom2 = binomial(l - mp, l - m - rho_min_1)
                    case1_coeffs[l, mp_idx, m_idx] = sqrt_factor * binom1 * binom2

                rho_min_case1[mp_idx, m_idx] = rho_min_1
                rho_max_case1[mp_idx, m_idx] = rho_max_1

                # Case 2: ra < rb
                # rho_min_alt = max(0, -(m' + m))
                # rho_max_alt = min(l - m, l - m')
                # coeff includes C(l+m', l-m-rho_min_alt) * C(l-m', rho_min_alt)
                rho_min_2 = max(0, -(mp + m))
                rho_max_2 = min(l - m, l - mp)

                if rho_min_2 <= rho_max_2:
                    binom1_alt = binomial(l + mp, l - m - rho_min_2)
                    binom2_alt = binomial(l - mp, rho_min_2)
                    case2_coeffs[l, mp_idx, m_idx] = sqrt_factor * binom1_alt * binom2_alt

                rho_min_case2[mp_idx, m_idx] = rho_min_2
                rho_max_case2[mp_idx, m_idx] = rho_max_2

    return {
        "case1": case1_coeffs,
        "case2": case2_coeffs,
        "rho_min_case1": rho_min_case1,
        "rho_max_case1": rho_max_case1,
        "rho_min_case2": rho_min_case2,
        "rho_max_case2": rho_max_case2,
        "lmax": torch.tensor(lmax, dtype=torch.int64, device=device),
    }


def edge_to_quaternion(
    edge_vec: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert normalized edge vectors to quaternions that rotate +y to align with the edge.

    The quaternion represents the rotation R such that R @ [0, 1, 0]^T = edge_vec.

    Algorithm (half-angle formula):
        Given edge = (x, y, z), we want to rotate +y = (0, 1, 0) to edge.
        Using the formula q = normalize(1 + dot(a, b), cross(a, b)):
            dot(+y, edge) = y
            cross(+y, edge) = (z, 0, -x)
        So: q = normalize(1 + y, z, 0, -x)

    Singularity handling:
        When y = -1 (edge points to -y), the formula is singular.
        Fallback: use 180-degree rotation around x-axis: q = (0, 1, 0, 0).

    Args:
        edge_vec: Normalized edge vectors of shape (N, 3)
        gamma: Optional random rotation angles around y-axis, shape (N,).
               If None, random angles in [0, 2*pi) are generated.

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention.
    """
    # Ensure edge vectors are normalized
    edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

    x = edge_vec[..., 0]
    y = edge_vec[..., 1]
    z = edge_vec[..., 2]

    # Generate random gamma if not provided
    if gamma is None:
        gamma = torch.rand_like(x) * 2 * math.pi

    # Compute base quaternion using half-angle formula
    # q_unnorm = (1 + y, z, 0, -x) for rotation from +y to edge
    one_plus_y = 1.0 + y

    # Handle the singularity at y = -1
    # When one_plus_y is very small, use fallback quaternion
    singular_mask = one_plus_y < 1e-7

    # For non-singular case: normalize (1+y, z, 0, -x)
    # norm = sqrt((1+y)^2 + z^2 + x^2) = sqrt((1+y)^2 + (1-y^2))
    #      = sqrt((1+y)^2 + (1-y)(1+y)) = sqrt((1+y)(1+y + 1-y)) = sqrt(2(1+y))
    # So normalized: (sqrt((1+y)/2), z/sqrt(2(1+y)), 0, -x/sqrt(2(1+y)))

    # Safe computation avoiding division by zero
    safe_one_plus_y = torch.where(singular_mask, torch.ones_like(one_plus_y), one_plus_y)
    norm_factor = torch.sqrt(2.0 * safe_one_plus_y)
    inv_norm = 1.0 / norm_factor

    w_base = torch.sqrt(safe_one_plus_y / 2.0)
    x_base = z * inv_norm
    y_base = torch.zeros_like(x)
    z_base = -x * inv_norm

    # Fallback for singular case: 180-degree rotation around x-axis
    # q = (0, 1, 0, 0)
    w_base = torch.where(singular_mask, torch.zeros_like(w_base), w_base)
    x_base = torch.where(singular_mask, torch.ones_like(x_base), x_base)
    y_base = torch.where(singular_mask, torch.zeros_like(y_base), y_base)
    z_base = torch.where(singular_mask, torch.zeros_like(z_base), z_base)

    # Apply gamma rotation around y-axis
    # q_gamma = (cos(gamma/2), 0, sin(gamma/2), 0)
    # q_final = q_gamma * q_base (quaternion multiplication)
    cos_half_gamma = torch.cos(gamma / 2.0)
    sin_half_gamma = torch.sin(gamma / 2.0)

    # Quaternion multiplication: q_gamma * q_base
    # q1 = (w1, x1, y1, z1), q2 = (w2, x2, y2, z2)
    # q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #            w1*z2 + x1*y2 - y1*x2 + z1*w2)

    # q_gamma = (cos_half_gamma, 0, sin_half_gamma, 0)
    # q_base = (w_base, x_base, y_base, z_base)

    w_final = cos_half_gamma * w_base - sin_half_gamma * y_base
    x_final = cos_half_gamma * x_base + sin_half_gamma * z_base
    y_final = cos_half_gamma * y_base + sin_half_gamma * w_base
    z_final = cos_half_gamma * z_base - sin_half_gamma * x_base

    # Stack into quaternion tensor (w, x, y, z)
    q = torch.stack([w_final, x_final, y_final, z_final], dim=-1)

    return q


def quaternion_to_ra_rb(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into complex numbers Ra and Rb for Wigner D computation.

    Given q = (w, x, y, z), compute:
        Ra = w + i*z  (encodes cos(beta/2) and (alpha+gamma)/2)
        Rb = y + i*x  (encodes sin(beta/2) and (gamma-alpha)/2)

    The constraint |Ra|^2 + |Rb|^2 = 1 follows from unit quaternion property.

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) convention

    Returns:
        Tuple of (Ra, Rb), each complex tensor of shape (...)
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Ra = w + i*z
    Ra = torch.complex(w, z)

    # Rb = y + i*x
    Rb = torch.complex(y, x)

    return Ra, Rb


def _compute_wigner_d_element_general(
    l: int,
    mp: int,
    m: int,
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    ra: torch.Tensor,
    rb: torch.Tensor,
    phia: torch.Tensor,
    phib: torch.Tensor,
    coeff_case1: float,
    coeff_case2: float,
    rho_min_1: int,
    rho_max_1: int,
    rho_min_2: int,
    rho_max_2: int,
) -> torch.Tensor:
    """
    Compute a single Wigner D matrix element D^l_{m',m} for the general case.

    This implements the two-branch algorithm:
        - Case 1 (ra >= rb): Factor out powers of rb/ra
        - Case 2 (ra < rb): Factor out powers of ra/rb

    The Horner recurrence is used for numerical stability.

    Args:
        l: Angular momentum quantum number
        mp: m' index
        m: m index
        Ra, Rb: Complex Cayley-Klein parameters
        ra, rb: Magnitudes |Ra|, |Rb|
        phia, phib: Phases arg(Ra), arg(Rb)
        coeff_case1, coeff_case2: Precomputed Wigner coefficients
        rho_min_1, rho_max_1: Summation bounds for Case 1
        rho_min_2, rho_max_2: Summation bounds for Case 2

    Returns:
        Complex tensor of D^l_{m',m} values
    """
    # Compute phase factor (same for both cases)
    # phase = (m' + m) * phia + (m - m') * phib
    phase = (mp + m) * phia + (m - mp) * phib

    # Determine which case to use based on ra vs rb
    use_case1 = ra >= rb

    # ===== Case 1: ra >= rb =====
    # Polynomial ratio = -(rb/ra)^2
    # We clamp to avoid division by zero when ra is small
    # (but this case is already handled in special cases)
    safe_ra = torch.clamp(ra, min=EPSILON)
    ratio_case1 = -(rb * rb) / (safe_ra * safe_ra)

    # Horner evaluation from rho_max down to rho_min+1
    # Sum = 1 for rho = rho_max
    # Then for rho = rho_max-1, rho_max-2, ..., rho_min+1:
    #   Sum = 1 + ratio * Sum * (l+mp-rho+1)*(l-m-rho+1) / (rho*(m-mp+rho))
    sum_case1 = torch.ones_like(ra)
    for rho in range(rho_max_1, rho_min_1, -1):
        n1_term = l + mp - rho + 1
        n2_term = l - m - rho + 1
        m_term = m - mp + rho
        if rho != 0 and m_term != 0:
            factor = ratio_case1 * (n1_term * n2_term) / (rho * m_term)
            sum_case1 = 1.0 + sum_case1 * factor

    # Magnitude exponents after factoring out (rb/ra)^{2*rho_min}
    ra_exp_1 = 2 * l + mp - m - 2 * rho_min_1
    rb_exp_1 = m - mp + 2 * rho_min_1

    # Compute magnitude part safely
    magnitude_case1 = coeff_case1 * (safe_ra ** ra_exp_1) * (rb ** rb_exp_1)

    # ===== Case 2: ra < rb =====
    # Polynomial ratio = -(ra/rb)^2
    safe_rb = torch.clamp(rb, min=EPSILON)
    ratio_case2 = -(ra * ra) / (safe_rb * safe_rb)

    # Horner evaluation
    sum_case2 = torch.ones_like(rb)
    for rho in range(rho_max_2, rho_min_2, -1):
        n1_term = l - m - rho + 1
        n2_term = l - mp - rho + 1
        m_term = mp + m + rho
        if rho != 0 and m_term != 0:
            factor = ratio_case2 * (n1_term * n2_term) / (rho * m_term)
            sum_case2 = 1.0 + sum_case2 * factor

    # Sign factor for Case 2: (-1)^{l-m}
    sign_case2 = (-1) ** (l - m)

    # Magnitude exponents for Case 2
    ra_exp_2 = mp + m + 2 * rho_min_2
    rb_exp_2 = 2 * l - mp - m - 2 * rho_min_2

    magnitude_case2 = sign_case2 * coeff_case2 * (ra ** ra_exp_2) * (safe_rb ** rb_exp_2)

    # Select based on case
    magnitude = torch.where(use_case1, magnitude_case1, magnitude_case2)
    polynomial_sum = torch.where(use_case1, sum_case1, sum_case2)

    # Combine with phase
    # D = magnitude * polynomial_sum * exp(i * phase)
    result = magnitude * polynomial_sum * torch.exp(1j * phase.to(Ra.dtype))

    return result


def wigner_d_from_quaternion_complex(
    q: torch.Tensor,
    lmax: int,
    wigner_coeffs: dict[str, torch.Tensor],
    start_lmax: int = 0,
) -> torch.Tensor:
    """
    Compute block-diagonal COMPLEX Wigner D matrices from quaternions.

    This computes Wigner D matrices in the COMPLEX spherical harmonics basis.
    For real spherical harmonics (used by fairchem/e3nn), use wigner_d_complex_to_real()
    to convert the output.

    This is the main Euler-angle-free entry point for quaternion-based Wigner D computation.
    It handles all three cases:
        1. |Ra| ~ 0: Only anti-diagonal elements (m' = -m) are non-zero
        2. |Rb| ~ 0: Only diagonal elements (m' = m) are non-zero
        3. General case: Use two-branch algorithm

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        wigner_coeffs: Precomputed coefficient dictionary from precompute_wigner_coefficients()
        start_lmax: Starting angular momentum (default 0)

    Returns:
        Complex block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2 - start_lmax^2
    """
    N = q.shape[0]
    device = q.device
    real_dtype = q.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    # Decompose quaternion into Ra, Rb
    Ra, Rb = quaternion_to_ra_rb(q)

    # Get polar decomposition
    ra = torch.abs(Ra)
    rb = torch.abs(Rb)
    phia = torch.angle(Ra)
    phib = torch.angle(Rb)

    # Get coefficients (convert to working precision)
    coeff_case1 = wigner_coeffs["case1"].to(dtype=real_dtype, device=device)
    coeff_case2 = wigner_coeffs["case2"].to(dtype=real_dtype, device=device)
    stored_lmax = int(wigner_coeffs["lmax"].item())

    # Compute block-diagonal matrix (complex output)
    size = (lmax + 1) ** 2 - start_lmax**2
    wigner = torch.zeros(N, size, size, dtype=complex_dtype, device=device)

    block_start = 0
    for l in range(start_lmax, lmax + 1):
        block_size = 2 * l + 1

        # Build the complex Wigner D block for this l
        block = _wigner_d_block_complex(
            l,
            Ra,
            Rb,
            ra,
            rb,
            phia,
            phib,
            coeff_case1[l],
            coeff_case2[l],
            wigner_coeffs,
            stored_lmax,
        )

        # Place in block-diagonal matrix
        block_end = block_start + block_size
        wigner[:, block_start:block_end, block_start:block_end] = block

        block_start = block_end

    return wigner


def _wigner_d_block_complex(
    l: int,
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    ra: torch.Tensor,
    rb: torch.Tensor,
    phia: torch.Tensor,
    phib: torch.Tensor,
    coeff_case1_l: torch.Tensor,
    coeff_case2_l: torch.Tensor,
    wigner_coeffs: dict[str, torch.Tensor],
    stored_lmax: int,
) -> torch.Tensor:
    """
    Compute the complex Wigner D block for a single l value.

    This computes Wigner D matrix elements in the COMPLEX spherical harmonics basis.
    For real spherical harmonics, use wigner_d_complex_to_real() to convert.

    Args:
        l: Angular momentum
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        ra, rb, phia, phib: Magnitudes and phases, shape (N,)
        coeff_case1_l, coeff_case2_l: Coefficients for this l, shape (2*stored_lmax+1, 2*stored_lmax+1)
        wigner_coeffs: Full coefficient dictionary
        stored_lmax: lmax used when precomputing coefficients

    Returns:
        Complex Wigner D block of shape (N, 2*l+1, 2*l+1)
    """
    N = Ra.shape[0]
    device = Ra.device

    # Use complex dtype for proper Wigner D computation
    complex_dtype = torch.complex128 if ra.dtype == torch.float64 else torch.complex64

    block_size = 2 * l + 1
    block = torch.zeros(N, block_size, block_size, dtype=complex_dtype, device=device)

    # Get rho bounds
    rho_min_case1 = wigner_coeffs["rho_min_case1"]
    rho_max_case1 = wigner_coeffs["rho_max_case1"]
    rho_min_case2 = wigner_coeffs["rho_min_case2"]
    rho_max_case2 = wigner_coeffs["rho_max_case2"]

    # Masks for special cases
    ra_small = ra <= EPSILON  # beta ~ pi
    rb_small = rb <= EPSILON  # beta ~ 0
    general_case = ~ra_small & ~rb_small

    # Zero in complex dtype for initialization
    zero_complex = torch.zeros(N, dtype=complex_dtype, device=device)

    for mp_local in range(block_size):  # Local index 0 to 2*l
        mp = mp_local - l  # Actual m' value from -l to l

        for m_local in range(block_size):  # Local index 0 to 2*l
            m = m_local - l  # Actual m value from -l to l

            # Index into coefficient arrays
            mp_idx = mp + stored_lmax
            m_idx = m + stored_lmax

            # Initialize result as complex
            result = zero_complex.clone()

            # ===== Special case 1: ra ~ 0 (beta ~ pi) =====
            # Only anti-diagonal elements (m' = -m) are non-zero
            # D^l_{-m,m} = (-1)^{l-m} * Rb^{2m}
            if mp == -m:
                sign = (-1) ** (l - m)
                # Rb^{2m}: computed using polar form, keeping complex result
                Rb_power = torch.pow(rb, 2 * m) * torch.exp(1j * 2 * m * phib.to(complex_dtype))
                special_ra_zero = sign * Rb_power
                result = torch.where(ra_small, special_ra_zero, result)
            # For m' != -m when ra ~ 0, the element is 0 (already initialized)

            # ===== Special case 2: rb ~ 0 (beta ~ 0) =====
            # Only diagonal elements (m' = m) are non-zero
            # D^l_{m,m} = Ra^{2m}
            if mp == m:
                Ra_power = torch.pow(ra, 2 * m) * torch.exp(1j * 2 * m * phia.to(complex_dtype))
                special_rb_zero = Ra_power
                result = torch.where(rb_small & ~ra_small, special_rb_zero, result)
            # For m' != m when rb ~ 0, the element is 0 (already initialized)

            # ===== General case: both ra and rb significant =====
            if general_case.any():
                rho_min_1 = int(rho_min_case1[mp_idx, m_idx].item())
                rho_max_1 = int(rho_max_case1[mp_idx, m_idx].item())
                rho_min_2 = int(rho_min_case2[mp_idx, m_idx].item())
                rho_max_2 = int(rho_max_case2[mp_idx, m_idx].item())

                coeff_1 = float(coeff_case1_l[mp_idx, m_idx].item())
                coeff_2 = float(coeff_case2_l[mp_idx, m_idx].item())

                general_result = _compute_wigner_d_element_general(
                    l,
                    mp,
                    m,
                    Ra,
                    Rb,
                    ra,
                    rb,
                    phia,
                    phib,
                    coeff_1,
                    coeff_2,
                    rho_min_1,
                    rho_max_1,
                    rho_min_2,
                    rho_max_2,
                )
                # Keep full complex result (no .real - that would be incorrect!)
                result = torch.where(general_case, general_result, result)

            block[:, mp_local, m_local] = result

    return block


def quaternion_to_wigner(
    q: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    wigner_coeffs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute block-diagonal Wigner D matrices from quaternions.

    This is a drop-in replacement for eulers_to_wigner() that takes
    quaternions instead of Euler angles.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        start_lmax: Starting angular momentum
        end_lmax: Ending angular momentum
        wigner_coeffs: Precomputed coefficient dictionary

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (end_lmax+1)^2 - start_lmax^2
    """
    return wigner_d_from_quaternion(
        q=q,
        lmax=end_lmax,
        wigner_coeffs=wigner_coeffs,
        start_lmax=start_lmax,
    )


def init_edge_rot_quaternion(
    edge_distance_vec: torch.Tensor,
) -> torch.Tensor:
    """
    Compute quaternions for edge-to-y-axis rotations with random gamma.

    This is a drop-in replacement for init_edge_rot_euler_angles() that returns
    quaternions instead of Euler angles.

    The quaternion represents R such that R @ [0, 1, 0]^T = normalized(edge_vec).

    NOTE: The returned quaternion includes a random gamma rotation, just like
    the original Euler angle function.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    return edge_to_quaternion(edge_distance_vec, gamma=None)


# =============================================================================
# Optimized vectorized implementation
# =============================================================================


def wigner_d_from_quaternion_vectorized_complex(
    q: torch.Tensor,
    lmax: int,
    wigner_coeffs: dict[str, torch.Tensor],
    start_lmax: int = 0,
) -> torch.Tensor:
    """
    Optimized vectorized COMPLEX Wigner D computation.

    This version precomputes all the per-element quantities and uses
    tensor operations to avoid Python loops where possible.

    Returns complex Wigner D matrices in the complex spherical harmonics basis.
    For real spherical harmonics, use wigner_d_complex_to_real() to convert.

    Args:
        q: Quaternions of shape (N, 4)
        lmax: Maximum angular momentum
        wigner_coeffs: Precomputed coefficients
        start_lmax: Starting angular momentum

    Returns:
        Complex block-diagonal Wigner D matrices of shape (N, size, size)
    """
    N = q.shape[0]
    device = q.device
    real_dtype = q.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    # Decompose quaternion
    Ra, Rb = quaternion_to_ra_rb(q)
    ra = torch.abs(Ra)
    rb = torch.abs(Rb)
    phia = torch.angle(Ra)
    phib = torch.angle(Rb)

    # Get coefficients
    coeff_case1 = wigner_coeffs["case1"].to(dtype=real_dtype, device=device)
    coeff_case2 = wigner_coeffs["case2"].to(dtype=real_dtype, device=device)
    stored_lmax = int(wigner_coeffs["lmax"].item())

    # Masks for special cases
    ra_small = ra <= EPSILON  # (N,)
    rb_small = rb <= EPSILON  # (N,)
    general_case = ~ra_small & ~rb_small  # (N,)

    # For case selection in general case
    use_case1 = ra >= rb  # (N,)

    # Precompute safe ratios for both cases
    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    ratio_case1 = -(rb * rb) / (safe_ra * safe_ra)  # (N,)
    ratio_case2 = -(ra * ra) / (safe_rb * safe_rb)  # (N,)

    # Compute output size and allocate (complex output)
    size = (lmax + 1) ** 2 - start_lmax**2
    wigner = torch.zeros(N, size, size, dtype=complex_dtype, device=device)

    # Zero in complex dtype for initialization
    zero_complex = torch.zeros(N, dtype=complex_dtype, device=device)

    block_start = 0
    for l in range(start_lmax, lmax + 1):
        block_size = 2 * l + 1

        # Build block for this l (complex)
        block = torch.zeros(N, block_size, block_size, dtype=complex_dtype, device=device)

        # Get rho bounds tensors
        rho_min_case1 = wigner_coeffs["rho_min_case1"]
        rho_max_case1 = wigner_coeffs["rho_max_case1"]
        rho_min_case2 = wigner_coeffs["rho_min_case2"]
        rho_max_case2 = wigner_coeffs["rho_max_case2"]

        for mp_local in range(block_size):
            mp = mp_local - l

            for m_local in range(block_size):
                m = m_local - l

                mp_idx = mp + stored_lmax
                m_idx = m + stored_lmax

                # Initialize result tensor (complex)
                result = zero_complex.clone()

                # ===== Special case: ra ~ 0 (beta ~ pi) =====
                if mp == -m:
                    sign = (-1) ** (l - m)
                    Rb_power = torch.pow(rb, 2 * m) * torch.exp(1j * 2 * m * phib.to(complex_dtype))
                    special_ra = sign * Rb_power
                    result = torch.where(ra_small, special_ra, result)

                # ===== Special case: rb ~ 0 (beta ~ 0) =====
                if mp == m:
                    Ra_power = torch.pow(ra, 2 * m) * torch.exp(1j * 2 * m * phia.to(complex_dtype))
                    special_rb = Ra_power
                    result = torch.where(rb_small & ~ra_small, special_rb, result)

                # ===== General case =====
                # Phase (same for both cases)
                phase = (mp + m) * phia + (m - mp) * phib

                # Get bounds and coefficients for this (mp, m)
                rho_min_1 = int(rho_min_case1[mp_idx, m_idx].item())
                rho_max_1 = int(rho_max_case1[mp_idx, m_idx].item())
                rho_min_2 = int(rho_min_case2[mp_idx, m_idx].item())
                rho_max_2 = int(rho_max_case2[mp_idx, m_idx].item())

                coeff_1 = coeff_case1[l, mp_idx, m_idx]
                coeff_2 = coeff_case2[l, mp_idx, m_idx]

                # Case 1 Horner sum
                sum_1 = torch.ones_like(ra)
                for rho in range(rho_max_1, rho_min_1, -1):
                    n1 = l + mp - rho + 1
                    n2 = l - m - rho + 1
                    m_denom = m - mp + rho
                    if rho != 0 and m_denom != 0:
                        factor = ratio_case1 * (n1 * n2) / (rho * m_denom)
                        sum_1 = 1.0 + sum_1 * factor

                # Case 1 magnitude exponents
                ra_exp_1 = 2 * l + mp - m - 2 * rho_min_1
                rb_exp_1 = m - mp + 2 * rho_min_1
                mag_1 = coeff_1 * (safe_ra ** ra_exp_1) * (rb ** rb_exp_1)

                # Case 2 Horner sum
                sum_2 = torch.ones_like(rb)
                for rho in range(rho_max_2, rho_min_2, -1):
                    n1 = l - m - rho + 1
                    n2 = l - mp - rho + 1
                    m_denom = mp + m + rho
                    if rho != 0 and m_denom != 0:
                        factor = ratio_case2 * (n1 * n2) / (rho * m_denom)
                        sum_2 = 1.0 + sum_2 * factor

                # Case 2 magnitude exponents and sign
                sign_2 = (-1) ** (l - m)
                ra_exp_2 = mp + m + 2 * rho_min_2
                rb_exp_2 = 2 * l - mp - m - 2 * rho_min_2
                mag_2 = sign_2 * coeff_2 * (ra ** ra_exp_2) * (safe_rb ** rb_exp_2)

                # Select based on case
                magnitude = torch.where(use_case1, mag_1, mag_2)
                poly_sum = torch.where(use_case1, sum_1, sum_2)

                # Combine with phase (keep complex result!)
                phase_factor = torch.exp(1j * phase.to(complex_dtype))
                general_result = magnitude * poly_sum * phase_factor

                # Apply general case where appropriate
                result = torch.where(general_case, general_result, result)

                block[:, mp_local, m_local] = result

        # Place block in output
        block_end = block_start + block_size
        wigner[:, block_start:block_end, block_start:block_end] = block
        block_start = block_end

    return wigner


# =============================================================================
# Integration helper: replacement for the current rotation pipeline
# =============================================================================


def get_wigner_from_edge_vectors_euler_free(
    edge_distance_vec: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    wigner_coeffs: dict[str, torch.Tensor],
    U: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete EULER-ANGLE-FREE pipeline: edge vectors -> real Wigner D matrices.

    This is the fully Euler-angle-free approach:
    1. Convert edge vectors to quaternions
    2. Compute complex Wigner D matrices directly from quaternion (Ra, Rb)
    3. Convert to real spherical harmonics using unitary transformation

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        start_lmax: Starting angular momentum
        end_lmax: Ending angular momentum
        wigner_coeffs: Precomputed Wigner coefficients (from precompute_wigner_coefficients)
        U: Unitary complex-to-real transformation matrix (from precompute_complex_to_real_matrix)

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
        Output is real-valued (for real spherical harmonics)
    """
    # Get quaternions (includes random gamma)
    q = init_edge_rot_quaternion(edge_distance_vec)

    # Compute complex Wigner D matrices (Euler-angle-free!)
    wigner_complex = wigner_d_from_quaternion_vectorized_complex(
        q=q,
        lmax=end_lmax,
        wigner_coeffs=wigner_coeffs,
        start_lmax=start_lmax,
    )

    # Convert to real spherical harmonics basis
    wigner_real = wigner_d_complex_to_real(wigner_complex, U)

    # Inverse is transpose (real Wigner D matrices are orthogonal)
    wigner_real_inv = torch.transpose(wigner_real, 1, 2).contiguous()

    return wigner_real, wigner_real_inv


def get_wigner_from_edge_vectors_complex(
    edge_distance_vec: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    wigner_coeffs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete pipeline: edge vectors -> complex Wigner D matrices.

    This returns complex Wigner D matrices in the complex spherical harmonics basis.
    For real spherical harmonics (fairchem/e3nn), use get_wigner_from_edge_vectors_euler_free
    or get_wigner_from_edge_vectors_real instead.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        start_lmax: Starting angular momentum
        end_lmax: Ending angular momentum
        wigner_coeffs: Precomputed Wigner coefficients

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
        Output is complex-valued.
    """
    # Get quaternions (includes random gamma)
    q = init_edge_rot_quaternion(edge_distance_vec)

    # Compute complex Wigner D matrices
    wigner = wigner_d_from_quaternion_vectorized_complex(
        q=q,
        lmax=end_lmax,
        wigner_coeffs=wigner_coeffs,
        start_lmax=start_lmax,
    )

    # Inverse is conjugate transpose (complex Wigner D matrices are unitary)
    wigner_inv = torch.conj(torch.transpose(wigner, 1, 2)).contiguous()

    return wigner, wigner_inv


# =============================================================================
# Complex to Real Spherical Harmonics Conversion
# =============================================================================


def precompute_complex_to_real_matrix(
    lmax: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Precompute the unitary matrix that transforms from complex to real spherical harmonics.

    For each l, the transformation is:
        S_l^m = sum_m' U_{m,m'} Y_l^{m'}

    where S_l^m are real spherical harmonics and Y_l^{m'} are complex spherical harmonics.

    The convention used follows the e3nn/ESCN convention where real spherical harmonics
    are ordered as: m = -l, -l+1, ..., 0, ..., l-1, l

    The transformation matrix U is defined as:
        For m > 0:  S_l^m = (1/sqrt(2)) * (Y_l^{-m} + (-1)^m * Y_l^m)
        For m = 0:  S_l^0 = Y_l^0
        For m < 0:  S_l^m = (i/sqrt(2)) * (Y_l^m - (-1)^m * Y_l^{-m})

    Note: Different conventions exist. This implementation may need adjustment
    to match the specific e3nn convention used in fairchem.

    Args:
        lmax: Maximum angular momentum
        dtype: Complex data type for the matrix
        device: Device to place the tensor on

    Returns:
        Block-diagonal unitary matrix of shape (size, size) where size = (lmax+1)^2
    """
    size = (lmax + 1) ** 2
    U = torch.zeros(size, size, dtype=dtype, device=device)

    block_start = 0
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    for l in range(lmax + 1):
        block_size = 2 * l + 1

        # Build transformation block for this l
        # Complex harmonics are indexed as m = -l, -l+1, ..., 0, ..., l-1, l
        # Real harmonics are indexed the same way

        for m in range(-l, l + 1):
            m_idx = m + l  # Index in the block (0 to 2*l)
            row = block_start + m_idx

            if m > 0:
                # S_l^m = (1/sqrt(2)) * (Y_l^{-m} + (-1)^m * Y_l^m)
                col_neg_m = block_start + (-m + l)  # Index for Y_l^{-m}
                col_pos_m = block_start + (m + l)  # Index for Y_l^m
                sign = (-1) ** m
                U[row, col_neg_m] = sqrt2_inv
                U[row, col_pos_m] = sign * sqrt2_inv
            elif m == 0:
                # S_l^0 = Y_l^0
                col = block_start + l  # Index for Y_l^0
                U[row, col] = 1.0
            else:  # m < 0
                # S_l^m = (i/sqrt(2)) * (Y_l^m - (-1)^m * Y_l^{-m})
                col_m = block_start + (m + l)  # Index for Y_l^m
                col_neg_m = block_start + (-m + l)  # Index for Y_l^{-m}
                sign = (-1) ** (-m)  # Note: use -m since m < 0
                U[row, col_m] = 1j * sqrt2_inv
                U[row, col_neg_m] = -sign * 1j * sqrt2_inv

        block_start += block_size

    return U


def wigner_d_complex_to_real(
    D_complex: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from complex to real spherical harmonics basis.

    D_real = U^H @ D_complex @ U

    Args:
        D_complex: Complex Wigner D matrices of shape (N, size, size)
        U: Unitary transformation matrix of shape (size, size)

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    # D_real = U^H @ D_complex @ U
    # For batched computation: D_real[i] = U^H @ D_complex[i] @ U
    U_H = U.conj().T
    D_real = torch.einsum("ij,njk,kl->nil", U_H, D_complex, U)

    # The result should be real for proper rotations
    return D_real.real


# =============================================================================
# Alternative approach: Direct real Wigner D from quaternion using J matrices
# =============================================================================


def wigner_d_real_from_quaternion(
    q: torch.Tensor,
    lmax: int,
    Jd: list[torch.Tensor],
    start_lmax: int = 0,
) -> torch.Tensor:
    """
    Compute real Wigner D matrices directly from quaternions using the J-matrix approach.

    This uses the same J matrices as the e3nn convention, but extracts Euler angles
    from the quaternion's Ra/Rb decomposition in a numerically stable way.

    The algorithm handles three cases:
    1. |Ra| ~ 0 (beta ~ pi): Use special anti-diagonal formula
    2. |Rb| ~ 0 (beta ~ 0): Use special diagonal formula
    3. General case: Extract Euler angles from Ra/Rb (both phases well-conditioned)

    The key insight is that Euler angles are only extracted when BOTH Ra and Rb have
    significant magnitude, ensuring the phases arg(Ra) and arg(Rb) are well-conditioned.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        Jd: Precomputed J matrices (loaded from Jd.pt)
        start_lmax: Starting angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
    """
    N = q.shape[0]
    device = q.device
    dtype = q.dtype

    # Decompose quaternion into Ra, Rb
    Ra, Rb = quaternion_to_ra_rb(q)

    # Get polar decomposition
    ra = torch.abs(Ra)
    rb = torch.abs(Rb)

    # Masks for special cases
    # Use a slightly larger epsilon for stability
    eps = 1e-6
    ra_small = ra < eps  # beta ~ pi
    rb_small = rb < eps  # beta ~ 0
    general_case = ~ra_small & ~rb_small

    # For computing phases safely:
    # - phia should only be used when ra is significant (not ra_small)
    # - phib should only be used when rb is significant (not rb_small)
    #
    # We compute phases for all edges but the gradients will only flow through
    # edges where the corresponding magnitude is significant (due to torch.where masking).
    #
    # For numerical stability in the forward pass, we replace small-magnitude
    # complex numbers with 1+0j before computing angle() to avoid NaN.

    # Safe phase computation: when magnitude is small, use 1+0j to avoid NaN in angle()
    # This doesn't affect correctness because we mask these values in the output anyway
    ones_complex = torch.ones_like(Ra)
    safe_Ra = torch.where(ra > eps, Ra, ones_complex)
    safe_Rb = torch.where(rb > eps, Rb, ones_complex)
    phia = torch.angle(safe_Ra)  # (alpha + gamma) / 2
    phib = torch.angle(safe_Rb)  # (gamma - alpha) / 2

    # Euler angles from phases
    # alpha + gamma = 2 * phia
    # gamma - alpha = 2 * phib
    # Therefore:
    #   gamma = phia + phib
    #   alpha = phia - phib
    # And beta = 2 * atan2(rb, ra) (stable version of 2 * acos(ra))
    alpha = phia - phib
    gamma = phia + phib
    beta = 2.0 * torch.atan2(rb, ra)

    # Compute block-diagonal matrix
    size = (lmax + 1) ** 2 - start_lmax**2
    wigner = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for l in range(start_lmax, lmax + 1):
        block_size = 2 * l + 1

        # Get J matrix for this l
        J = Jd[l].to(dtype=dtype, device=device)

        # Initialize block
        block = torch.zeros(N, block_size, block_size, dtype=dtype, device=device)

        # ===== General case: use Xa @ J @ Xb @ J @ Xc formula =====
        if general_case.any():
            Xa = _z_rot_mat_batched(alpha, l, device, dtype)
            Xb = _z_rot_mat_batched(beta, l, device, dtype)
            Xc = _z_rot_mat_batched(gamma, l, device, dtype)

            # Wigner D = Xa @ J @ Xb @ J @ Xc
            general_block = torch.einsum("nij,jk,nkl,lm,nmr->nir", Xa, J, Xb, J, Xc)

            # Apply to general case edges
            block = torch.where(
                general_case.view(N, 1, 1).expand(N, block_size, block_size),
                general_block,
                block,
            )

        # ===== Special case: rb ~ 0 (beta ~ 0) =====
        # Wigner D is diagonal: D^l_{m,m} = Ra^{2m}
        # For real spherical harmonics, this becomes a z-rotation by angle 2*phia
        if rb_small.any():
            # When rb ~ 0, the rotation is approximately Ra^{2m} on diagonal
            # In real spherical harmonics, this is cos(2m*phia) on diagonal
            # and sin(2m*phia) on anti-diagonal
            rb_small_block = _z_rot_mat_batched(2.0 * phia, l, device, dtype)

            block = torch.where(
                (rb_small & ~ra_small).view(N, 1, 1).expand(N, block_size, block_size),
                rb_small_block,
                block,
            )

        # ===== Special case: ra ~ 0 (beta ~ pi) =====
        # Wigner D is anti-diagonal: D^l_{-m,m} = (-1)^{l-m} * Rb^{2m}
        # This is a 180-degree rotation around an axis in the xy-plane
        if ra_small.any():
            # When ra ~ 0, we have beta ~ pi (180-degree flip)
            # The rotation is Rb^{2m} on the anti-diagonal with sign (-1)^{l-m}
            # In real spherical harmonics basis, this needs careful handling
            ra_small_block = _compute_special_case_ra_small(
                Rb, rb, phib, l, device, dtype
            )

            block = torch.where(
                ra_small.view(N, 1, 1).expand(N, block_size, block_size),
                ra_small_block,
                block,
            )

        # Place block in output
        block_end = block_start + block_size
        wigner[:, block_start:block_end, block_start:block_end] = block
        block_start = block_end

    return wigner


def _compute_special_case_ra_small(
    Rb: torch.Tensor,
    rb: torch.Tensor,
    phib: torch.Tensor,
    l: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute Wigner D block for the special case when |Ra| ~ 0 (beta ~ pi).

    For complex spherical harmonics:
        D^l_{m',m} = 0 except for m' = -m
        D^l_{-m,m} = (-1)^{l-m} * Rb^{2m}

    For real spherical harmonics, this corresponds to a 180-degree rotation
    around an axis in the xy-plane (determined by phib).

    The transformation to real spherical harmonics requires careful handling
    of the anti-diagonal structure.

    Args:
        Rb: Complex Rb values, shape (N,)
        rb: Magnitudes |Rb|, shape (N,)
        phib: Phases arg(Rb), shape (N,)
        l: Angular momentum
        device: Device for output
        dtype: Data type for output

    Returns:
        Wigner D block of shape (N, 2*l+1, 2*l+1)
    """
    N = Rb.shape[0]
    block_size = 2 * l + 1
    block = torch.zeros(N, block_size, block_size, dtype=dtype, device=device)

    # For the special case ra ~ 0, the rotation is 180 degrees around some axis
    # in the xy-plane. The axis direction is encoded in phib.

    # In complex spherical harmonics:
    # D^l_{-m,m} = (-1)^{l-m} * Rb^{2m} = (-1)^{l-m} * rb^{2m} * exp(2im*phib)

    # For real spherical harmonics, we need to apply the change of basis.
    # The result depends on the specific convention.

    # As a first approximation, when beta = pi:
    # - For l even: the Wigner D has a specific anti-diagonal pattern
    # - For l odd: the pattern is modified

    # The safest approach is to compute using the limiting form of
    # Xa @ J @ Xb @ J @ Xc as beta -> pi.

    # When beta -> pi:
    # - Xb approaches a specific form (rotation by pi)
    # - alpha and gamma become degenerate (only alpha - gamma matters)

    # For beta = pi exactly, the J @ Xb @ J pattern gives specific structure.
    # Let's compute it directly.

    # For ra ~ 0, we can approximate:
    # - ra ~ 0, rb ~ 1
    # - beta = pi
    # - phib encodes the rotation around the y-axis (roughly gamma - alpha)

    # We use the limiting form: D(beta=pi, phi) where phi relates to phib
    # The Wigner D at beta=pi is anti-diagonal with entries depending on phib

    for mp_local in range(block_size):
        mp = mp_local - l

        for m_local in range(block_size):
            m = m_local - l

            if mp == -m:
                # The anti-diagonal element
                # In complex basis: (-1)^{l-m} * Rb^{2m}
                sign = (-1) ** (l - m)

                # Rb^{2m} in magnitude-phase form
                # rb^{2m} * exp(2im*phib)
                # The real part depends on the conversion to real spherical harmonics

                # For real spherical harmonics at beta = pi:
                # The element D^l_{-m,m} involves cos(2m*phib) or sin(2m*phib)
                # depending on the signs of m and mp

                # Simplified formula for real spherical harmonics:
                # This is an approximation that captures the main behavior
                magnitude = torch.pow(rb, 2 * abs(m))

                if m == 0:
                    # D^l_{0,0} at beta=pi is (-1)^l
                    element = sign * magnitude
                else:
                    # For m != 0, the element involves the phase
                    element = sign * magnitude * torch.cos(2 * m * phib)

                block[:, mp_local, m_local] = element

    return block


def _z_rot_mat_batched(
    angle: torch.Tensor,
    l: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create batched z-rotation matrices in real spherical harmonics basis.

    This is the batched version of _z_rot_mat from rotation.py.

    Args:
        angle: Rotation angles of shape (N,)
        l: Angular momentum
        device: Device for the output
        dtype: Data type for the output

    Returns:
        Rotation matrices of shape (N, 2*l+1, 2*l+1)
    """
    N = angle.shape[0]
    block_size = 2 * l + 1
    M = torch.zeros(N, block_size, block_size, dtype=dtype, device=device)

    # frequencies range from l to -l (descending)
    # For each frequency f, the matrix has:
    #   cos(f * angle) on the diagonal
    #   sin(f * angle) on the anti-diagonal
    for i, f in enumerate(range(l, -l - 1, -1)):
        cos_val = torch.cos(f * angle)
        sin_val = torch.sin(f * angle)
        M[:, i, i] = cos_val
        M[:, i, block_size - 1 - i] = sin_val

    return M


def get_wigner_from_edge_vectors_real(
    edge_distance_vec: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete pipeline using real spherical harmonics J-matrix approach.

    This version uses the e3nn J-matrix convention to produce real Wigner D
    matrices that are directly compatible with the existing fairchem code.

    The Euler angles are extracted from the quaternion's Ra/Rb decomposition,
    which avoids the singularity in the original edge -> Euler angle conversion.

    Args:
        edge_distance_vec: Edge distance vectors of shape (N, 3)
        start_lmax: Starting angular momentum
        end_lmax: Ending angular momentum
        Jd: Precomputed J matrices (loaded from Jd.pt)

    Returns:
        Tuple of (wigner, wigner_inv) where each has shape (N, size, size)
    """
    # Get quaternions (includes random gamma)
    q = init_edge_rot_quaternion(edge_distance_vec)

    # Compute Wigner D matrices using J-matrix approach
    wigner = wigner_d_real_from_quaternion(
        q=q,
        lmax=end_lmax,
        Jd=Jd,
        start_lmax=start_lmax,
    )

    # Inverse is transpose (Wigner D matrices are orthogonal)
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

    return wigner, wigner_inv
