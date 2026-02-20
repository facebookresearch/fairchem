"""
Shared utilities for Wigner D matrix computation.

This module provides the foundational functions used by all three Wigner D
computation methods (matrix exponential, hybrid, and polynomial):
- Quaternion operations
- SO(3) Lie algebra generators with Euler-matching basis transformation
- Ra/Rb decomposition and polynomial computation
- Complex-to-real transformation
- Caching utilities

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# =============================================================================
# Data Structures for Wigner Coefficients
# =============================================================================


class CaseCoeffsModule(nn.Module):
    """Polynomial coefficients for one case (|Ra|>=|Rb| or |Ra|<|Rb|) as nn.Module."""

    def __init__(
        self,
        coeff: torch.Tensor,
        horner: torch.Tensor,
        poly_len: torch.Tensor,
        ra_exp: torch.Tensor,
        rb_exp: torch.Tensor,
        sign: torch.Tensor,
    ):
        super().__init__()
        # Use persistent=False since these are computed, not learned
        self.register_buffer("coeff", coeff, persistent=False)
        self.register_buffer("horner", horner, persistent=False)
        self.register_buffer("poly_len", poly_len, persistent=False)
        self.register_buffer("ra_exp", ra_exp, persistent=False)
        self.register_buffer("rb_exp", rb_exp, persistent=False)
        self.register_buffer("sign", sign, persistent=False)


class WignerCoefficientsModule(nn.Module):
    """Precomputed coefficients for Wigner D matrix computation as nn.Module."""

    def __init__(
        self,
        lmin: int,
        lmax: int,
        size: int,
        max_poly_len: int,
        n_primary: int,
        n_derived: int,
        primary_row: torch.Tensor,
        primary_col: torch.Tensor,
        case1: CaseCoeffsModule,
        case2: CaseCoeffsModule,
        mp_plus_m: torch.Tensor,
        m_minus_mp: torch.Tensor,
        diagonal_mask: torch.Tensor,
        anti_diagonal_mask: torch.Tensor,
        special_2m: torch.Tensor,
        anti_diag_sign: torch.Tensor,
        derived_row: torch.Tensor,
        derived_col: torch.Tensor,
        derived_primary_idx: torch.Tensor,
        derived_sign: torch.Tensor,
    ):
        super().__init__()
        # Metadata (regular attributes, not tensors)
        self.lmin = lmin
        self.lmax = lmax
        self.size = size
        self.max_poly_len = max_poly_len
        self.n_primary = n_primary
        self.n_derived = n_derived

        # Primary element indices (persistent=False since these are computed)
        self.register_buffer("primary_row", primary_row, persistent=False)
        self.register_buffer("primary_col", primary_col, persistent=False)

        # Case coefficients (submodules)
        self.case1 = case1
        self.case2 = case2

        # Phase computation
        self.register_buffer("mp_plus_m", mp_plus_m, persistent=False)
        self.register_buffer("m_minus_mp", m_minus_mp, persistent=False)

        # Special cases (Ra~0 or Rb~0)
        self.register_buffer("diagonal_mask", diagonal_mask, persistent=False)
        self.register_buffer("anti_diagonal_mask", anti_diagonal_mask, persistent=False)
        self.register_buffer("special_2m", special_2m, persistent=False)
        self.register_buffer("anti_diag_sign", anti_diag_sign, persistent=False)

        # Derived element mapping
        self.register_buffer("derived_row", derived_row, persistent=False)
        self.register_buffer("derived_col", derived_col, persistent=False)
        self.register_buffer(
            "derived_primary_idx", derived_primary_idx, persistent=False
        )
        self.register_buffer("derived_sign", derived_sign, persistent=False)


class WignerDataModule(nn.Module):
    """
    Combined Wigner coefficients and U transformation blocks as nn.Module.

    This module holds all precomputed data needed for Wigner D computation,
    and automatically moves with the parent model via .to(device).

    U_blocks are stored as real/imaginary pairs for torch.compile compatibility.
    """

    def __init__(
        self,
        coeffs: WignerCoefficientsModule,
        U_blocks: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.coeffs = coeffs

        # Register U_blocks as non-persistent buffers (computed, not learned)
        # Each U_block is a (U_re, U_im) tuple
        self._n_U_blocks = len(U_blocks)
        for i, (U_re, U_im) in enumerate(U_blocks):
            self.register_buffer(f"U_block_{i}_re", U_re, persistent=False)
            self.register_buffer(f"U_block_{i}_im", U_im, persistent=False)

    @property
    def U_blocks(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return U_blocks as a list of (U_re, U_im) tuples for compatibility with existing code."""
        return [
            (getattr(self, f"U_block_{i}_re"), getattr(self, f"U_block_{i}_im"))
            for i in range(self._n_U_blocks)
        ]


@dataclass
class CaseCoeffs:
    """Polynomial coefficients for one case (|Ra|>=|Rb| or |Ra|<|Rb|)."""

    coeff: torch.Tensor  # Leading coefficient
    horner: torch.Tensor  # Horner polynomial factors
    poly_len: torch.Tensor  # Polynomial length per element
    ra_exp: torch.Tensor  # Ra exponent
    rb_exp: torch.Tensor  # Rb exponent
    sign: torch.Tensor  # Sign factor


@dataclass
class WignerCoefficients:
    """Precomputed coefficients for Wigner D matrix computation."""

    # Metadata
    lmin: int
    lmax: int
    size: int
    max_poly_len: int

    # Primary element indices
    primary_row: torch.Tensor
    primary_col: torch.Tensor
    n_primary: int

    # Case coefficients
    case1: CaseCoeffs
    case2: CaseCoeffs

    # Phase computation
    mp_plus_m: torch.Tensor
    m_minus_mp: torch.Tensor

    # Special cases (Ra~0 or Rb~0)
    diagonal_mask: torch.Tensor
    anti_diagonal_mask: torch.Tensor
    special_2m: torch.Tensor
    anti_diag_sign: torch.Tensor

    # Derived element mapping
    n_derived: int
    derived_row: torch.Tensor
    derived_col: torch.Tensor
    derived_primary_idx: torch.Tensor
    derived_sign: torch.Tensor


# =============================================================================
# Constants
# =============================================================================

# Blend region parameters for two-chart quaternion computation
# The blend region is ey in [_BLEND_START, _BLEND_START + _BLEND_WIDTH]
# which corresponds to ey in [-0.9, 0.9]
_BLEND_START = -0.9
_BLEND_WIDTH = 1.8

# Default cache directory for precomputed coefficients
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fairchem" / "wigner_coeffs"


# =============================================================================
# Global Caches (consolidated from all modules)
# =============================================================================

# Coefficient cache for Ra/Rb polynomial (real-pair version)
_COEFF_REAL_CACHE: dict[tuple[int, int, torch.dtype, torch.device], tuple] = {}


def clear_memory_caches() -> None:
    """
    Clear all in-memory caches for Wigner D computation.

    This clears the module-level dictionaries that cache coefficients and U blocks.
    Useful for testing or reducing memory.

    Also clears caches in the individual method modules if they are loaded.
    """
    _COEFF_REAL_CACHE.clear()


# =============================================================================
# Coefficient Caching Functions
# =============================================================================


def get_ra_rb_coefficients_real(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
    lmin: int = 0,
) -> tuple[WignerCoefficients, list]:
    """Get cached Ra/Rb polynomial coefficients with real-pair U blocks.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        lmin: Minimum angular momentum (default 0)

    Returns:
        Tuple of (WignerCoefficients, U_blocks_real list)
    """
    key = (lmin, lmax, dtype, device)

    if key not in _COEFF_REAL_CACHE:
        coeffs = precompute_wigner_coefficients(
            lmax, dtype=dtype, device=device, lmin=lmin
        )
        # Get full real U blocks and slice to range
        full_U_blocks_real = precompute_U_blocks_euler_aligned_real(
            lmax, dtype=dtype, device=device
        )
        U_blocks_range_real = full_U_blocks_real[lmin:]
        _COEFF_REAL_CACHE[key] = (coeffs, U_blocks_range_real)

    return _COEFF_REAL_CACHE[key]


def create_wigner_data_module(
    lmax: int,
    lmin: int = 4,
) -> WignerDataModule:
    """
    Create a WignerDataModule with precomputed coefficients and U blocks.

    This creates an nn.Module that holds all precomputed Wigner D data,
    which can be registered as a submodule and will automatically move
    with the parent model via .to(device).

    The module is created on CPU with float64 precision. When the parent
    model is moved to a device or dtype, the buffers will be converted
    automatically.

    Args:
        lmax: Maximum angular momentum
        lmin: Minimum angular momentum (default 4, matching hybrid method
              which uses custom kernels for l=0,1,2,3)

    Returns:
        WignerDataModule containing coefficients and U_blocks
    """
    # Create on CPU with float64 (will be converted when model moves)
    dtype = torch.float64
    device = torch.device("cpu")

    # Use existing precompute functions
    coeffs_dataclass = precompute_wigner_coefficients(
        lmax, dtype=dtype, device=device, lmin=lmin
    )
    # Get full real U blocks and slice to range
    full_U_blocks_real = precompute_U_blocks_euler_aligned_real(
        lmax, dtype=dtype, device=device
    )
    U_blocks = full_U_blocks_real[lmin:]

    # Convert CaseCoeffs dataclass to CaseCoeffsModule
    case1_module = CaseCoeffsModule(
        coeff=coeffs_dataclass.case1.coeff,
        horner=coeffs_dataclass.case1.horner,
        poly_len=coeffs_dataclass.case1.poly_len,
        ra_exp=coeffs_dataclass.case1.ra_exp,
        rb_exp=coeffs_dataclass.case1.rb_exp,
        sign=coeffs_dataclass.case1.sign,
    )
    case2_module = CaseCoeffsModule(
        coeff=coeffs_dataclass.case2.coeff,
        horner=coeffs_dataclass.case2.horner,
        poly_len=coeffs_dataclass.case2.poly_len,
        ra_exp=coeffs_dataclass.case2.ra_exp,
        rb_exp=coeffs_dataclass.case2.rb_exp,
        sign=coeffs_dataclass.case2.sign,
    )

    # Create WignerCoefficientsModule
    coeffs_module = WignerCoefficientsModule(
        lmin=coeffs_dataclass.lmin,
        lmax=coeffs_dataclass.lmax,
        size=coeffs_dataclass.size,
        max_poly_len=coeffs_dataclass.max_poly_len,
        n_primary=coeffs_dataclass.n_primary,
        n_derived=coeffs_dataclass.n_derived,
        primary_row=coeffs_dataclass.primary_row,
        primary_col=coeffs_dataclass.primary_col,
        case1=case1_module,
        case2=case2_module,
        mp_plus_m=coeffs_dataclass.mp_plus_m,
        m_minus_mp=coeffs_dataclass.m_minus_mp,
        diagonal_mask=coeffs_dataclass.diagonal_mask,
        anti_diagonal_mask=coeffs_dataclass.anti_diagonal_mask,
        special_2m=coeffs_dataclass.special_2m,
        anti_diag_sign=coeffs_dataclass.anti_diag_sign,
        derived_row=coeffs_dataclass.derived_row,
        derived_col=coeffs_dataclass.derived_col,
        derived_primary_idx=coeffs_dataclass.derived_primary_idx,
        derived_sign=coeffs_dataclass.derived_sign,
    )

    return WignerDataModule(coeffs=coeffs_module, U_blocks=U_blocks)


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


def _allocate_case_coeffs(
    n_primary: int,
    max_poly_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CaseCoeffs:
    """Allocate tensors for one case (Case1 or Case2)."""
    return CaseCoeffs(
        coeff=torch.zeros(n_primary, dtype=dtype, device=device),
        horner=torch.zeros(n_primary, max_poly_len, dtype=dtype, device=device),
        poly_len=torch.zeros(n_primary, dtype=torch.int64, device=device),
        ra_exp=torch.zeros(n_primary, dtype=dtype, device=device),
        rb_exp=torch.zeros(n_primary, dtype=dtype, device=device),
        sign=torch.zeros(n_primary, dtype=dtype, device=device),
    )


def _compute_case_coefficients(
    case: CaseCoeffs,
    idx: int,
    ell: int,
    mp: int,
    m: int,
    sqrt_factor: float,
    factorial: torch.Tensor,
    is_case1: bool,
) -> None:
    """
    Compute polynomial coefficients for Case1 or Case2.

    Case1 (|Ra| >= |Rb|): rho ranges [max(0, mp-m), min(l+mp, l-m)]
    Case2 (|Ra| < |Rb|): rho ranges [max(0, -(mp+m)), min(l-m, l-mp)]

    Args:
        case: CaseCoeffs structure to fill
        idx: Index in the primary element arrays
        ell: Angular momentum quantum number
        mp, m: Magnetic quantum numbers
        sqrt_factor: Precomputed sqrt(factorial ratios)
        factorial: Factorial lookup table
        is_case1: True for Case1, False for Case2
    """
    if is_case1:
        rho_min = max(0, mp - m)
        rho_max = min(ell + mp, ell - m)
    else:
        rho_min = max(0, -(mp + m))
        rho_max = min(ell - m, ell - mp)

    if rho_min > rho_max:
        return

    # Compute leading coefficient
    if is_case1:
        binom1 = _binomial(ell + mp, rho_min, factorial)
        binom2 = _binomial(ell - mp, ell - m - rho_min, factorial)
    else:
        binom1 = _binomial(ell + mp, ell - m - rho_min, factorial)
        binom2 = _binomial(ell - mp, rho_min, factorial)
    case.coeff[idx] = sqrt_factor * binom1 * binom2

    # Polynomial length
    poly_len = rho_max - rho_min + 1
    case.poly_len[idx] = poly_len

    # Horner coefficients (from highest rho down to rho_min+1)
    for i, rho in enumerate(range(rho_max, rho_min, -1)):
        if is_case1:
            n1 = ell + mp - rho + 1
            n2 = ell - m - rho + 1
            d1 = rho
            d2 = m - mp + rho
        else:
            n1 = ell - m - rho + 1
            n2 = ell - mp - rho + 1
            d1 = rho
            d2 = mp + m + rho
        if d1 != 0 and d2 != 0:
            case.horner[idx, i] = (n1 * n2) / (d1 * d2)

    # Exponents
    if is_case1:
        case.ra_exp[idx] = 2 * ell + mp - m - 2 * rho_min
        case.rb_exp[idx] = m - mp + 2 * rho_min
        case.sign[idx] = (-1) ** rho_min
    else:
        case.ra_exp[idx] = mp + m + 2 * rho_min
        case.rb_exp[idx] = 2 * ell - mp - m - 2 * rho_min
        case.sign[idx] = ((-1) ** (ell - m)) * ((-1) ** rho_min)


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
        ratio: The ratio term -(rb/ra)^2 or -(ra/rb)^2, shape (N,)
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
    result = torch.ones(N, n_elements, dtype=dtype, device=device)

    # ratio broadcasted to (N, n_elements)
    ratio_expanded = ratio.unsqueeze(1).expand(N, n_elements)

    # Iterate through Horner steps (from highest term down)
    for i in range(max_poly_len - 1):
        coeff = horner_coeffs[:, i]
        mask = i < (poly_len - 1)
        factor = ratio_expanded * coeff.unsqueeze(0)
        new_result = 1.0 + result * factor
        result = torch.where(mask.unsqueeze(0), new_result, result)

    return result


def _compute_case_magnitude(
    log_ra: torch.Tensor,
    log_rb: torch.Tensor,
    ratio: torch.Tensor,
    case: CaseCoeffs,
    max_poly_len: int,
) -> torch.Tensor:
    """
    Compute the real-valued magnitude factor for a general case.

    This is the common computation for both Case 1 (|Ra| >= |Rb|) and
    Case 2 (|Ra| < |Rb|), used by both complex and real-pair versions.

    Args:
        log_ra: Log of |Ra| magnitudes, shape (N,)
        log_rb: Log of |Rb| magnitudes, shape (N,)
        ratio: -(rb/ra)^2 for Case 1 or -(ra/rb)^2 for Case 2, shape (N,)
        case: CaseCoeffs with polynomial coefficients for this case
        max_poly_len: Maximum polynomial length

    Returns:
        Magnitude factor of shape (N, n_primary), real-valued
    """
    horner_sum = _vectorized_horner(ratio, case.horner, case.poly_len, max_poly_len)
    ra_powers = torch.exp(torch.outer(log_ra, case.ra_exp))
    rb_powers = torch.exp(torch.outer(log_rb, case.rb_exp))
    magnitude = (case.sign * case.coeff) * ra_powers * rb_powers
    return magnitude * horner_sum


def _scatter_primary_to_matrix(
    result: torch.Tensor,
    D: torch.Tensor,
    coeffs: WignerCoefficients,
) -> None:
    """
    Scatter primary element results into the block-diagonal output matrix.

    Args:
        result: Primary element values, shape (N, n_primary)
        D: Output matrix to fill, shape (N, size, size)
        coeffs: WignerCoefficients with primary_row/primary_col indices
    """
    N = result.shape[0]
    device = result.device
    batch_indices = (
        torch.arange(N, device=device).unsqueeze(1).expand(N, coeffs.n_primary)
    )
    row_expanded = coeffs.primary_row.unsqueeze(0).expand(N, coeffs.n_primary)
    col_expanded = coeffs.primary_col.unsqueeze(0).expand(N, coeffs.n_primary)
    D[batch_indices, row_expanded, col_expanded] = result


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

    Args:
        t: Input tensor, will be clamped to [0, 1]

    Returns:
        Smooth step values in [0, 1]
    """
    t_clamped = t.clamp(0, 1)
    eps = torch.finfo(t.dtype).eps

    numerator = 2.0 * t_clamped - 1.0
    denominator = t_clamped * (1.0 - t_clamped)
    denom_safe = denominator.clamp(min=eps)
    arg = numerator / denom_safe
    result = torch.sigmoid(arg)

    result = torch.where(t_clamped < eps, torch.zeros_like(result), result)
    result = torch.where(t_clamped > 1 - eps, torch.ones_like(result), result)

    return result


# =============================================================================
# Quaternion Operations
# =============================================================================


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions: q1 * q2.

    Uses Hamilton product convention: (w, x, y, z).

    Args:
        q1: First quaternion of shape (N, 4) or (4,)
        q2: Second quaternion of shape (N, 4) or (4,)

    Returns:
        Product quaternion of shape (N, 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


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


def quaternion_nlerp(
    q1: torch.Tensor,
    q2: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Normalized linear interpolation between quaternions.

    nlerp(q1, q2, t) = normalize((1-t) * q1 + t * q2)

    Args:
        q1: First quaternion, shape (..., 4)
        q2: Second quaternion, shape (..., 4)
        t: Interpolation parameter, shape (...)

    Returns:
        Interpolated quaternion, shape (..., 4)
    """
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q1_aligned = torch.where(dot < 0, -q1, q1)

    t_expanded = t.unsqueeze(-1) if t.dim() < q1.dim() else t
    result = torch.nn.functional.normalize(
        (1.0 - t_expanded) * q1_aligned + t_expanded * q2, dim=-1
    )

    return result


# =============================================================================
# Two-Chart Quaternion Edge -> +Y
# =============================================================================


def _quaternion_chart1_standard(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Standard quaternion: edge -> +Y directly. Singular at edge = -Y.

    Uses the half-vector formula:
        q = normalize(1 + ey, -ez, 0, ex)

    Args:
        ex, ey, ez: Edge vector components

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) convention
    """
    w = 1.0 + ey
    x = -ez
    y = torch.zeros_like(ex)
    z = ex

    q = torch.stack([w, x, y, z], dim=-1)
    q_sq = torch.sum(q**2, dim=-1, keepdim=True)
    eps = torch.finfo(ex.dtype).eps
    # q_sq → 0 at this chart's singularity (ey = -1), but this chart is
    # unused there so we don't see the divide by zero. The clamp detaches
    # the gradients so that NaNs don't flow through the backward pass.
    norm = torch.sqrt(torch.clamp(q_sq, min=eps))

    return q / norm


def _quaternion_chart2_via_minus_y(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Alternative quaternion: edge -> +Y via -Y. Singular at edge = +Y.

    Path: edge -> -Y -> +Y (compose with 180 deg about X)

    Args:
        ex, ey, ez: Edge vector components

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) convention
    """
    w = -ez
    x = 1.0 - ey
    y = ex
    z = torch.zeros_like(ex)

    q = torch.stack([w, x, y, z], dim=-1)
    q_sq = torch.sum(q**2, dim=-1, keepdim=True)
    eps = torch.finfo(ex.dtype).eps
    # q_sq → 0 at this chart's singularity (ey = +1), but this chart is
    # unused there so we don't see the divide by zero. The clamp detaches
    # the gradients so that NaNs don't flow through the backward pass.
    norm = torch.sqrt(torch.clamp(q_sq, min=eps))

    return q / norm


def quaternion_edge_to_y_stable(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion for edge -> +Y using two charts with NLERP blending.

    Uses two quaternion charts to avoid singularities:
    - Chart 1: q = normalize(1+ey, -ez, 0, ex) - singular at -Y
    - Chart 2: q = normalize(-ez, 1-ey, ex, 0) - singular at +Y

    NLERP blend in ey in [-0.9, 0.9]:
    - Uses Chart 2 when near -Y (stable there)
    - Uses Chart 1 when near +Y (stable there)
    - Smoothly interpolates in between

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    q_chart1 = _quaternion_chart1_standard(ex, ey, ez)
    q_chart2 = _quaternion_chart2_via_minus_y(ex, ey, ez)

    t = (ey - _BLEND_START) / _BLEND_WIDTH
    t_smooth = _smooth_step_cinf(t)

    q = quaternion_nlerp(q_chart2, q_chart1, t_smooth)

    return q


# =============================================================================
# Gamma Computation for Euler Matching
# =============================================================================


def compute_euler_matching_gamma(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute gamma to match the Euler convention.

    Uses a two-chart approach matching the quaternion_edge_to_y_stable function:
    - Chart 1 (ey >= 0.9): gamma = -atan2(ex, ez)
    - Chart 2 (ey <= -0.9): gamma = +atan2(ex, ez)
    - Blend region (-0.9 < ey < 0.9): smooth interpolation

    For edges on Y-axis (ex = ez ~ 0): gamma = 0 (degenerate case).

    Note: In the blend region, there is inherent approximation error
    due to the NLERP quaternion blending. This is acceptable for the intended
    use case of matching Euler output for testing/validation. Properly determined
    gamma values are used in the test.

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Gamma angles of shape (N,)
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    # Chart 1 gamma (used for ey >= 0.9)
    gamma_chart1 = -torch.atan2(ex, ez)

    # Chart 2 gamma (used for ey <= -0.9)
    gamma_chart2 = torch.atan2(ex, ez)

    # Blend factor (same as quaternion_edge_to_y_stable)
    t = (ey - _BLEND_START) / _BLEND_WIDTH
    t_smooth = _smooth_step_cinf(t)

    # Interpolate: t_smooth=0 -> chart2, t_smooth=1 -> chart1
    gamma = t_smooth * gamma_chart1 + (1 - t_smooth) * gamma_chart2

    return gamma


# =============================================================================
# SO(3) Generators and Euler Transform
# =============================================================================


def _compute_transform_sign(ell: int, m: int) -> int:
    """
    Compute the sign for the Euler-matching basis transformation.

    The transformation is a signed row permutation of Jd[ell] that
    converts axis-angle Wigner D matrices to match Euler Wigner D matrices.

    For even |m|: sign = (-1)^((l - |m|) / 2)
    For odd |m|, m < 0: sign = (-1)^((l + |m| + 1) // 2)
    For odd |m|, m > 0: sign = (-1)^((l + |m| + 1) // 2 + 1)
    """
    abs_m = abs(m)
    if abs_m % 2 == 0:
        return (-1) ** ((ell - abs_m) // 2)
    else:
        base = (ell + abs_m + 1) // 2
        if m < 0:
            return (-1) ** base
        else:
            return (-1) ** (base + 1)


def _build_euler_transform(ell: int, Jd: torch.Tensor) -> torch.Tensor:
    """
    Build the basis transformation U for level ell.

    U transforms axis-angle Wigner D to match Euler Wigner D:
        D_euler = U @ D_axis @ U.T

    Args:
        ell: Angular momentum level
        Jd: Wigner d matrix at beta=pi/2 for level ell, shape (2*ell+1, 2*ell+1)

    Returns:
        Orthogonal transformation matrix U of shape (2*ell+1, 2*ell+1)
    """
    size = 2 * ell + 1
    U = torch.zeros(size, size, dtype=Jd.dtype, device=Jd.device)

    for i in range(size):
        m = i - ell
        abs_m = abs(m)
        if abs_m % 2 == 1:
            jd_row = (-m) + ell
        else:
            jd_row = i

        sign = _compute_transform_sign(ell, m)
        U[i, :] = sign * Jd[jd_row, :]

    return U


def _build_u_matrix(
    ell: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build complex-to-real spherical harmonic transformation matrix.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        ell: Angular momentum quantum number
        dtype: Complex data type for the matrix (default: complex128)
        device: Device for the tensor (default: cpu)

    Returns:
        U matrix of shape (2*ell+1, 2*ell+1)
    """
    if device is None:
        device = torch.device("cpu")
    size = 2 * ell + 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    U = torch.zeros(size, size, dtype=dtype, device=device)

    for m in range(-ell, ell + 1):
        row = m + ell

        if m > 0:
            col_pos = m + ell
            col_neg = -m + ell
            sign = (-1) ** m
            U[row, col_pos] = sign * sqrt2_inv
            U[row, col_neg] = sqrt2_inv
        elif m == 0:
            U[row, ell] = 1.0
        else:
            abs_m = abs(m)
            col_pos = abs_m + ell
            col_neg = -abs_m + ell
            sign = (-1) ** abs_m
            U[row, col_neg] = 1j * sqrt2_inv
            U[row, col_pos] = -sign * 1j * sqrt2_inv

    return U


# =============================================================================
# Quaternion to Ra/Rb Decomposition
# =============================================================================


def quaternion_to_ra_rb_real(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into real/imaginary parts of Ra and Rb.

    Uses real arithmetic throughout for torch.compile compatibility.

    For q = (w, x, y, z):
        Ra = w + i*z  ->  (ra_re=w, ra_im=z)
        Rb = y + i*x  ->  (rb_re=y, rb_im=x)

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) convention

    Returns:
        Tuple (ra_re, ra_im, rb_re, rb_im) of real tensors with shape (...)
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    return w, z, y, x


# =============================================================================
# Precomputation of Wigner Coefficients
# =============================================================================


def precompute_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    lmin: int = 0,
) -> WignerCoefficients:
    """
    Precompute Wigner D coefficients for l in [lmin, lmax].

    Uses the symmetry D^l_{-m',-m} = (-1)^{m'-m} x conj(D^l_{m',m}) to compute
    only ~half the elements ("primary") and derive the rest ("derived").

    Primary elements: m' + m > 0, OR (m' + m = 0 AND m' >= 0)

    This version supports an optional lmin parameter for memory-efficient
    computation when lower l values are computed via other methods.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        lmin: Minimum angular momentum (default 0)

    Returns:
        WignerCoefficients dataclass with symmetric coefficient tables
    """
    if device is None:
        device = torch.device("cpu")
    factorial = _factorial_table(2 * lmax + 1, dtype, device)

    # Count elements
    n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))
    n_primary = sum(
        1
        for ell in range(lmin, lmax + 1)
        for mp in range(-ell, ell + 1)
        for m in range(-ell, ell + 1)
        if mp + m > 0 or (mp + m == 0 and mp >= 0)
    )
    n_derived = n_total - n_primary
    max_poly_len = lmax + 1
    size = (lmax + 1) ** 2 - lmin**2

    # Allocate primary element arrays
    primary_row = torch.zeros(n_primary, dtype=torch.int64, device=device)
    primary_col = torch.zeros(n_primary, dtype=torch.int64, device=device)
    mp_plus_m = torch.zeros(n_primary, dtype=dtype, device=device)
    m_minus_mp = torch.zeros(n_primary, dtype=dtype, device=device)

    # Special case arrays
    diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    anti_diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    special_2m = torch.zeros(n_primary, dtype=dtype, device=device)
    anti_diag_sign = torch.zeros(n_primary, dtype=dtype, device=device)

    # Allocate case coefficients using helper
    case1 = _allocate_case_coeffs(n_primary, max_poly_len, dtype, device)
    case2 = _allocate_case_coeffs(n_primary, max_poly_len, dtype, device)

    # Derived element arrays
    derived_row = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_col = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_primary_idx = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_sign = torch.zeros(n_derived, dtype=dtype, device=device)

    primary_map = {}
    primary_idx = 0
    block_start = 0

    # First pass: compute primary elements
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
                    continue

                primary_map[(row, col)] = primary_idx
                primary_row[primary_idx] = row
                primary_col[primary_idx] = col
                mp_plus_m[primary_idx] = mp + m
                m_minus_mp[primary_idx] = m - mp

                diagonal_mask[primary_idx] = mp == m
                anti_diagonal_mask[primary_idx] = mp == -m
                special_2m[primary_idx] = 2 * m
                anti_diag_sign[primary_idx] = (-1) ** (ell - m)

                sqrt_factor = math.sqrt(
                    float(factorial[ell + m] * factorial[ell - m])
                    / float(factorial[ell + mp] * factorial[ell - mp])
                )

                # Compute both cases using helper function
                _compute_case_coefficients(
                    case1,
                    primary_idx,
                    ell,
                    mp,
                    m,
                    sqrt_factor,
                    factorial,
                    is_case1=True,
                )
                _compute_case_coefficients(
                    case2,
                    primary_idx,
                    ell,
                    mp,
                    m,
                    sqrt_factor,
                    factorial,
                    is_case1=False,
                )

                primary_idx += 1

        block_start += block_size

    # Second pass: compute derived elements
    derived_idx = 0
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
                if is_primary:
                    continue

                neg_mp_local = -mp + ell
                neg_m_local = -m + ell
                primary_row_idx = block_start + neg_mp_local
                primary_col_idx = block_start + neg_m_local

                derived_row[derived_idx] = row
                derived_col[derived_idx] = col
                derived_primary_idx[derived_idx] = primary_map[
                    (primary_row_idx, primary_col_idx)
                ]
                derived_sign[derived_idx] = (-1) ** (mp - m)

                derived_idx += 1

        block_start += block_size

    return WignerCoefficients(
        lmin=lmin,
        lmax=lmax,
        size=size,
        max_poly_len=max_poly_len,
        primary_row=primary_row,
        primary_col=primary_col,
        n_primary=n_primary,
        case1=case1,
        case2=case2,
        mp_plus_m=mp_plus_m,
        m_minus_mp=m_minus_mp,
        diagonal_mask=diagonal_mask,
        anti_diagonal_mask=anti_diagonal_mask,
        special_2m=special_2m,
        anti_diag_sign=anti_diag_sign,
        n_derived=n_derived,
        derived_row=derived_row,
        derived_col=derived_col,
        derived_primary_idx=derived_primary_idx,
        derived_sign=derived_sign,
    )


# =============================================================================
# Real-Pair Wigner D Matrix Computation (torch.compile compatible)
# =============================================================================


def wigner_d_matrix_real(
    ra_re: torch.Tensor,
    ra_im: torch.Tensor,
    rb_re: torch.Tensor,
    rb_im: torch.Tensor,
    coeffs: WignerCoefficients,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices using real arithmetic only.

    Uses real-pair arithmetic throughout for torch.compile compatibility.

    Args:
        ra_re, ra_im: Real and imaginary parts of Ra, shape (N,)
        rb_re, rb_im: Real and imaginary parts of Rb, shape (N,)
        coeffs: Precomputed WignerCoefficients from precompute_wigner_coefficients

    Returns:
        Tuple (D_re, D_im) - real and imaginary parts of the complex
        block-diagonal matrices, each of shape (N, size, size)
    """
    N = ra_re.shape[0]
    device = ra_re.device
    input_dtype = ra_re.dtype

    # Upcast to fp64 for numerical stability: this function evaluates
    # degree-2l polynomials (up to degree 12 for lmax=6) and exp/log of
    # magnitudes that can span 100+ orders. Lower precisions overflow in
    # masked-out branches, leaking NaN through torch.where backward.
    # This is a no-op (same tensor object, no copy) when already fp64.
    ra_re = ra_re.to(torch.float64)
    ra_im = ra_im.to(torch.float64)
    rb_re = rb_re.to(torch.float64)
    rb_im = rb_im.to(torch.float64)
    dtype = torch.float64

    # Compute squared magnitudes and masks first.
    # sqrt(0) has gradient 1/(2*sqrt(0)) = inf, causing NaN via autograd
    # even when masked by torch.where (because 0 * inf = NaN in IEEE 754).
    # Clamping the sqrt input prevents this: torch.clamp gradient is 0
    # below min, so the NaN-producing gradient path is cut off.
    eps = torch.finfo(dtype).eps
    eps_sq = eps * eps
    ra_sq = ra_re * ra_re + ra_im * ra_im
    rb_sq = rb_re * rb_re + rb_im * rb_im
    ra_small = ra_sq <= eps_sq
    rb_small = rb_sq <= eps_sq
    ra = torch.sqrt(torch.clamp(ra_sq, min=eps_sq))
    rb = torch.sqrt(torch.clamp(rb_sq, min=eps_sq))
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    # Guard atan2 inputs: (0,0) produces NaN gradient; path is masked,
    # prevents gradient NaN propagation
    safe_ra_re_phi = torch.where(ra_small, torch.ones_like(ra_re), ra_re)
    safe_ra_im_phi = torch.where(ra_small, torch.zeros_like(ra_im), ra_im)
    phia = torch.atan2(safe_ra_im_phi, safe_ra_re_phi)

    safe_rb_re_phi = torch.where(rb_small, torch.ones_like(rb_re), rb_re)
    safe_rb_im_phi = torch.where(rb_small, torch.zeros_like(rb_im), rb_im)
    phib = torch.atan2(safe_rb_im_phi, safe_rb_re_phi)

    phase = torch.outer(phia, coeffs.mp_plus_m) + torch.outer(phib, coeffs.m_minus_mp)
    exp_phase_re = torch.cos(phase)
    exp_phase_im = torch.sin(phase)

    safe_ra = torch.clamp(ra, min=eps)
    safe_rb = torch.clamp(rb, min=eps)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    result_re = torch.zeros(N, coeffs.n_primary, dtype=dtype, device=device)
    result_im = torch.zeros(N, coeffs.n_primary, dtype=dtype, device=device)

    # Special Case 1: |Ra| ~ 0 - anti-diagonal elements
    # phib used since path is masked when rb_small, prevents gradient NaN
    arg_rb = phib

    log_mag_rb_power = torch.outer(log_rb, coeffs.special_2m)
    rb_power_mag = torch.exp(log_mag_rb_power)
    rb_power_phase = torch.outer(arg_rb, coeffs.special_2m)
    rb_power_re = rb_power_mag * torch.cos(rb_power_phase)
    rb_power_im = rb_power_mag * torch.sin(rb_power_phase)

    special_val_antidiag_re = coeffs.anti_diag_sign.unsqueeze(0) * rb_power_re
    special_val_antidiag_im = coeffs.anti_diag_sign.unsqueeze(0) * rb_power_im

    mask_antidiag = ra_small.unsqueeze(1) & coeffs.anti_diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_antidiag, special_val_antidiag_re, result_re)
    result_im = torch.where(mask_antidiag, special_val_antidiag_im, result_im)

    # Special Case 2: |Rb| ~ 0 - diagonal elements
    # phia used since path is masked when ra_small, prevents gradient NaN
    arg_ra = phia

    log_mag_ra_power = torch.outer(log_ra, coeffs.special_2m)
    ra_power_mag = torch.exp(log_mag_ra_power)
    ra_power_phase = torch.outer(arg_ra, coeffs.special_2m)
    ra_power_re = ra_power_mag * torch.cos(ra_power_phase)
    ra_power_im = ra_power_mag * torch.sin(ra_power_phase)

    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & coeffs.diagonal_mask.unsqueeze(0)
    result_re = torch.where(mask_diag, ra_power_re, result_re)
    result_im = torch.where(mask_diag, ra_power_im, result_im)

    # General Case 1: |Ra| >= |Rb|
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    real_factor1 = _compute_case_magnitude(
        log_ra, log_rb, ratio1, coeffs.case1, coeffs.max_poly_len
    )
    val1_re = real_factor1 * exp_phase_re
    val1_im = real_factor1 * exp_phase_im

    valid_case1 = coeffs.case1.poly_len > 0
    mask1 = use_case1.unsqueeze(1) & valid_case1.unsqueeze(0)
    result_re = torch.where(mask1, val1_re, result_re)
    result_im = torch.where(mask1, val1_im, result_im)

    # General Case 2: |Ra| < |Rb|
    ratio2 = -(ra * ra) / (safe_rb * safe_rb)
    real_factor2 = _compute_case_magnitude(
        log_ra, log_rb, ratio2, coeffs.case2, coeffs.max_poly_len
    )
    val2_re = real_factor2 * exp_phase_re
    val2_im = real_factor2 * exp_phase_im

    valid_case2 = coeffs.case2.poly_len > 0
    mask2 = use_case2.unsqueeze(1) & valid_case2.unsqueeze(0)
    result_re = torch.where(mask2, val2_re, result_re)
    result_im = torch.where(mask2, val2_im, result_im)

    # Scatter primary results into output matrix
    D_re = torch.zeros(N, coeffs.size, coeffs.size, dtype=dtype, device=device)
    D_im = torch.zeros(N, coeffs.size, coeffs.size, dtype=dtype, device=device)
    _scatter_primary_to_matrix(result_re, D_re, coeffs)
    _scatter_primary_to_matrix(result_im, D_im, coeffs)

    # Fill derived elements using symmetry
    if coeffs.n_derived > 0:
        primary_re = result_re[:, coeffs.derived_primary_idx]
        primary_im = result_im[:, coeffs.derived_primary_idx]

        derived_sign_expanded = coeffs.derived_sign.unsqueeze(0)
        derived_re = derived_sign_expanded * primary_re
        derived_im = -derived_sign_expanded * primary_im

        batch_indices_d = (
            torch.arange(N, device=device).unsqueeze(1).expand(N, coeffs.n_derived)
        )
        row_expanded_d = coeffs.derived_row.unsqueeze(0).expand(N, coeffs.n_derived)
        col_expanded_d = coeffs.derived_col.unsqueeze(0).expand(N, coeffs.n_derived)

        D_re[batch_indices_d, row_expanded_d, col_expanded_d] = derived_re
        D_im[batch_indices_d, row_expanded_d, col_expanded_d] = derived_im

    return D_re.to(input_dtype), D_im.to(input_dtype)


# =============================================================================
# Complex to Real Spherical Harmonics Transformation (U blocks)
# =============================================================================


def _precompute_U_blocks_euler_aligned(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    lmin: int = 0,
) -> list[torch.Tensor]:
    """
    Private helper to precompute complex U transformation matrices.

    Used internally by precompute_U_blocks_euler_aligned_real.

    This combines the complex->real transformation with:
    - For l=1: The Cartesian permutation P (m-ordering -> x,y,z)
    - For l>=2: The Euler-matching basis transformation

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device
        lmin: Minimum angular momentum (default 0)

    Returns:
        List of combined U matrices (complex) where U_blocks[i] corresponds to l=lmin+i
    """
    if device is None:
        device = torch.device("cpu")
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        complex_dtype = torch.complex128

    P = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=complex_dtype,
        device=device,
    )

    jd_path = Path(__file__).parent.parent / "Jd.pt"
    Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

    U_combined = []
    for ell in range(lmin, lmax + 1):
        # Build U block directly
        U_ell = _build_u_matrix(ell, complex_dtype, device)

        if ell == 0:
            U_combined.append(U_ell)
        elif ell == 1:
            U_combined.append(P @ U_ell)
        else:
            Jd = Jd_list[ell].to(dtype=dtype, device=device)
            U_euler = _build_euler_transform(ell, Jd).to(
                dtype=complex_dtype, device=device
            )
            U_combined.append(U_euler @ U_ell)

    return U_combined


def precompute_U_blocks_euler_aligned_real(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute Euler-aligned U transformation matrices as real/imag pairs.

    This is a torch.compile-compatible version.

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype
        device: Torch device

    Returns:
        List of (U_re, U_im) tuples where each has shape (2*ell+1, 2*ell+1)
    """
    if device is None:
        device = torch.device("cpu")
    U_blocks_complex = _precompute_U_blocks_euler_aligned(
        lmax, dtype=dtype, device=device
    )
    return [(U.real.to(dtype=dtype), U.imag.to(dtype=dtype)) for U in U_blocks_complex]


def wigner_d_pair_to_real(
    D_re: torch.Tensor,
    D_im: torch.Tensor,
    U_blocks_real: list[tuple[torch.Tensor, torch.Tensor]],
    lmax: int,
    lmin: int = 0,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from real-pair to real basis using real arithmetic.

    Uses real arithmetic throughout for torch.compile compatibility.

    Args:
        D_re: Real part of complex Wigner D matrices, shape (N, size, size)
        D_im: Imaginary part of complex Wigner D matrices, shape (N, size, size)
        U_blocks_real: List of (U_re, U_im) tuples for l in [lmin, lmax]
        lmax: Maximum angular momentum
        lmin: Minimum angular momentum (default 0)

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

        D_block_re = D_re[:, block_start:block_end, block_start:block_end]
        D_block_im = D_im[:, block_start:block_end, block_start:block_end]

        U_re, U_im = U_blocks_real[idx]
        U_re = U_re.to(dtype=dtype, device=device)
        U_im = U_im.to(dtype=dtype, device=device)

        U_re_T = U_re.T
        U_im_T = U_im.T

        temp_re = torch.matmul(D_block_re, U_re_T) + torch.matmul(D_block_im, U_im_T)
        temp_im = torch.matmul(D_block_im, U_re_T) - torch.matmul(D_block_re, U_im_T)

        result_re = torch.matmul(U_re, temp_re) - torch.matmul(U_im, temp_im)

        D_real[:, block_start:block_end, block_start:block_end] = result_re

        block_start = block_end

    return D_real


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

    version = "v2"  # Bumped for dataclass format
    return cache_dir / f"wigner_{variant}_lmax{lmax}_{version}.pt"


def _coeffs_to_dict(coeffs: WignerCoefficients) -> dict:
    """Convert WignerCoefficients dataclass to dict for serialization."""
    return {
        "lmin": coeffs.lmin,
        "lmax": coeffs.lmax,
        "size": coeffs.size,
        "max_poly_len": coeffs.max_poly_len,
        "primary_row": coeffs.primary_row,
        "primary_col": coeffs.primary_col,
        "n_primary": coeffs.n_primary,
        "case1_coeff": coeffs.case1.coeff,
        "case1_horner": coeffs.case1.horner,
        "case1_poly_len": coeffs.case1.poly_len,
        "case1_ra_exp": coeffs.case1.ra_exp,
        "case1_rb_exp": coeffs.case1.rb_exp,
        "case1_sign": coeffs.case1.sign,
        "case2_coeff": coeffs.case2.coeff,
        "case2_horner": coeffs.case2.horner,
        "case2_poly_len": coeffs.case2.poly_len,
        "case2_ra_exp": coeffs.case2.ra_exp,
        "case2_rb_exp": coeffs.case2.rb_exp,
        "case2_sign": coeffs.case2.sign,
        "mp_plus_m": coeffs.mp_plus_m,
        "m_minus_mp": coeffs.m_minus_mp,
        "diagonal_mask": coeffs.diagonal_mask,
        "anti_diagonal_mask": coeffs.anti_diagonal_mask,
        "special_2m": coeffs.special_2m,
        "anti_diag_sign": coeffs.anti_diag_sign,
        "n_derived": coeffs.n_derived,
        "derived_row": coeffs.derived_row,
        "derived_col": coeffs.derived_col,
        "derived_primary_idx": coeffs.derived_primary_idx,
        "derived_sign": coeffs.derived_sign,
    }


def _dict_to_coeffs(d: dict) -> WignerCoefficients:
    """Convert dict back to WignerCoefficients dataclass."""
    case1 = CaseCoeffs(
        coeff=d["case1_coeff"],
        horner=d["case1_horner"],
        poly_len=d["case1_poly_len"],
        ra_exp=d["case1_ra_exp"],
        rb_exp=d["case1_rb_exp"],
        sign=d["case1_sign"],
    )
    case2 = CaseCoeffs(
        coeff=d["case2_coeff"],
        horner=d["case2_horner"],
        poly_len=d["case2_poly_len"],
        ra_exp=d["case2_ra_exp"],
        rb_exp=d["case2_rb_exp"],
        sign=d["case2_sign"],
    )
    return WignerCoefficients(
        lmin=d["lmin"],
        lmax=d["lmax"],
        size=d["size"],
        max_poly_len=d["max_poly_len"],
        primary_row=d["primary_row"],
        primary_col=d["primary_col"],
        n_primary=d["n_primary"],
        case1=case1,
        case2=case2,
        mp_plus_m=d["mp_plus_m"],
        m_minus_mp=d["m_minus_mp"],
        diagonal_mask=d["diagonal_mask"],
        anti_diagonal_mask=d["anti_diagonal_mask"],
        special_2m=d["special_2m"],
        anti_diag_sign=d["anti_diag_sign"],
        n_derived=d["n_derived"],
        derived_row=d["derived_row"],
        derived_col=d["derived_col"],
        derived_primary_idx=d["derived_primary_idx"],
        derived_sign=d["derived_sign"],
    )


def _save_coefficients(coeffs: WignerCoefficients, path: Path) -> None:
    """Save coefficients to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coeffs_dict = _coeffs_to_dict(coeffs)
    coeffs_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in coeffs_dict.items()
    }
    torch.save(coeffs_cpu, path)


def _load_coefficients(
    path: Path,
    device: torch.device,
) -> Optional[WignerCoefficients]:
    """Load coefficients from disk, returning None if not found or invalid."""
    if not path.exists():
        return None
    coeffs_dict = torch.load(path, map_location=device, weights_only=True)
    return _dict_to_coeffs(coeffs_dict)


def get_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> WignerCoefficients:
    """
    Get precomputed Wigner D coefficients, loading from cache if available.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching

    Returns:
        WignerCoefficients with precomputed coefficient tensors
    """
    if device is None:
        device = torch.device("cpu")
    cache_path = _get_cache_path(lmax, "symmetric", cache_dir)

    if use_cache:
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None and coeffs.lmax == lmax and dtype != torch.float64:
            # Convert floating point tensors to requested dtype
            def convert_case(case: CaseCoeffs) -> CaseCoeffs:
                return CaseCoeffs(
                    coeff=case.coeff.to(dtype=dtype),
                    horner=case.horner.to(dtype=dtype),
                    poly_len=case.poly_len,
                    ra_exp=case.ra_exp.to(dtype=dtype),
                    rb_exp=case.rb_exp.to(dtype=dtype),
                    sign=case.sign.to(dtype=dtype),
                )

            coeffs = WignerCoefficients(
                lmin=coeffs.lmin,
                lmax=coeffs.lmax,
                size=coeffs.size,
                max_poly_len=coeffs.max_poly_len,
                primary_row=coeffs.primary_row,
                primary_col=coeffs.primary_col,
                n_primary=coeffs.n_primary,
                case1=convert_case(coeffs.case1),
                case2=convert_case(coeffs.case2),
                mp_plus_m=coeffs.mp_plus_m.to(dtype=dtype),
                m_minus_mp=coeffs.m_minus_mp.to(dtype=dtype),
                diagonal_mask=coeffs.diagonal_mask,
                anti_diagonal_mask=coeffs.anti_diagonal_mask,
                special_2m=coeffs.special_2m.to(dtype=dtype),
                anti_diag_sign=coeffs.anti_diag_sign.to(dtype=dtype),
                n_derived=coeffs.n_derived,
                derived_row=coeffs.derived_row,
                derived_col=coeffs.derived_col,
                derived_primary_idx=coeffs.derived_primary_idx,
                derived_sign=coeffs.derived_sign.to(dtype=dtype),
            )
            return coeffs

    coeffs = precompute_wigner_coefficients(lmax, dtype, device)

    if use_cache:
        _save_coefficients(coeffs, cache_path)

    return coeffs


def clear_wigner_cache(cache_dir: Optional[Path] = None) -> int:
    """
    Clear all cached Wigner coefficient files.

    Args:
        cache_dir: Directory to clear

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
        f.unlink()
        count += 1

    return count


# =============================================================================
# Reference Implementation for Testing (Not Used at Runtime)
# =============================================================================
#
# The following functions provide a mathematically principled reference
# implementation for computing Wigner D matrices via matrix exponential of
# SO(3) Lie algebra generators. They are kept for verification purposes in
# tests, which validate that the optimized polynomial kernels in
# wigner_d_custom_kernels.py produce correct results.
#
# These functions are NOT used by any runtime code paths.
# =============================================================================

# Cache for SO(3) generators (used only by reference implementation)
_GENERATOR_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}


def quaternion_to_axis_angle(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert quaternion to axis-angle representation.

    Uses the stable formula:
        angle = 2 * atan2(|xyz|, w)
        axis = xyz / |xyz|

    For small angles (|xyz| ~ 0), axis is undefined but angle ~ 0.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        (axis, angle) where:
        - axis has shape (N, 3), unit vectors
        - angle has shape (N,), in radians
    """
    w = q[..., 0]
    xyz = q[..., 1:4]

    xyz_norm = torch.linalg.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(xyz_norm, w)

    safe_xyz_norm = xyz_norm.clamp(min=1e-12)
    axis = xyz / safe_xyz_norm.unsqueeze(-1)

    small_angle = xyz_norm < 1e-8
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
    z_axis = z_axis.expand_as(axis)
    axis = torch.where(small_angle.unsqueeze(-1), z_axis, axis)

    return axis, angle


def _build_so3_generators(ell: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build SO(3) Lie algebra generators K_x, K_y, K_z for representation ell.

    These are real antisymmetric (2*ell+1) x (2*ell+1) matrices satisfying:
        D^ell(n, theta) = exp(theta * (n_x K_x + n_y K_y + n_z K_z))

    Args:
        ell: Angular momentum quantum number

    Returns:
        (K_x, K_y, K_z) tuple of generator matrices in float64
    """
    size = 2 * ell + 1

    if ell == 0:
        z = torch.zeros(1, 1, dtype=torch.float64)
        return z, z.clone(), z.clone()

    m_values = torch.arange(-ell, ell + 1, dtype=torch.float64)
    J_z = torch.diag(m_values.to(torch.complex128))

    J_plus = torch.zeros(size, size, dtype=torch.complex128)
    J_minus = torch.zeros(size, size, dtype=torch.complex128)

    for m in range(-ell, ell):
        coeff = math.sqrt(ell * (ell + 1) - m * (m + 1))
        J_plus[m + 1 + ell, m + ell] = coeff

    for m in range(-ell + 1, ell + 1):
        coeff = math.sqrt(ell * (ell + 1) - m * (m - 1))
        J_minus[m - 1 + ell, m + ell] = coeff

    J_x = (J_plus + J_minus) / 2
    J_y = (J_plus - J_minus) / 2j

    U = _build_u_matrix(ell)
    U_dag = U.conj().T

    K_x = (U @ (1j * J_x) @ U_dag).real
    K_y = -(U @ (1j * J_y) @ U_dag).real
    K_z = (U @ (1j * J_z) @ U_dag).real

    return K_x, K_y, K_z


def get_so3_generators(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    """
    Return cached K_x, K_y, K_z lists for l=0..lmax.

    For l >= 2, the generators include the Euler-matching transformation folded in,
    so the matrix exponential produces output directly in the Euler basis.

    For l=1, a permutation matrix P is also cached to convert to Cartesian basis.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for the generators
        device: Device for the generators

    Returns:
        Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P' for l=1 permutation
    """
    key = (lmax, dtype, device)

    if key not in _GENERATOR_CACHE:
        jd_path = Path(__file__).parent.parent / "Jd.pt"
        Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

        K_x_list = []
        K_y_list = []
        K_z_list = []

        for ell in range(lmax + 1):
            K_x, K_y, K_z = _build_so3_generators(ell)
            K_x = K_x.to(device=device, dtype=dtype)
            K_y = K_y.to(device=device, dtype=dtype)
            K_z = K_z.to(device=device, dtype=dtype)

            if ell >= 2:
                Jd = Jd_list[ell].to(dtype=dtype, device=device)
                U = _build_euler_transform(ell, Jd)
                K_x = U @ K_x @ U.T
                K_y = U @ K_y @ U.T
                K_z = U @ K_z @ U.T

            K_x_list.append(K_x)
            K_y_list.append(K_y)
            K_z_list.append(K_z)

        P = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        )

        _GENERATOR_CACHE[key] = {
            "K_x": K_x_list,
            "K_y": K_y_list,
            "K_z": K_z_list,
            "P": P,
        }

    return _GENERATOR_CACHE[key]
