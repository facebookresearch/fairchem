"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    CustomKernelModule,
)

# =============================================================================
# Data Structures for Wigner Coefficients
# =============================================================================


class CaseCoeffs(nn.Module):
    """
    Polynomial coefficients for one case (|Ra|>=|Rb| or |Ra|<|Rb|).
    """

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


class WignerCoefficients(nn.Module):
    """
    Precomputed coefficients for Wigner D matrix computation.
    """

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
        case1: CaseCoeffs,
        case2: CaseCoeffs,
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
    Combined Wigner coefficients, U transformation blocks, and custom kernel
    data as nn.Module.

    This module holds all precomputed data needed for Wigner D computation,
    and automatically moves with the parent model via .to(device).

    U_blocks are stored as real/imaginary pairs for torch.compile compatibility.
    """

    def __init__(
        self,
        coeffs: WignerCoefficients,
        U_blocks: list[tuple[torch.Tensor, torch.Tensor]],
        custom_kernels: CustomKernelModule,
    ):
        super().__init__()
        # Register as submodule so .to(device/dtype) propagates
        self.coeffs = coeffs
        self.custom_kernels = custom_kernels

        # Register U_blocks as non-persistent buffers (computed, not learned)
        # Each U_block is a (U_re, U_im) tuple
        self._n_U_blocks = len(U_blocks)
        for i, (U_re, U_im) in enumerate(U_blocks):
            self.register_buffer(f"U_block_{i}_re", U_re, persistent=False)
            self.register_buffer(f"U_block_{i}_im", U_im, persistent=False)

    @property
    def U_blocks(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Return U_blocks as a list of (U_re, U_im) tuples.
        """
        return [
            (getattr(self, f"U_block_{i}_re"), getattr(self, f"U_block_{i}_im"))
            for i in range(self._n_U_blocks)
        ]


# =============================================================================
# Constants
# =============================================================================

# Default cache directory for precomputed coefficients
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fairchem" / "wigner_coeffs"


def create_wigner_data_module(
    lmax: int,
    lmin: int = 5,
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
        lmin: Minimum angular momentum for Ra/Rb polynomial path (default 5,
              matching hybrid method which uses custom kernels for l=0..4)

    Returns:
        WignerDataModule containing coefficients and U_blocks
    """
    # Create on CPU with float64 (will be converted when model moves)
    dtype = torch.float64
    device = torch.device("cpu")

    coeffs = get_wigner_coefficients(lmax, dtype=dtype, device=device, lmin=lmin)
    full_U_blocks_real = precompute_U_blocks_euler_aligned_real(
        lmax, dtype=dtype, device=device
    )
    U_blocks = full_U_blocks_real[lmin:]

    return WignerDataModule(
        coeffs=coeffs,
        U_blocks=U_blocks,
        custom_kernels=CustomKernelModule(),
    )


# =============================================================================
# Core Helper Functions
# =============================================================================


def _factorial_table(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Compute factorial table [0!, 1!, 2!, ..., n!].
    """
    table = torch.zeros(n + 1, dtype=dtype, device=device)
    table[0] = 1.0
    for i in range(1, n + 1):
        table[i] = table[i - 1] * i
    return table


def _binomial(n: int, k: int, factorial: torch.Tensor) -> float:
    """
    Compute binomial coefficient C(n, k) using precomputed factorials.
    """
    if k < 0 or k > n:
        return 0.0
    return float(factorial[n] / (factorial[k] * factorial[n - k]))


def _allocate_case_coeffs(
    n_primary: int,
    max_poly_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CaseCoeffs:
    """
    Allocate tensors for one case (Case1 or Case2).
    """
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
        WignerCoefficients with symmetric coefficient tables
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
    lmin: int,
    dtype: torch.dtype,
    cache_dir: Path | None = None,
) -> Path:
    """
    Get the cache file path for a given lmax, lmin, dtype.

    Args:
        lmax: Maximum angular momentum
        lmin: Minimum angular momentum
        dtype: Data type for coefficients
        cache_dir: Override cache directory (default: ~/.cache/fairchem/wigner_coeffs)

    Returns:
        Path to the cache file
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    dtype_str = str(dtype).replace("torch.", "")
    version = "v3"
    return cache_dir / f"wigner_lmin{lmin}_lmax{lmax}_{dtype_str}_{version}.pt"


def _coeffs_to_dict(coeffs: WignerCoefficients) -> dict:
    """
    Convert WignerCoefficients to dict for serialization.
    """
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
    """
    Convert dict back to WignerCoefficients.
    """
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
    """
    Save coefficients to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    coeffs_dict = _coeffs_to_dict(coeffs)
    coeffs_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in coeffs_dict.items()
    }
    torch.save(coeffs_cpu, path)


def _load_coefficients(
    path: Path,
    device: torch.device,
) -> WignerCoefficients | None:
    """
    Load coefficients from disk, returning None if not found.
    """
    if not path.exists():
        return None
    coeffs_dict = torch.load(path, map_location=device, weights_only=True)
    return _dict_to_coeffs(coeffs_dict)


def get_wigner_coefficients(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    lmin: int = 0,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> WignerCoefficients:
    """
    Get precomputed Wigner D coefficients, loading from disk cache if available.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        lmin: Minimum angular momentum (default 0)
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching

    Returns:
        WignerCoefficients with precomputed coefficient tensors
    """
    if device is None:
        device = torch.device("cpu")

    if use_cache:
        cache_path = _get_cache_path(lmax, lmin, dtype, cache_dir)
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None:
            return coeffs

    coeffs = precompute_wigner_coefficients(lmax, dtype, device, lmin=lmin)

    if use_cache:
        _save_coefficients(coeffs, cache_path)

    return coeffs


def clear_wigner_cache(cache_dir: Path | None = None) -> int:
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
