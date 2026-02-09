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
# Global Caches (consolidated from all modules)
# =============================================================================

# Generator cache for SO(3) Lie algebra generators
_GENERATOR_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}

# Coefficient caches for Ra/Rb polynomial
_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}

# U block caches (complex and real-pair)
_U_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}
_U_REAL_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}


def clear_memory_caches() -> None:
    """
    Clear all in-memory caches for Wigner D computation.

    This clears the module-level dictionaries that cache generators,
    transforms, coefficients, and U blocks. Useful for testing or reducing memory.

    Also clears caches in the individual method modules if they are loaded.
    """
    _GENERATOR_CACHE.clear()
    _COEFF_CACHE.clear()
    _U_CACHE.clear()
    _U_REAL_CACHE.clear()

    # Also clear caches in method modules if they exist
    import sys
    for module_name in [
        "fairchem.core.models.uma.common.wigner_d_matexp",
        "fairchem.core.models.uma.common.wigner_d_hybrid",
        "fairchem.core.models.uma.common.wigner_d_polynomial",
    ]:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "clear_memory_caches"):
                module.clear_memory_caches()


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
    eps = 1e-7

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
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    safe_norm = norm.clamp(min=1e-12)

    return q / safe_norm


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
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    safe_norm = norm.clamp(min=1e-12)

    return q / safe_norm


def quaternion_edge_to_y_stable(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion for edge -> +Y using two charts with NLERP blending.

    Uses two quaternion charts to avoid singularities:
    - Chart 1: q = normalize(1+ey, -ez, 0, ex) - singular at -Y
    - Chart 2: q = normalize(-ez, 1-ey, ex, 0) - singular at +Y

    NLERP blend in ey in [-0.9, -0.7]:
    - Uses Chart 2 when near -Y (stable there)
    - Uses Chart 1 elsewhere (stable away from -Y)

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

    blend_start = -0.9
    blend_width = 0.2
    t = (ey - blend_start) / blend_width
    t_smooth = _smooth_step_cinf(t)

    q = quaternion_nlerp(q_chart2, q_chart1, t_smooth)

    return q


# =============================================================================
# Quaternion to Axis-Angle
# =============================================================================


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


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion directly to 3x3 rotation matrix (l=1 Wigner D).

    Uses the standard quaternion to rotation matrix formula.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        torch.stack([1 - 2*(y2 + z2), 2*(xy - wz),     2*(xz + wy)    ], dim=-1),
        torch.stack([2*(xy + wz),     1 - 2*(x2 + z2), 2*(yz - wx)    ], dim=-1),
        torch.stack([2*(xz - wy),     2*(yz + wx),     1 - 2*(x2 + y2)], dim=-1),
    ], dim=-2)

    return R


# =============================================================================
# Gamma Computation for Euler Matching
# =============================================================================


def compute_euler_matching_gamma(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute gamma to match the Euler convention.

    gamma = -atan2(ex, ez) is the roll correction that aligns the
    Rodrigues rotation with the ZYZ Euler decomposition.

    For edges on Y-axis (ex = ez ~ 0): gamma = 0 (degenerate case).

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Gamma angles of shape (N,)
    """
    ex = edge_vec[..., 0]
    ez = edge_vec[..., 2]

    gamma = -torch.atan2(ex, ez)

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


def _build_u_matrix(ell: int) -> torch.Tensor:
    """
    Build complex-to-real spherical harmonic transformation matrix.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).
    """
    size = 2 * ell + 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    U = torch.zeros(size, size, dtype=torch.complex128)

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

        P = torch.tensor([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=dtype, device=device)

        _GENERATOR_CACHE[key] = {
            'K_x': K_x_list,
            'K_y': K_y_list,
            'K_z': K_z_list,
            'P': P,
        }

    return _GENERATOR_CACHE[key]


# =============================================================================
# Quaternion to Ra/Rb Decomposition
# =============================================================================


def quaternion_to_ra_rb(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into complex numbers Ra and Rb.

    For q = (w, x, y, z):
        Ra = w + i*z
        Rb = y + i*x

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


def quaternion_to_ra_rb_real(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose quaternion into real/imaginary parts of Ra and Rb.

    This is a torch.compile-compatible alternative to quaternion_to_ra_rb
    that avoids creating complex tensors.

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
    device: torch.device = torch.device("cpu"),
    lmin: int = 0,
) -> dict[str, torch.Tensor]:
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
        Dictionary with symmetric coefficient tables for l in [lmin, lmax]
    """
    factorial = _factorial_table(2 * lmax + 1, dtype, device)

    n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))
    block_offset = lmin * lmin

    n_primary = 0
    for ell in range(lmin, lmax + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                if mp + m > 0 or (mp + m == 0 and mp >= 0):
                    n_primary += 1

    max_poly_len = lmax + 1

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

    diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    anti_diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
    special_2m = torch.zeros(n_primary, dtype=dtype, device=device)
    anti_diag_sign = torch.zeros(n_primary, dtype=dtype, device=device)

    n_derived = n_total - n_primary
    derived_row_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_col_indices = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_primary_idx = torch.zeros(n_derived, dtype=torch.int64, device=device)
    derived_sign = torch.zeros(n_derived, dtype=dtype, device=device)

    primary_map = {}

    primary_idx = 0
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
                    primary_map[(row, col)] = primary_idx

                    primary_row_indices[primary_idx] = row
                    primary_col_indices[primary_idx] = col

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
        "primary_row_indices": primary_row_indices,
        "primary_col_indices": primary_col_indices,
        "case1_coeff": case1_coeff,
        "case1_horner": case1_horner,
        "case1_poly_len": case1_poly_len,
        "case1_ra_exp": case1_ra_exp,
        "case1_rb_exp": case1_rb_exp,
        "case1_sign": case1_sign,
        "case2_coeff": case2_coeff,
        "case2_horner": case2_horner,
        "case2_poly_len": case2_poly_len,
        "case2_ra_exp": case2_ra_exp,
        "case2_rb_exp": case2_rb_exp,
        "case2_sign": case2_sign,
        "mp_plus_m": mp_plus_m,
        "m_minus_mp": m_minus_mp,
        "diagonal_mask": diagonal_mask,
        "anti_diagonal_mask": anti_diagonal_mask,
        "special_2m": special_2m,
        "anti_diag_sign": anti_diag_sign,
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
    coeffs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute complex Wigner D matrices exploiting conjugate symmetry.

    Computes only primary elements (~half) and derives the rest via:
        D^l_{-m',-m} = (-1)^{m'-m} x conj(D^l_{m',m})

    Args:
        Ra, Rb: Complex Cayley-Klein parameters, shape (N,)
        coeffs: Precomputed coefficient dictionary from precompute_wigner_coefficients

    Returns:
        Complex block-diagonal matrices of shape (N, size, size)
    """
    N = Ra.shape[0]
    device = Ra.device
    complex_dtype = Ra.dtype

    n_primary = coeffs["n_primary"]
    max_poly_len = coeffs["max_poly_len"]
    size = coeffs["size"]

    primary_row_indices = coeffs["primary_row_indices"]
    primary_col_indices = coeffs["primary_col_indices"]

    case1_coeff = coeffs["case1_coeff"]
    case1_horner = coeffs["case1_horner"]
    case1_poly_len = coeffs["case1_poly_len"]
    case1_ra_exp = coeffs["case1_ra_exp"]
    case1_rb_exp = coeffs["case1_rb_exp"]
    case1_sign = coeffs["case1_sign"]

    case2_coeff = coeffs["case2_coeff"]
    case2_horner = coeffs["case2_horner"]
    case2_poly_len = coeffs["case2_poly_len"]
    case2_ra_exp = coeffs["case2_ra_exp"]
    case2_rb_exp = coeffs["case2_rb_exp"]
    case2_sign = coeffs["case2_sign"]

    mp_plus_m = coeffs["mp_plus_m"]
    m_minus_mp = coeffs["m_minus_mp"]

    diagonal_mask = coeffs["diagonal_mask"]
    anti_diagonal_mask = coeffs["anti_diagonal_mask"]
    special_2m = coeffs["special_2m"]
    anti_diag_sign = coeffs["anti_diag_sign"]

    derived_row_indices = coeffs["derived_row_indices"]
    derived_col_indices = coeffs["derived_col_indices"]
    derived_primary_idx = coeffs["derived_primary_idx"]
    derived_sign = coeffs["derived_sign"]

    ra = torch.abs(Ra)
    rb = torch.abs(Rb)

    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    phia = torch.angle(Ra)
    phib = torch.angle(Rb)
    phase = torch.outer(phia, mp_plus_m) + torch.outer(phib, m_minus_mp)
    exp_phase = torch.exp(1j * phase)

    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    result = torch.zeros(N, n_primary, dtype=complex_dtype, device=device)

    # Special Case 1: |Ra| ~ 0 - anti-diagonal elements
    safe_Rb = torch.where(rb < EPSILON, torch.ones_like(Rb), Rb)
    log_abs_Rb = torch.log(torch.abs(safe_Rb))
    arg_Rb = torch.angle(safe_Rb)
    exponent = torch.outer(log_abs_Rb, special_2m) + 1j * torch.outer(arg_Rb, special_2m)
    rb_power = torch.exp(exponent.to(dtype=complex_dtype))
    special_val_antidiag = anti_diag_sign.unsqueeze(0).to(complex_dtype) * rb_power
    mask_antidiag = ra_small.unsqueeze(1) & anti_diagonal_mask.unsqueeze(0)
    result = torch.where(mask_antidiag, special_val_antidiag, result)

    # Special Case 2: |Rb| ~ 0 - diagonal elements
    safe_Ra = torch.where(ra < EPSILON, torch.ones_like(Ra), Ra)
    log_abs_Ra = torch.log(torch.abs(safe_Ra))
    arg_Ra = torch.angle(safe_Ra)
    exponent = torch.outer(log_abs_Ra, special_2m) + 1j * torch.outer(arg_Ra, special_2m)
    ra_power = torch.exp(exponent.to(dtype=complex_dtype))
    mask_diag = (rb_small & ~ra_small).unsqueeze(1) & diagonal_mask.unsqueeze(0)
    result = torch.where(mask_diag, ra_power, result)

    # General Case 1: |Ra| >= |Rb|
    ratio1 = -(rb * rb) / (safe_ra * safe_ra)
    horner_sum1 = _vectorized_horner(ratio1, case1_horner, case1_poly_len, max_poly_len)

    ra_powers1 = torch.exp(torch.outer(log_ra, case1_ra_exp))
    rb_powers1 = torch.exp(torch.outer(log_rb, case1_rb_exp))

    magnitude1 = (case1_sign * case1_coeff) * ra_powers1 * rb_powers1
    val1 = magnitude1 * horner_sum1 * exp_phase

    valid_case1 = case1_poly_len > 0
    mask1 = use_case1.unsqueeze(1) & valid_case1.unsqueeze(0)
    result = torch.where(mask1, val1.to(dtype=complex_dtype), result)

    # General Case 2: |Ra| < |Rb|
    ratio2 = -(ra * ra) / (safe_rb * safe_rb)
    horner_sum2 = _vectorized_horner(ratio2, case2_horner, case2_poly_len, max_poly_len)

    ra_powers2 = torch.exp(torch.outer(log_ra, case2_ra_exp))
    rb_powers2 = torch.exp(torch.outer(log_rb, case2_rb_exp))

    magnitude2 = (case2_sign * case2_coeff) * ra_powers2 * rb_powers2
    val2 = magnitude2 * horner_sum2 * exp_phase

    valid_case2 = case2_poly_len > 0
    mask2 = use_case2.unsqueeze(1) & valid_case2.unsqueeze(0)
    result = torch.where(mask2, val2.to(dtype=complex_dtype), result)

    # Scatter primary results into output matrix
    D = torch.zeros(N, size, size, dtype=complex_dtype, device=device)

    batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, n_primary)
    row_expanded = primary_row_indices.unsqueeze(0).expand(N, n_primary)
    col_expanded = primary_col_indices.unsqueeze(0).expand(N, n_primary)

    D[batch_indices, row_expanded, col_expanded] = result

    # Fill derived elements using symmetry
    n_derived = coeffs["n_derived"]
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
    coeffs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D matrices using real arithmetic only.

    This is a torch.compile-compatible alternative to wigner_d_matrix_complex.

    Args:
        ra_re, ra_im: Real and imaginary parts of Ra, shape (N,)
        rb_re, rb_im: Real and imaginary parts of Rb, shape (N,)
        coeffs: Precomputed coefficient dictionary from precompute_wigner_coefficients

    Returns:
        Tuple (D_re, D_im) - real and imaginary parts of the complex
        block-diagonal matrices, each of shape (N, size, size)
    """
    N = ra_re.shape[0]
    device = ra_re.device
    dtype = ra_re.dtype

    n_primary = coeffs["n_primary"]
    max_poly_len = coeffs["max_poly_len"]
    size = coeffs["size"]

    primary_row_indices = coeffs["primary_row_indices"]
    primary_col_indices = coeffs["primary_col_indices"]

    case1_coeff = coeffs["case1_coeff"]
    case1_horner = coeffs["case1_horner"]
    case1_poly_len = coeffs["case1_poly_len"]
    case1_ra_exp = coeffs["case1_ra_exp"]
    case1_rb_exp = coeffs["case1_rb_exp"]
    case1_sign = coeffs["case1_sign"]

    case2_coeff = coeffs["case2_coeff"]
    case2_horner = coeffs["case2_horner"]
    case2_poly_len = coeffs["case2_poly_len"]
    case2_ra_exp = coeffs["case2_ra_exp"]
    case2_rb_exp = coeffs["case2_rb_exp"]
    case2_sign = coeffs["case2_sign"]

    mp_plus_m = coeffs["mp_plus_m"]
    m_minus_mp = coeffs["m_minus_mp"]

    diagonal_mask = coeffs["diagonal_mask"]
    anti_diagonal_mask = coeffs["anti_diagonal_mask"]
    special_2m = coeffs["special_2m"]
    anti_diag_sign = coeffs["anti_diag_sign"]

    derived_row_indices = coeffs["derived_row_indices"]
    derived_col_indices = coeffs["derived_col_indices"]
    derived_primary_idx = coeffs["derived_primary_idx"]
    derived_sign = coeffs["derived_sign"]

    ra = torch.sqrt(ra_re * ra_re + ra_im * ra_im)
    rb = torch.sqrt(rb_re * rb_re + rb_im * rb_im)

    ra_small = ra <= EPSILON
    rb_small = rb <= EPSILON
    general_mask = ~ra_small & ~rb_small
    use_case1 = (ra >= rb) & general_mask
    use_case2 = (ra < rb) & general_mask

    phia = torch.atan2(ra_im, ra_re)
    phib = torch.atan2(rb_im, rb_re)

    phase = torch.outer(phia, mp_plus_m) + torch.outer(phib, m_minus_mp)
    exp_phase_re = torch.cos(phase)
    exp_phase_im = torch.sin(phase)

    safe_ra = torch.clamp(ra, min=EPSILON)
    safe_rb = torch.clamp(rb, min=EPSILON)
    log_ra = torch.log(safe_ra)
    log_rb = torch.log(safe_rb)

    result_re = torch.zeros(N, n_primary, dtype=dtype, device=device)
    result_im = torch.zeros(N, n_primary, dtype=dtype, device=device)

    # Special Case 1: |Ra| ~ 0 - anti-diagonal elements
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

    # Special Case 2: |Rb| ~ 0 - diagonal elements
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
    n_derived = coeffs["n_derived"]
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


# =============================================================================
# Complex to Real Spherical Harmonics Transformation (U blocks)
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

    for m in range(-ell, ell + 1):
        row = m + ell

        if m > 0:
            col_pos = m + ell
            col_neg = -m + ell
            sign = (-1) ** m
            U_ell[row, col_pos] = sign * sqrt2_inv
            U_ell[row, col_neg] = sqrt2_inv
        elif m == 0:
            col = ell
            U_ell[row, col] = 1.0
        else:
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
    lmin: int = 0,
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices for l in [lmin, lmax].

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64) - will use corresponding complex type
        device: Torch device
        lmin: Minimum angular momentum (default 0)

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
    lmin: int = 0,
) -> list[torch.Tensor]:
    """
    Precompute U transformation matrices with Euler basis alignment folded in.

    This combines the complex->real transformation with:
    - For l=1: The Cartesian permutation P (m-ordering -> x,y,z)
    - For l>=2: The Euler-matching basis transformation

    Args:
        lmax: Maximum angular momentum
        dtype: Real dtype (float32 or float64)
        device: Torch device
        lmin: Minimum angular momentum (default 0)

    Returns:
        List of combined U matrices where U_blocks[i] corresponds to l=lmin+i
    """
    U_blocks = precompute_U_blocks(lmax, dtype, device, lmin=lmin)

    if dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        complex_dtype = torch.complex128

    P = torch.tensor([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ], dtype=complex_dtype, device=device)

    jd_path = Path(__file__).parent.parent / "Jd.pt"
    Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

    U_combined = []
    for idx, ell in enumerate(range(lmin, lmax + 1)):
        U_ell = U_blocks[idx].to(dtype=complex_dtype, device=device)

        if ell == 0:
            U_combined.append(U_ell)
        elif ell == 1:
            U_combined.append(P @ U_ell)
        else:
            Jd = Jd_list[ell].to(dtype=dtype, device=device)
            U_euler = _build_euler_transform(ell, Jd).to(dtype=complex_dtype, device=device)
            U_combined.append(U_euler @ U_ell)

    return U_combined


def precompute_U_blocks_euler_aligned_real(
    lmax: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
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
    U_blocks_complex = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
    return [(U.real.to(dtype=dtype), U.imag.to(dtype=dtype)) for U in U_blocks_complex]


# =============================================================================
# Wigner D Complex to Real Transformation
# =============================================================================


def wigner_d_complex_to_real(
    D_complex: torch.Tensor,
    U_blocks: list[torch.Tensor],
    lmax: int,
    lmin: int = 0,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from complex to real, exploiting block structure.

    Args:
        D_complex: Complex Wigner D matrices of shape (N, size, size)
        U_blocks: List of U matrices for l in [lmin, lmax]
        lmax: Maximum angular momentum
        lmin: Minimum angular momentum (default 0)

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

        D_block = D_complex[:, block_start:block_end, block_start:block_end]

        U_ell = U_blocks[idx].to(dtype=complex_dtype, device=device)
        U_ell_H = U_ell.conj().T

        temp = torch.matmul(D_block, U_ell_H)
        D_block_real = torch.matmul(U_ell, temp)

        D_real[:, block_start:block_end, block_start:block_end] = D_block_real.real

        block_start = block_end

    return D_real


def wigner_d_pair_to_real(
    D_re: torch.Tensor,
    D_im: torch.Tensor,
    U_blocks_real: list[tuple[torch.Tensor, torch.Tensor]],
    lmax: int,
    lmin: int = 0,
) -> torch.Tensor:
    """
    Transform Wigner D matrix from real-pair to real basis using real arithmetic.

    This is a torch.compile-compatible alternative to wigner_d_complex_to_real.

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
# Block-wise Matrix Multiplication Utility
# =============================================================================


def bmm_block_diagonal(
    A: torch.Tensor,
    B: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """
    Block-wise matrix multiplication for block-diagonal matrices.

    Both A and B are assumed to be block-diagonal with blocks of sizes
    1, 3, 5, 7, ... (2*l+1 for l=0,1,2,...,lmax).

    Args:
        A: Block-diagonal matrices of shape (N, size, size)
        B: Block-diagonal matrices of shape (N, size, size)
        lmax: Maximum angular momentum

    Returns:
        C = A @ B, block-diagonal matrices of shape (N, size, size)
    """
    N = A.shape[0]
    size = A.shape[1]
    device = A.device
    dtype = A.dtype

    C = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        A_block = A[:, block_start:block_end, block_start:block_end]
        B_block = B[:, block_start:block_end, block_start:block_end]

        C[:, block_start:block_end, block_start:block_end] = torch.bmm(A_block, B_block)

        block_start = block_end

    return C


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

    version = "v1"
    return cache_dir / f"wigner_{variant}_lmax{lmax}_{version}.pt"


def _save_coefficients(coeffs: dict[str, torch.Tensor], path: Path) -> None:
    """Save coefficients to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
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

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for coefficients
        device: Device for tensors
        cache_dir: Directory for cache files
        use_cache: Whether to use disk caching

    Returns:
        Dictionary with precomputed coefficient tensors
    """
    cache_path = _get_cache_path(lmax, "symmetric", cache_dir)

    if use_cache:
        coeffs = _load_coefficients(cache_path, device)
        if coeffs is not None:
            if coeffs.get("lmax") == lmax:
                if dtype != torch.float64:
                    coeffs = {
                        k: v.to(dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                        for k, v in coeffs.items()
                    }
                return coeffs

    coeffs = precompute_wigner_coefficients(lmax, dtype, device)

    if use_cache:
        try:
            _save_coefficients(coeffs, cache_path)
        except Exception:
            pass

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
        try:
            f.unlink()
            count += 1
        except Exception:
            pass

    return count
