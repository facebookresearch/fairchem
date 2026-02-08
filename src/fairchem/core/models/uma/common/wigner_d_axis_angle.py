"""
Wigner D matrices using axis-angle representation and matrix exponentials.

This module provides an alternative implementation of Wigner D matrix computation
that uses SO(3) Lie algebra generators and matrix exponentials instead of the
Ra/Rb polynomial formula used by the quaternion module.

Key features:
- Uses torch.linalg.matrix_exp for stable computation
- Two quaternion charts with SLERP blending: no singularities anywhere
- Avoids the ZYZ Euler angle singularities at ±Y
- Output exactly matches Euler-based code (rotation.py) - drop-in replacement
- Optional Ra/Rb polynomial mode for faster GPU computation

The output uses the same real spherical harmonic basis as the Euler-based
implementation. The Euler-matching basis transformation for l >= 2 is folded
into the SO(3) generators, so no separate transformation step is needed.
This makes axis_angle_wigner() a drop-in replacement for the Euler code.

The implementation:
1. Computes the edge → +Y quaternion using two charts with SLERP blending:
   - Chart 1: singular at -Y (used when ey > -0.7)
   - Chart 2: singular at +Y (used when ey < -0.9)
   - SLERP blend in ey ∈ [-0.9, -0.7] for C-infinity continuity
2. Converts the quaternion to axis-angle representation
3. Computes Wigner D via D^l = exp(θ * (n · K^l)) where K are SO(3) generators
   (with Euler-matching transformation pre-applied for l >= 2)
4. Applies an optional gamma roll correction about the Y-axis

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# Import Ra/Rb functions at module level to avoid per-call import overhead
from fairchem.core.models.uma.common.wigner_d_quaternion import (
    precompute_wigner_coefficients_symmetric,
    precompute_wigner_coefficients_range,
    precompute_U_blocks_euler_aligned,
    precompute_U_blocks_euler_aligned_range,
    quaternion_to_ra_rb,
    wigner_d_matrix_complex,
    wigner_d_matrix_complex_range,
    wigner_d_complex_to_real_blockwise,
    wigner_d_complex_to_real_range,
)


# =============================================================================
# Generator and Transform Caching
# =============================================================================

_GENERATOR_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_RA_RB_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_RA_RB_U_CACHE: dict[tuple[int, torch.dtype, torch.device], list] = {}
_RA_RB_RANGE_CACHE: dict[tuple[int, int, torch.dtype, torch.device], tuple] = {}


def clear_memory_caches() -> None:
    """
    Clear all in-memory caches for Wigner D computation.

    This clears the module-level dictionaries that cache generators,
    transforms, and Ra/Rb coefficients. Useful for testing or reducing memory.
    """
    _GENERATOR_CACHE.clear()
    _RA_RB_COEFF_CACHE.clear()
    _RA_RB_U_CACHE.clear()
    _RA_RB_RANGE_CACHE.clear()


# =============================================================================
# Euler-Matching Basis Transformation
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
    if abs_m % 2 == 0:  # even |m|
        return (-1) ** ((ell - abs_m) // 2)
    else:  # odd |m|
        base = (ell + abs_m + 1) // 2
        if m < 0:
            return (-1) ** base
        else:  # m > 0
            return (-1) ** (base + 1)


def _build_euler_transform(ell: int, Jd: torch.Tensor) -> torch.Tensor:
    """
    Build the basis transformation U for level ell.

    U transforms axis-angle Wigner D to match Euler Wigner D:
        D_euler = U @ D_axis @ U.T

    The transformation is a signed row permutation of Jd[ell]:
    - Permutation: swap m ↔ -m when |m| is odd
    - Signs are computed by _compute_transform_sign

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

        # Permutation: swap m ↔ -m when |m| is odd
        abs_m = abs(m)
        if abs_m % 2 == 1:
            jd_row = (-m) + ell
        else:
            jd_row = i

        # Sign
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
        else:  # m < 0
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

    These are real antisymmetric (2*ell+1) × (2*ell+1) matrices satisfying:
        D^ell(n, θ) = exp(θ * (n_x K_x + n_y K_y + n_z K_z))

    For l=1, the basis is m = (-1, 0, +1) which corresponds to (y, z, x) in Cartesian.

    Args:
        ell: Angular momentum quantum number

    Returns:
        (K_x, K_y, K_z) tuple of generator matrices in float64
    """
    size = 2 * ell + 1

    if ell == 0:
        z = torch.zeros(1, 1, dtype=torch.float64)
        return z, z.clone(), z.clone()

    # Build angular momentum operators in complex basis
    m_values = torch.arange(-ell, ell + 1, dtype=torch.float64)
    J_z = torch.diag(m_values.to(torch.complex128))

    # J_+ (raising) and J_- (lowering) operators
    J_plus = torch.zeros(size, size, dtype=torch.complex128)
    J_minus = torch.zeros(size, size, dtype=torch.complex128)

    for m in range(-ell, ell):
        # J_+ |ell,m> = sqrt(ell(ell+1) - m(m+1)) |ell,m+1>
        coeff = math.sqrt(ell * (ell + 1) - m * (m + 1))
        J_plus[m + 1 + ell, m + ell] = coeff

    for m in range(-ell + 1, ell + 1):
        # J_- |ell,m> = sqrt(ell(ell+1) - m(m-1)) |ell,m-1>
        coeff = math.sqrt(ell * (ell + 1) - m * (m - 1))
        J_minus[m - 1 + ell, m + ell] = coeff

    # J_x = (J_+ + J_-) / 2, J_y = (J_+ - J_-) / 2i
    J_x = (J_plus + J_minus) / 2
    J_y = (J_plus - J_minus) / 2j

    # Transform to real spherical harmonic basis
    U = _build_u_matrix(ell)
    U_dag = U.conj().T

    # The SO(3) generator in real basis
    K_x = (U @ (1j * J_x) @ U_dag).real
    K_y = -(U @ (1j * J_y) @ U_dag).real  # Negated for correct commutation relations
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
    so the matrix exponential (or Rodrigues/Cayley-Hamilton) produces output directly
    in the Euler basis.

    For l=1, a permutation matrix P is also cached to convert to Cartesian basis
    when needed.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for the generators
        device: Device for the generators

    Returns:
        Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P' for l=1 permutation
    """
    from pathlib import Path

    key = (lmax, dtype, device)

    if key not in _GENERATOR_CACHE:
        # Load Jd matrices for Euler transforms (l >= 2)
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
                # Apply Euler-matching transformation: K_euler = U @ K @ U.T
                Jd = Jd_list[ell].to(dtype=dtype, device=device)
                U = _build_euler_transform(ell, Jd)
                K_x = U @ K_x @ U.T
                K_y = U @ K_y @ U.T
                K_z = U @ K_z @ U.T

            K_x_list.append(K_x)
            K_y_list.append(K_y)
            K_z_list.append(K_z)

        # Permutation matrix for l=1: (y,z,x) -> (x,y,z)
        P = torch.tensor([
            [0., 0., 1.],  # x from position 2
            [1., 0., 0.],  # y from position 0
            [0., 1., 0.]   # z from position 1
        ], dtype=dtype, device=device)

        _GENERATOR_CACHE[key] = {
            'K_x': K_x_list,
            'K_y': K_y_list,
            'K_z': K_z_list,
            'P': P,
        }

    return _GENERATOR_CACHE[key]


# =============================================================================
# Quaternion Edge → +Y (Two Charts with SLERP Blending)
# =============================================================================


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


def _quaternion_chart1_standard(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Standard quaternion: edge → +Y directly. Singular at edge = -Y.

    Uses the half-vector formula:
        q = normalize(1 + ey, -ez, 0, ex)

    This is derived from the quaternion for rotating v1 to v2:
        q = normalize(|v1||v2| + v1·v2, v1 × v2)

    With v1 = edge = (ex, ey, ez) and v2 = (0, 1, 0):
        v1·v2 = ey
        v1 × v2 = (-ez, 0, ex)

    Magnitude: |q|² = 2(1+ey), so norm → 0 as ey → -1

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

    # Handle singularity at -Y: norm ≈ 0 when ey ≈ -1
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    safe_norm = norm.clamp(min=1e-12)

    return q / safe_norm


def _quaternion_chart2_via_minus_y(
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    """
    Alternative quaternion: edge → +Y via -Y. Singular at edge = +Y.

    Path: edge → -Y → +Y (compose with 180° about X)

    Derivation:
    - q_{edge→-Y} = normalize(1-ey, ez, 0, -ex)  (standard formula with target=-Y)
    - q_{-Y→+Y} = (0, 1, 0, 0)  (180° rotation about X)
    - q_{edge→+Y} = q_{-Y→+Y} ⊗ q_{edge→-Y}

    Quaternion multiplication (0, 1, 0, 0) ⊗ (w, x, y, z):
        w' = 0*w - 1*x - 0*y - 0*z = -x
        x' = 0*x + 1*w + 0*z - 0*y = w
        y' = 0*y + 1*z + 0*w - 0*x = z
        z' = 0*z - 1*y + 0*x + 0*w = -y

    With q_{edge→-Y} = normalize(1-ey, ez, 0, -ex):
        q_{composed} = normalize(-ez, 1-ey, -ex, 0)

    But we can negate (same rotation): normalize(-ez, 1-ey, ex, 0)

    Formula: q = normalize(-ez, 1-ey, ex, 0)
    Magnitude: |q|² = 2(1-ey), so norm → 0 as ey → +1

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

    # Handle singularity at +Y: norm ≈ 0 when ey ≈ 1
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    safe_norm = norm.clamp(min=1e-12)

    return q / safe_norm


def quaternion_slerp(
    q1: torch.Tensor,
    q2: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Spherical linear interpolation between quaternions.

    slerp(q1, q2, t) = (sin((1-t)θ) * q1 + sin(tθ) * q2) / sin(θ)
    where θ = arccos(|q1 · q2|)

    Args:
        q1: First quaternion, shape (..., 4)
        q2: Second quaternion, shape (..., 4)
        t: Interpolation parameter, shape (...)

    Returns:
        Interpolated quaternion, shape (..., 4)
    """
    # Ensure shortest path by aligning q1 to q2 (negate q1 if dot < 0)
    # This ensures that at t=1, we get exactly q2 (not -q2)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q1_aligned = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)  # Clamp for numerical stability

    # Compute angle between quaternions
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # SLERP weights
    t_expanded = t.unsqueeze(-1) if t.dim() < q1.dim() else t
    w1 = torch.sin((1.0 - t_expanded) * theta) / sin_theta.clamp(min=1e-8)
    w2 = torch.sin(t_expanded * theta) / sin_theta.clamp(min=1e-8)

    result = w1 * q1_aligned + w2 * q2

    # Fall back to NLERP for small angles (theta ≈ 0, quaternions nearly equal)
    small_angle = sin_theta.squeeze(-1) < 1e-6
    result_nlerp = torch.nn.functional.normalize(
        (1.0 - t_expanded) * q1_aligned + t_expanded * q2, dim=-1
    )

    return torch.where(small_angle.unsqueeze(-1), result_nlerp, result)


def quaternion_edge_to_y_stable(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion for edge → +Y using two charts with SLERP blending.

    Uses two quaternion charts to avoid singularities:
    - Chart 1: q = normalize(1+ey, -ez, 0, ex) - singular at -Y
    - Chart 2: q = normalize(-ez, 1-ey, ex, 0) - singular at +Y

    SLERP blend in ey ∈ [-0.9, -0.7]:
    - Uses Chart 2 when near -Y (stable there)
    - Uses Chart 1 elsewhere (stable away from -Y)

    This ensures numerically stable computation for all edge directions
    with C-infinity smooth blending between charts.

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    # Chart 1: singular at -Y, stable at +Y
    q_chart1 = _quaternion_chart1_standard(ex, ey, ez)

    # Chart 2: singular at +Y, stable at -Y
    q_chart2 = _quaternion_chart2_via_minus_y(ex, ey, ez)

    # Blend region: ey ∈ [-0.9, -0.7]
    # t=0 at ey=-0.9 (use Chart 2), t=1 at ey=-0.7 (use Chart 1)
    blend_start = -0.9
    blend_width = 0.2
    t = (ey - blend_start) / blend_width
    t_smooth = _smooth_step_cinf(t)

    # SLERP: interpolate from Chart 2 (t=0) to Chart 1 (t=1)
    q = quaternion_slerp(q_chart2, q_chart1, t_smooth)

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

    For small angles (|xyz| ≈ 0), axis is undefined but angle ≈ 0.

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

    # Safe axis computation
    safe_xyz_norm = xyz_norm.clamp(min=1e-12)
    axis = xyz / safe_xyz_norm.unsqueeze(-1)

    # For small angles, use Z-axis as default (arbitrary but consistent)
    small_angle = xyz_norm < 1e-8
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
    z_axis = z_axis.expand_as(axis)
    axis = torch.where(small_angle.unsqueeze(-1), z_axis, axis)

    return axis, angle


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion directly to 3x3 rotation matrix (l=1 Wigner D).

    This is faster than going through axis-angle + Rodrigues because it
    avoids: atan2, sqrt for normalization, sin/cos, K matrix construction.

    Uses the standard quaternion to rotation matrix formula:
        R = [[1-2(y²+z²),  2(xy-wz),   2(xz+wy) ],
             [2(xy+wz),    1-2(x²+z²), 2(yz-wx) ],
             [2(xz-wy),    2(yz+wx),   1-2(x²+y²)]]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute products (each used multiple times)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build rotation matrix
    R = torch.stack([
        torch.stack([1 - 2*(y2 + z2), 2*(xy - wz),     2*(xz + wy)    ], dim=-1),
        torch.stack([2*(xy + wz),     1 - 2*(x2 + z2), 2*(yz - wx)    ], dim=-1),
        torch.stack([2*(xz - wy),     2*(yz + wx),     1 - 2*(x2 + y2)], dim=-1),
    ], dim=-2)

    return R


def quaternion_to_wigner_d_l2(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion directly to 5x5 l=2 Wigner D matrix.

    Uses degree-4 polynomial formulas in quaternion components (w,x,y,z),
    avoiding the Cayley-Hamilton overhead (bmm, sqrt, torch.where, sin/cos).
    This provides faster backward pass since polynomials have simple gradients.

    Output matches the Euler-aligned basis of _cayley_hamilton_exp_l2.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 5, 5) for l=2
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    dtype = q.dtype
    device = q.device

    # Precompute powers
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    w3, x3, y3, z3 = w2 * w, x2 * x, y2 * y, z2 * z
    w4, x4, y4, z4 = w2 * w2, x2 * x2, y2 * y2, z2 * z2

    # Precompute mixed terms (degree 4)
    w2x2 = w2 * x2
    w2y2 = w2 * y2
    w2z2 = w2 * z2
    x2y2 = x2 * y2
    x2z2 = x2 * z2
    y2z2 = y2 * z2

    # Precompute degree-3 terms with single factors
    w3x, w3y, w3z = w3 * x, w3 * y, w3 * z
    wx3, wy3, wz3 = w * x3, w * y3, w * z3
    x3y, x3z = x3 * y, x3 * z
    xy3, xz3 = x * y3, x * z3
    y3z, yz3 = y3 * z, y * z3

    # Precompute degree-2 mixed terms for degree-4 products
    w2xy, w2xz, w2yz = w2 * x * y, w2 * x * z, w2 * y * z
    wx2y, wx2z = w * x2 * y, w * x2 * z
    wxy2, wxz2 = w * x * y2, w * x * z2
    wy2z, wyz2 = w * y2 * z, w * y * z2
    x2yz, xy2z, xyz2 = x2 * y * z, x * y2 * z, x * y * z2
    wxyz = w * x * y * z

    sqrt3 = 1.7320508075688772  # math.sqrt(3)

    # Build the 5x5 Wigner D matrix using derived polynomial formulas
    D = torch.zeros(N, 5, 5, dtype=dtype, device=device)

    # D[0,0] = w4 + y4 - x4 - z4 - 6*w2y2 + 6*x2z2
    D[:, 0, 0] = w4 + y4 - x4 - z4 - 6 * w2y2 + 6 * x2z2

    # D[0,1] = 2*w3x + 2*wx3 - 2*y3z - 2*yz3 - 6*wxy2 - 6*wxz2 + 6*w2yz + 6*x2yz
    D[:, 0, 1] = 2 * w3x + 2 * wx3 - 2 * y3z - 2 * yz3 - 6 * wxy2 - 6 * wxz2 + 6 * w2yz + 6 * x2yz

    # D[0,2] = 4*sqrt3 * (xy2z + wx2y - wyz2 - w2xz)
    D[:, 0, 2] = 4 * sqrt3 * (xy2z + wx2y - wyz2 - w2xz)

    # D[0,3] = -2*w3z - 2*wz3 - 2*x3y - 2*xy3 + 6*w2xy + 6*wx2z + 6*wy2z + 6*xyz2
    D[:, 0, 3] = -2 * w3z - 2 * wz3 - 2 * x3y - 2 * xy3 + 6 * w2xy + 6 * wx2z + 6 * wy2z + 6 * xyz2

    # D[0,4] = 4 * (w3y - wy3 - x3z + xz3)
    D[:, 0, 4] = 4 * (w3y - wy3 - x3z + xz3)

    # D[1,0] = -2*w3x - 2*wx3 - 2*y3z - 2*yz3 + 6*wxy2 + 6*wxz2 + 6*w2yz + 6*x2yz
    D[:, 1, 0] = -2 * w3x - 2 * wx3 - 2 * y3z - 2 * yz3 + 6 * wxy2 + 6 * wxz2 + 6 * w2yz + 6 * x2yz

    # D[1,1] = w4 - y4 - x4 + z4 + 6*x2y2 - 6*w2z2
    D[:, 1, 1] = w4 - y4 - x4 + z4 + 6 * x2y2 - 6 * w2z2

    # D[1,2] = 2*sqrt3 * (-w3z + wz3 - x3y + xy3 + w2xy + wx2z - wy2z - xyz2)
    D[:, 1, 2] = 2 * sqrt3 * (-w3z + wz3 - x3y + xy3 + w2xy + wx2z - wy2z - xyz2)

    # D[1,3] = 2*w3y + 2*wy3 - 2*x3z - 2*xz3 + 6*w2xz - 6*wx2y + 6*xy2z - 6*wyz2
    D[:, 1, 3] = 2 * w3y + 2 * wy3 - 2 * x3z - 2 * xz3 + 6 * w2xz - 6 * wx2y + 6 * xy2z - 6 * wyz2

    # D[1,4] = -2*w3z + 2*wz3 - 2*x3y + 2*xy3 - 6*w2xy - 6*wx2z + 6*wy2z + 6*xyz2
    D[:, 1, 4] = -2 * w3z + 2 * wz3 - 2 * x3y + 2 * xy3 - 6 * w2xy - 6 * wx2z + 6 * wy2z + 6 * xyz2

    # D[2,0] = 4*sqrt3 * (xy2z - wx2y + wyz2 - w2xz)
    D[:, 2, 0] = 4 * sqrt3 * (xy2z - wx2y + wyz2 - w2xz)

    # D[2,1] = 2*sqrt3 * (w3z - wz3 - x3y + xy3 + w2xy - wx2z + wy2z - xyz2)
    D[:, 2, 1] = 2 * sqrt3 * (w3z - wz3 - x3y + xy3 + w2xy - wx2z + wy2z - xyz2)

    # D[2,2] = w4 + x4 + y4 + z4 - 4*w2x2 + 2*w2y2 - 4*w2z2 - 4*x2y2 + 2*x2z2 - 4*y2z2
    D[:, 2, 2] = w4 + x4 + y4 + z4 - 4 * w2x2 + 2 * w2y2 - 4 * w2z2 - 4 * x2y2 + 2 * x2z2 - 4 * y2z2

    # D[2,3] = 2*sqrt3 * (-w3x + wx3 + y3z - yz3 - x2yz + w2yz - wxy2 + wxz2)
    D[:, 2, 3] = 2 * sqrt3 * (-w3x + wx3 + y3z - yz3 - x2yz + w2yz - wxy2 + wxz2)

    # D[2,4] = 2*sqrt3 * (w2x2 - w2z2 - x2y2 + y2z2 - 4*wxyz)
    D[:, 2, 4] = 2 * sqrt3 * (w2x2 - w2z2 - x2y2 + y2z2 - 4 * wxyz)

    # D[3,0] = 2*w3z + 2*wz3 - 2*x3y - 2*xy3 + 6*w2xy - 6*wx2z - 6*wy2z + 6*xyz2
    D[:, 3, 0] = 2 * w3z + 2 * wz3 - 2 * x3y - 2 * xy3 + 6 * w2xy - 6 * wx2z - 6 * wy2z + 6 * xyz2

    # D[3,1] = -2*w3y - 2*wy3 - 2*x3z - 2*xz3 + 6*w2xz + 6*wx2y + 6*xy2z + 6*wyz2
    D[:, 3, 1] = -2 * w3y - 2 * wy3 - 2 * x3z - 2 * xz3 + 6 * w2xz + 6 * wx2y + 6 * xy2z + 6 * wyz2

    # D[3,2] = 2*sqrt3 * (w3x - wx3 + y3z - yz3 - x2yz + w2yz + wxy2 - wxz2)
    D[:, 3, 2] = 2 * sqrt3 * (w3x - wx3 + y3z - yz3 - x2yz + w2yz + wxy2 - wxz2)

    # D[3,3] = w4 + x4 - y4 - z4 - 6*w2x2 + 6*y2z2
    D[:, 3, 3] = w4 + x4 - y4 - z4 - 6 * w2x2 + 6 * y2z2

    # D[3,4] = -2*w3x + 2*wx3 - 2*y3z + 2*yz3 + 6*wxy2 - 6*wxz2 + 6*w2yz - 6*x2yz
    D[:, 3, 4] = -2 * w3x + 2 * wx3 - 2 * y3z + 2 * yz3 + 6 * wxy2 - 6 * wxz2 + 6 * w2yz - 6 * x2yz

    # D[4,0] = 4 * (-w3y + wy3 - x3z + xz3)
    D[:, 4, 0] = 4 * (-w3y + wy3 - x3z + xz3)

    # D[4,1] = 2*w3z - 2*wz3 - 2*x3y + 2*xy3 - 6*w2xy - 6*wy2z + 6*wx2z + 6*xyz2
    D[:, 4, 1] = 2 * w3z - 2 * wz3 - 2 * x3y + 2 * xy3 - 6 * w2xy - 6 * wy2z + 6 * wx2z + 6 * xyz2

    # D[4,2] = 2*sqrt3 * (w2x2 - w2z2 - x2y2 + y2z2 + 4*wxyz)
    D[:, 4, 2] = 2 * sqrt3 * (w2x2 - w2z2 - x2y2 + y2z2 + 4 * wxyz)

    # D[4,3] = 2*w3x - 2*wx3 - 2*y3z + 2*yz3 + 6*wxz2 - 6*wxy2 + 6*w2yz - 6*x2yz
    D[:, 4, 3] = 2 * w3x - 2 * wx3 - 2 * y3z + 2 * yz3 + 6 * wxz2 - 6 * wxy2 + 6 * w2yz - 6 * x2yz

    # D[4,4] = w4 + x4 + y4 + z4 - 6*w2y2 - 6*x2z2
    D[:, 4, 4] = w4 + x4 + y4 + z4 - 6 * w2y2 - 6 * x2z2

    return D


# =============================================================================
# Cayley-Hamilton for l=2 (5x5 antisymmetric matrices)
# =============================================================================


def _cayley_hamilton_exp_l2(K: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(θK) for 5×5 antisymmetric K using Cayley-Hamilton.

    For antisymmetric K with eigenvalues 0, ±iλ₁, ±iλ₂:
        exp(θK) = I + c₁K + c₂K² + c₃K³ + c₄K⁴

    The coefficients are derived via Lagrange interpolation on the eigenvalues.
    This is ~1.4x faster than matrix_exp for l=2 blocks.

    Args:
        K: Antisymmetric matrices of shape (N, 5, 5)
        angle: Rotation angles of shape (N,)

    Returns:
        Rotation matrices exp(θK) of shape (N, 5, 5)
    """
    device = K.device
    dtype = K.dtype

    I = torch.eye(5, device=device, dtype=dtype).unsqueeze(0)

    # Compute powers of K
    K2 = torch.bmm(K, K)
    K3 = torch.bmm(K2, K)
    K4 = torch.bmm(K2, K2)

    # Extract eigenvalue info from traces
    # For antisymmetric K: eigenvalues are 0, ±iλ₁, ±iλ₂
    # tr(K²) = -2(λ₁² + λ₂²)
    # tr(K⁴) = 2(λ₁⁴ + λ₂⁴)
    tr_K2 = torch.einsum('nii->n', K2)
    tr_K4 = torch.einsum('nii->n', K4)

    s1 = -tr_K2 / 2  # λ₁² + λ₂²
    s2 = tr_K4 / 2   # λ₁⁴ + λ₂⁴

    # Solve for σ₁ = λ₁², σ₂ = λ₂²
    # Product p = σ₁ * σ₂ = (s1² - s2) / 2
    p = (s1 * s1 - s2) / 2

    # σ₁, σ₂ are roots of t² - s1*t + p = 0
    discriminant = (s1 * s1 - 4 * p).clamp(min=0)
    sqrt_disc = torch.sqrt(discriminant)

    sigma1 = ((s1 + sqrt_disc) / 2).clamp(min=0)
    sigma2 = ((s1 - sqrt_disc) / 2).clamp(min=0)

    lambda1 = torch.sqrt(sigma1)
    lambda2 = torch.sqrt(sigma2)

    theta = angle
    eps = 1e-12

    # Compute sinc-like functions:
    # A = (cos(θλ₁) - 1) / σ₁
    # B = (cos(θλ₂) - 1) / σ₂
    # C = sin(θλ₁) / λ₁
    # D = sin(θλ₂) / λ₂

    theta_l1 = theta * lambda1
    theta_l2 = theta * lambda2

    cos_l1 = torch.cos(theta_l1)
    cos_l2 = torch.cos(theta_l2)
    sin_l1 = torch.sin(theta_l1)
    sin_l2 = torch.sin(theta_l2)

    # Safe division for A = (cos(θλ) - 1) / σ
    sigma1_safe = torch.where(sigma1 < eps, torch.ones_like(sigma1), sigma1)
    sigma2_safe = torch.where(sigma2 < eps, torch.ones_like(sigma2), sigma2)
    A = torch.where(sigma1 < eps, -theta * theta / 2, (cos_l1 - 1) / sigma1_safe)
    B = torch.where(sigma2 < eps, -theta * theta / 2, (cos_l2 - 1) / sigma2_safe)

    # Safe division for C = sin(θλ) / λ
    lambda1_safe = torch.where(lambda1 < eps, torch.ones_like(lambda1), lambda1)
    lambda2_safe = torch.where(lambda2 < eps, torch.ones_like(lambda2), lambda2)
    C = torch.where(lambda1 < eps, theta, sin_l1 / lambda1_safe)
    D = torch.where(lambda2 < eps, theta, sin_l2 / lambda2_safe)

    # Coefficients via Lagrange interpolation:
    #   c₁ = (σ₂C - σ₁D) / (σ₂ - σ₁)
    #   c₂ = (σ₁B - σ₂A) / (σ₂ - σ₁)
    #   c₃ = (C - D) / (σ₂ - σ₁)
    #   c₄ = (B - A) / (σ₂ - σ₁)

    sigma_diff = sigma2 - sigma1
    degenerate = sigma_diff.abs() < eps
    sigma_diff_safe = torch.where(degenerate, torch.ones_like(sigma_diff), sigma_diff)

    # Non-degenerate case
    c1_nondeg = (sigma2 * C - sigma1 * D) / sigma_diff_safe
    c2_nondeg = (sigma1 * B - sigma2 * A) / sigma_diff_safe
    c3_nondeg = (C - D) / sigma_diff_safe
    c4_nondeg = (B - A) / sigma_diff_safe

    # Degenerate case (σ₁ = σ₂ = σ): Rodrigues-like formula
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

    # exp(θK) = I + c₁K + c₂K² + c₃K³ + c₄K⁴
    return I + c1 * K + c2 * K2 + c3 * K3 + c4 * K4


# =============================================================================
# Ra/Rb Polynomial-based Wigner D (GPU-optimized)
# =============================================================================


def _get_ra_rb_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached Ra/Rb polynomial coefficients with Euler-aligned U blocks."""
    key = (lmax, dtype, device)

    if key not in _RA_RB_COEFF_CACHE:
        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)
        _RA_RB_COEFF_CACHE[key] = coeffs

    if key not in _RA_RB_U_CACHE:
        U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
        _RA_RB_U_CACHE[key] = U_blocks

    return _RA_RB_COEFF_CACHE[key], _RA_RB_U_CACHE[key]


def _get_ra_rb_coefficients_range(
    lmin: int,
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, list]:
    """Get cached Ra/Rb polynomial coefficients for l in [lmin, lmax] only."""
    key = (lmin, lmax, dtype, device)

    if key not in _RA_RB_RANGE_CACHE:
        coeffs = precompute_wigner_coefficients_range(lmin, lmax, dtype=dtype, device=device)
        U_blocks = precompute_U_blocks_euler_aligned_range(lmin, lmax, dtype=dtype, device=device)
        _RA_RB_RANGE_CACHE[key] = (coeffs, U_blocks)

    return _RA_RB_RANGE_CACHE[key]


def wigner_d_from_quaternion_polynomial(
    q: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from quaternions using Ra/Rb polynomial.

    This is faster than matrix_exp on GPU, especially for higher lmax.
    Uses the same algorithm as wigner_d_quaternion.py but takes the quaternion
    directly (computed by axis_angle's SLERP-blended two-chart approach).

    Output is in Euler-aligned basis with l=1 Cartesian permutation and l>=2
    Euler basis transformation folded into the U blocks.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum

    Returns:
        Real Wigner D matrices of shape (N, size, size)
    """
    from fairchem.core.models.uma.common.wigner_d_quaternion import (
        quaternion_to_ra_rb,
        wigner_d_matrix_complex,
        wigner_d_complex_to_real_blockwise,
    )

    dtype = q.dtype
    device = q.device

    coeffs, U_blocks = _get_ra_rb_coefficients(lmax, dtype, device)

    Ra, Rb = quaternion_to_ra_rb(q)
    D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)
    D_real = wigner_d_complex_to_real_blockwise(D_complex, U_blocks, lmax)

    return D_real


def axis_angle_wigner_polynomial(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using Ra/Rb polynomial (GPU-optimized version).

    This is a GPU-optimized alternative to axis_angle_wigner that uses
    the Ra/Rb polynomial formula instead of matrix_exp. It's ~1.5-2x faster
    on GPU for lmax >= 4.

    Uses the same SLERP-blended two-chart quaternion approach as axis_angle_wigner
    to handle singularities correctly. Combines edge→Y and gamma rotations into
    a single quaternion before computing the Wigner D.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype
    N = edge_distance_vec.shape[0]

    # Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Compute gamma if not provided
    if gamma is None:
        gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Compute quaternion (edge → +Y) using SLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Create Y-rotation quaternion and combine with edge→Y
    # Combined rotation: first edge→Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Compute Wigner D using Ra/Rb polynomial
    D = wigner_d_from_quaternion_polynomial(q_combined, lmax)

    # Return D and its inverse (transpose for orthogonal matrices)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


# =============================================================================
# Wigner D from Axis-Angle (Batched)
# =============================================================================


def wigner_d_from_axis_angle_batched(
    axis: torch.Tensor,
    angle: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from axis-angle representation.

    D^l = exp(angle * (axis · K^l)) for each l block.
    The l=1 block is transformed to Cartesian basis (x,y,z) for compatibility.

    Uses optimizations:
    - l=0: Trivial (identity)
    - l=1: Rodrigues formula (4-5x faster than matrix_exp)
    - l>=2: matrix_exp

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)²
    """
    N = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    size = (lmax + 1) ** 2

    K_x_list = generators['K_x']
    K_y_list = generators['K_y']
    K_z_list = generators['K_z']
    P = generators['P']

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        if ell == 0:
            # l=0 is trivial: 1x1 identity
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # l=1: Use Rodrigues formula for ~5x speedup
            # exp(θK) = I + sin(θ)K + (1-cos(θ))K² for 3x3 rotation
            K_x = K_x_list[1]
            K_y = K_y_list[1]
            K_z = K_z_list[1]

            K = (
                axis[:, 0:1, None, None] * K_x +
                axis[:, 1:2, None, None] * K_y +
                axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            I = torch.eye(3, dtype=dtype, device=device)
            sin_t = torch.sin(angle)[:, None, None]
            cos_t = torch.cos(angle)[:, None, None]
            K2 = torch.bmm(K, K)
            D_ell = I + sin_t * K + (1 - cos_t) * K2

            # Transform from m-ordering (y,z,x) to Cartesian (x,y,z) via index reordering
            # P transforms [y,z,x] -> [x,y,z], which is index permutation [2,0,1]
            D[:, 1:4, 1:4] = D_ell[:, [2, 0, 1], :][:, :, [2, 0, 1]]
        else:
            # l>=2: Use Cayley-Hamilton for l=2, matrix_exp for l>=3
            # Note: Cayley-Hamilton is ~1.4x faster for l=2 but slower for l>=5
            K_x = K_x_list[ell]
            K_y = K_y_list[ell]
            K_z = K_z_list[ell]

            K = (
                axis[:, 0:1, None, None] * K_x +
                axis[:, 1:2, None, None] * K_y +
                axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            if ell == 2:
                # Use Cayley-Hamilton formula for l=2 (5x5 matrices)
                D_ell = _cayley_hamilton_exp_l2(K, angle)
            else:
                # Use matrix_exp for l>=3
                D_ell = torch.linalg.matrix_exp(angle[:, None, None] * K)
            D[:, block_start:block_end, block_start:block_end] = D_ell

        block_start = block_end

    return D


def wigner_d_from_axis_angle_hybrid(
    axis: torch.Tensor,
    angle: torch.Tensor,
    q: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomials (faster backward pass)
    - l>=3: Ra/Rb polynomial from quaternion (faster than matrix_exp on GPU)

    The caller should pass Euler-aligned generators for l>=2.

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)²
    """
    N = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    size = (lmax + 1) ** 2

    K_x_list = generators['K_x']
    K_y_list = generators['K_y']
    K_z_list = generators['K_z']
    P = generators['P']

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    # Compute l=0, l=1, l=2 using axis-angle (Rodrigues + Cayley-Hamilton)
    block_start = 0
    for ell in range(min(lmax + 1, 3)):  # Only l=0,1,2
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        if ell == 0:
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # Direct quaternion to rotation matrix (faster than axis-angle + Rodrigues)
            # This avoids: atan2, sqrt, sin/cos, K matrix construction, bmm
            # The result is already in Cartesian (x,y,z) basis - no permutation needed
            D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
        elif ell == 2:
            # Direct quaternion to Wigner D l=2 (faster backward than Cayley-Hamilton)
            # Uses degree-4 polynomial formulas, avoiding bmm/sqrt/where overhead
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2(q)

        block_start = block_end

    # Compute l>=3 using Ra/Rb polynomial from quaternion (range version)
    if lmax >= 3:
        # Get Ra/Rb coefficients for l>=3 only (more efficient)
        coeffs_range, U_blocks_range = _get_ra_rb_coefficients_range(3, lmax, dtype, device)
        Ra, Rb = quaternion_to_ra_rb(q)
        D_complex_range = wigner_d_matrix_complex_range(Ra, Rb, coeffs_range)
        D_ra_rb_range = wigner_d_complex_to_real_range(D_complex_range, U_blocks_range, 3, lmax)

        # Copy l>=3 blocks directly from the range result
        # D_ra_rb_range is already just the l>=3 blocks
        block_offset = 9  # Skip l=0,1,2 in full matrix (1 + 3 + 5 = 9)
        D[:, block_offset:, block_offset:] = D_ra_rb_range

    return D


# =============================================================================
# Gamma Computation for Euler Matching
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


def compute_euler_matching_gamma(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute gamma to match the Euler convention.

    gamma = -atan2(ex, ez) is the roll correction that aligns the
    Rodrigues rotation with the ZYZ Euler decomposition.

    For edges on Y-axis (ex = ez ≈ 0): gamma = 0 (degenerate case).

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Gamma angles of shape (N,)
    """
    ex = edge_vec[..., 0]
    ez = edge_vec[..., 2]

    gamma = -torch.atan2(ex, ez)

    return gamma


def bmm_block_diagonal(
    A: torch.Tensor,
    B: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """
    Block-wise matrix multiplication for block-diagonal matrices.

    Both A and B are assumed to be block-diagonal with blocks of sizes
    1, 3, 5, 7, ... (2*l+1 for l=0,1,2,...,lmax).

    This is much faster than full bmm because:
    - Full bmm: O(N * size³) where size = (lmax+1)²
    - Block-wise: O(N * sum((2*l+1)³)) which is much smaller

    For lmax=6: full = 49³ = 117,649 vs block = 4,753 (25x fewer ops)

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
# Main Entry Point
# =============================================================================


def axis_angle_wigner(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D from edge vectors using axis-angle representation.

    This approach uses two quaternion charts with SLERP blending to map edge → +Y,
    which eliminates all singularities (the two charts have complementary singular
    points at +Y and -Y respectively). This avoids both the single-chart Rodrigues
    singularity at -Y and the ZYZ Euler angle singularities at ±Y.

    The output uses the same real spherical harmonic basis as the Euler-based
    implementation (rotation.py), making this a drop-in replacement.

    Combines edge→Y and gamma rotations into a single quaternion before computing
    the Wigner D, avoiding the overhead of computing two separate Wigner D matrices.

    Uses Euler-aligned generators that have the Euler basis transformation folded in,
    so the matrix exponential output is directly in Euler basis.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.
               Note: this introduces gradient singularity at edge = +Y.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
        - wigner_edge_to_y: the edge → +Y rotation (D @ edge = +Y for l=1)
        - wigner_y_to_edge: the +Y → edge rotation (D @ +Y = edge for l=1)
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        if use_euler_gamma:
            # Use atan2-based gamma to exactly match Euler code output
            # Note: has gradient singularity at edge = +Y
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            # Random gamma for SO(2) equivariance (default for training)
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge → +Y) using SLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge→Y
    # Combined rotation: first edge→Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Extract axis-angle from combined quaternion
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached)
    # These have the Euler transform folded in for l >= 2
    generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute single Wigner D from combined rotation via matrix_exp
    # Output is directly in Euler basis thanks to Euler-aligned generators
    D = wigner_d_from_axis_angle_batched(axis, angle, generators, lmax)

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


def axis_angle_wigner_hybrid(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D using hybrid approach (Rodrigues/Cayley-Hamilton + Ra/Rb).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Cayley-Hamilton from axis-angle with Euler-aligned generators
    - l>=3: Ra/Rb polynomial from quaternion with Euler-aligned U blocks

    Combines the edge→Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Uses Euler-aligned generators (l=2) and U blocks (l>=3) that have the
    Euler basis transformation folded in, eliminating the separate transform step.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)².
    """
    # Handle single vector input
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)

    N = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Compute gamma if not provided
    if gamma is None:
        if use_euler_gamma:
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 3: Compute quaternion (edge → +Y)
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge→Y
    # Combined rotation: first edge→Y, then rotate about Y by gamma
    # q_combined = q_gamma * q_edge_to_y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Extract axis-angle from combined quaternion (needed for l=2)
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached)
    # These have the Euler transform folded in for l=2
    generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute Wigner D using hybrid approach
    D = wigner_d_from_axis_angle_hybrid(axis, angle, q_combined, generators, lmax)

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv

