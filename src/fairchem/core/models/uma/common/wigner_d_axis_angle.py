"""
Wigner D matrices using axis-angle representation and matrix exponentials.

This module provides an alternative implementation of Wigner D matrix computation
that uses SO(3) Lie algebra generators and matrix exponentials instead of the
Ra/Rb polynomial formula used by the quaternion module.

Key features:
- Uses torch.linalg.matrix_exp for stable computation
- Only has a singularity at edge = -Y (180° rotation ambiguity)
- Avoids the ZYZ Euler angle singularities at ±Y
- Output exactly matches Euler-based code (rotation.py) - drop-in replacement

The output uses the same real spherical harmonic basis as the Euler-based
implementation, achieved via an automatic basis transformation for l >= 2.
This makes axis_angle_wigner() a drop-in replacement for the Euler code.

The implementation:
1. Computes the Rodrigues (minimal-arc) quaternion for the edge → +Y rotation
2. Converts the quaternion to axis-angle representation
3. Computes Wigner D via D^l = exp(θ * (n · K^l)) where K are SO(3) generators
4. Applies an optional gamma roll correction about the Y-axis
5. Applies Euler-matching basis transformation for l >= 2

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# =============================================================================
# Generator and Transform Caching
# =============================================================================

_GENERATOR_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}
_EULER_TRANSFORM_CACHE: dict[tuple[int, torch.dtype, torch.device], list[torch.Tensor]] = {}


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


def get_euler_transforms(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Get cached Euler-matching transformation matrices for l=0..lmax.

    These matrices transform axis-angle Wigner D to match Euler Wigner D:
        D_euler[l] = U[l] @ D_axis[l] @ U[l].T  for l >= 2

    For l=0 and l=1, returns identity matrices (no transformation needed).

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for the matrices
        device: Device for the matrices

    Returns:
        List of transformation matrices [U_0, U_1, ..., U_lmax]
    """
    key = (lmax, dtype, device)

    if key not in _EULER_TRANSFORM_CACHE:
        from pathlib import Path

        # Load Jd matrices
        jd_path = Path(__file__).parent.parent / "Jd.pt"
        Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

        U_list = []
        for ell in range(lmax + 1):
            if ell <= 1:
                # l=0 and l=1 match directly, use identity
                size = 2 * ell + 1
                U = torch.eye(size, dtype=dtype, device=device)
            else:
                Jd = Jd_list[ell].to(dtype=dtype, device=device)
                U = _build_euler_transform(ell, Jd)
            U_list.append(U)

        _EULER_TRANSFORM_CACHE[key] = U_list

    return _EULER_TRANSFORM_CACHE[key]


def apply_euler_transform(
    D: torch.Tensor,
    lmax: int,
    U_list: list[torch.Tensor],
) -> torch.Tensor:
    """
    Apply Euler-matching transformation to a block-diagonal Wigner D matrix.

    Transforms each l-block: D_euler[l] = U[l] @ D_axis[l] @ U[l].T

    Args:
        D: Wigner D matrix of shape (N, size, size) where size = (lmax+1)^2
        lmax: Maximum angular momentum
        U_list: List of transformation matrices from get_euler_transforms

    Returns:
        Transformed Wigner D matrix of shape (N, size, size)
    """
    D_out = D.clone()

    for ell in range(2, lmax + 1):  # Skip l=0,1 (identity transform)
        start = ell * ell
        end = start + 2 * ell + 1
        U = U_list[ell]

        # D_out[:, start:end, start:end] = U @ D[:, start:end, start:end] @ U.T
        block = D[:, start:end, start:end]
        transformed = torch.einsum("ij,njk,lk->nil", U, block, U)
        D_out[:, start:end, start:end] = transformed

    return D_out


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

    The generators are stored in m-ordered basis. For l=1, a permutation
    matrix P is also cached to convert to Cartesian basis when needed.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for the generators
        device: Device for the generators

    Returns:
        Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P' for l=1 permutation
    """
    key = (lmax, dtype, device)

    if key not in _GENERATOR_CACHE:
        K_x_list = []
        K_y_list = []
        K_z_list = []

        for ell in range(lmax + 1):
            K_x, K_y, K_z = _build_so3_generators(ell)
            K_x_list.append(K_x.to(device=device, dtype=dtype))
            K_y_list.append(K_y.to(device=device, dtype=dtype))
            K_z_list.append(K_z.to(device=device, dtype=dtype))

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
# Quaternion Edge → +Y (Stable Rodrigues Formula)
# =============================================================================


def quaternion_edge_to_y_stable(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion for minimal-arc (Rodrigues) rotation edge → +Y.

    Uses the half-vector formula:
        q = normalize(1 + ey, -ez, 0, ex)

    This is derived from the quaternion for rotating v1 to v2:
        q = normalize(|v1||v2| + v1·v2, v1 × v2)

    With v1 = edge = (ex, ey, ez) and v2 = (0, 1, 0):
        v1·v2 = ey
        v1 × v2 = (ez*0 - ey*0, ex*0 - ez*1, ey*0 - ex*0) = (-ez, 0, ex)

    Stable everywhere except edge = -Y (180° ambiguity).
    At -Y: 180° rotation about X-axis is used.

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    # q = (1 + ey, -ez, 0, ex) then normalize
    # This is the half-vector formula
    w = 1.0 + ey
    x = -ez
    y = torch.zeros_like(ex)
    z = ex

    q = torch.stack([w, x, y, z], dim=-1)

    # Compute norm for normalization
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)

    # Handle edge near -Y: norm ≈ 0 when ey ≈ -1
    # Use 180° rotation about X: (0, 1, 0, 0)
    near_minus_y = norm.squeeze(-1) < 1e-6
    q_180x = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    q_180x = q_180x.expand_as(q)

    # Safe normalization
    safe_norm = norm.clamp(min=1e-12)
    q_normalized = q / safe_norm

    # Select based on singularity
    q_out = torch.where(near_minus_y.unsqueeze(-1), q_180x, q_normalized)

    return q_out


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

        K_x = K_x_list[ell]
        K_y = K_y_list[ell]
        K_z = K_z_list[ell]

        # K = axis · K^l: shape (N, block_size, block_size)
        # axis: (N, 3), K_x/y/z: (block_size, block_size)
        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)

        # D^l = exp(angle * K)
        D_ell = torch.linalg.matrix_exp(angle[:, None, None] * K)

        # For l=1, transform from m-ordering (y,z,x) to Cartesian (x,y,z)
        if ell == 1:
            D_ell = torch.einsum('ij,njk,kl->nil', P, D_ell, P.T)

        D[:, block_start:block_end, block_start:block_end] = D_ell
        block_start = block_end

    return D


# =============================================================================
# Y-Rotation (Roll Correction)
# =============================================================================


def wigner_d_y_rotation_batched(
    gamma: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
) -> torch.Tensor:
    """
    Compute Wigner D matrices for rotations about the Y-axis.

    D_y(gamma) = exp(gamma * K_y) for roll correction.
    The l=1 block is transformed to Cartesian basis for compatibility.

    Args:
        gamma: Rotation angles of shape (N,)
        generators: Dictionary with 'K_y' list and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
    """
    N = gamma.shape[0]
    device = gamma.device
    dtype = gamma.dtype
    size = (lmax + 1) ** 2

    K_y_list = generators['K_y']
    P = generators['P']

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        K_y = K_y_list[ell]

        # D^l = exp(gamma * K_y)
        D_ell = torch.linalg.matrix_exp(gamma[:, None, None] * K_y)

        # For l=1, transform to Cartesian basis
        if ell == 1:
            D_ell = torch.einsum('ij,njk,kl->nil', P, D_ell, P.T)

        D[:, block_start:block_end, block_start:block_end] = D_ell
        block_start = block_end

    return D


# =============================================================================
# Gamma Computation for Euler Matching
# =============================================================================


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

    This approach uses the Rodrigues (minimal-arc) rotation to map edge → +Y,
    which only has a singularity at edge = -Y (180° rotation ambiguity).
    This avoids the ZYZ Euler angle singularities at ±Y.

    The output uses the same real spherical harmonic basis as the Euler-based
    implementation (rotation.py), making this a drop-in replacement.

    Pipeline:
    1. Normalize edges
    2. Compute Rodrigues quaternion (edge → +Y)
    3. Extract axis-angle
    4. Compute D_rodrigues via matrix_exp
    5. Compute gamma (random by default, or -atan2(ex, ez) for Euler matching)
    6. Apply D_y(gamma) roll correction
    7. Apply Euler-matching basis transformation for l >= 2
    8. Return D = D_y(gamma) @ D_rodrigues (transformed)

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

    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    # Step 1: Normalize edges
    edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)

    # Step 2: Get generators (cached)
    generators = get_so3_generators(lmax, dtype, device)

    # Step 3: Compute Rodrigues quaternion (edge → +Y)
    # This has a pole only at -Y, not at +Y like the Euler approach
    q = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Extract axis-angle
    axis, angle = quaternion_to_axis_angle(q)

    # Step 5: Compute D_rodrigues via matrix_exp
    D_rodrigues = wigner_d_from_axis_angle_batched(axis, angle, generators, lmax)

    # Step 6: Compute gamma if not provided
    if gamma is None:
        if use_euler_gamma:
            # Use atan2-based gamma to exactly match Euler code output
            # Note: has gradient singularity at edge = +Y
            gamma = compute_euler_matching_gamma(edge_normalized)
        else:
            # Random gamma for SO(2) equivariance (default for training)
            N = edge_normalized.shape[0]
            gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    # Step 7: Compute D_y(gamma) roll correction
    D_y_gamma = wigner_d_y_rotation_batched(gamma, generators, lmax)

    # Step 8: Combine: D = D_y(gamma) @ D_rodrigues
    D = torch.bmm(D_y_gamma, D_rodrigues)

    # Step 9: Apply Euler-matching basis transformation for l >= 2
    if lmax >= 2:
        U_list = get_euler_transforms(lmax, dtype, device)
        D = apply_euler_transform(D, lmax, U_list)

    # Step 10: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv


# =============================================================================
# Convenience function matching quaternion_wigner interface
# =============================================================================


def axis_angle_wigner_random_gamma(
    edge_distance_vec: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D with random gamma (matching training behavior).

    This matches the behavior of the Euler-based code which uses random gamma
    for SO(2) equivariance during training. The output uses the same basis
    as the Euler code, making this a drop-in replacement.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge)
    """
    N = edge_distance_vec.shape[0]
    dtype = edge_distance_vec.dtype
    device = edge_distance_vec.device

    gamma = torch.rand(N, dtype=dtype, device=device) * 2 * math.pi

    return axis_angle_wigner(edge_distance_vec, lmax, gamma=gamma)
