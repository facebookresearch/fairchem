"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

# =============================================================================
# Constants
# =============================================================================

# Blend region parameters for two-chart quaternion computation
# The blend region is ey in [_BLEND_START, _BLEND_START + _BLEND_WIDTH]
# which corresponds to ey in [-0.9, 0.9]
_BLEND_START = -0.9
_BLEND_WIDTH = 1.8


# =============================================================================
# Core Helper Functions
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
    # q_sq -> 0 at this chart's singularity (ey = -1), but this chart is
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
    # q_sq -> 0 at this chart's singularity (ey = +1), but this chart is
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
