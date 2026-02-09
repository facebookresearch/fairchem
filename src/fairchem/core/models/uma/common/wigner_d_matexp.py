"""
Wigner D matrices via matrix exponential of SO(3) generators.

This module provides Wigner D computation using torch.linalg.matrix_exp with
SO(3) Lie algebra generators. Uses specialized optimizations:
- l=0: Trivial (identity)
- l=1: Rodrigues formula (~5x faster than matrix_exp)
- l=2: Quaternion einsum (~20x faster on GPU without compile)
- l=3,4: Quaternion matmul
- l>=5: torch.linalg.matrix_exp

Entry point:
- axis_angle_wigner: Main function for Wigner D from edge vectors

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    compute_euler_matching_gamma,
    get_so3_generators,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_y_rotation,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_l3_matmul,
    quaternion_to_wigner_d_l4_matmul,
)

# =============================================================================
# Wigner D from Axis-Angle (Batched)
# =============================================================================


def wigner_d_from_axis_angle_batched(
    axis: torch.Tensor,
    angle: torch.Tensor,
    generators: dict[str, list[torch.Tensor]],
    lmax: int,
    l3_l4_kernels: bool = False,
) -> torch.Tensor:
    """
    Compute Wigner D matrices from axis-angle representation.

    D^l = exp(angle * (axis . K^l)) for each l block.
    The l=1 block is transformed to Cartesian basis (x,y,z) for compatibility.

    Uses optimizations:
    - l=0: Trivial (identity)
    - l=1: Rodrigues formula (4-5x faster than matrix_exp)
    - l=2: Quaternion einsum (faster than Cayley-Hamilton)
    - l=3,4: Quaternion matmul (faster than matrix_exp)
    - l>=5: matrix_exp

    Args:
        axis: Rotation axes of shape (N, 3), unit vectors
        angle: Rotation angles of shape (N,), in radians
        generators: Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P'
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    size = (lmax + 1) ** 2

    K_x_list = generators["K_x"]
    K_y_list = generators["K_y"]
    K_z_list = generators["K_z"]

    D = torch.zeros(N, size, size, dtype=dtype, device=device)

    # Convert axis-angle to quaternion for l>=1 kernels
    q = None
    if lmax >= 1:
        half_angle = angle * 0.5
        cos_half = torch.cos(half_angle)
        sin_half = torch.sin(half_angle)
        q = torch.stack(
            [
                cos_half,
                sin_half * axis[:, 0],
                sin_half * axis[:, 1],
                sin_half * axis[:, 2],
            ],
            dim=-1,
        )

    block_start = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        block_end = block_start + block_size

        if ell == 0:
            # l=0 is trivial: 1x1 identity
            D[:, 0, 0] = 1.0
        elif ell == 1:
            # l=1: Direct quaternion to rotation matrix (faster than Rodrigues)
            D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
        elif ell == 2:
            # l=2: Use quaternion einsum (faster than Cayley-Hamilton)
            D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q)
        elif l3_l4_kernels and ell == 3:
            # l=3: Use quaternion matmul (faster than matrix_exp)
            D[:, 9:16, 9:16] = quaternion_to_wigner_d_l3_matmul(q)
        elif l3_l4_kernels and ell == 4:
            # l=4: Use quaternion matmul (faster than matrix_exp)
            D[:, 16:25, 16:25] = quaternion_to_wigner_d_l4_matmul(q)
        else:
            # l>=5: Use matrix_exp
            K_x = K_x_list[ell]
            K_y = K_y_list[ell]
            K_z = K_z_list[ell]

            K = (
                axis[:, 0:1, None, None] * K_x
                + axis[:, 1:2, None, None] * K_y
                + axis[:, 2:3, None, None] * K_z
            ).squeeze(1)

            D_ell = torch.linalg.matrix_exp(angle[:, None, None] * K)
            D[:, block_start:block_end, block_start:block_end] = D_ell

        block_start = block_end

    return D


# =============================================================================
# Main Entry Point
# =============================================================================


def axis_angle_wigner(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
    generators: Optional[dict[str, list[torch.Tensor]]] = None,
    l3_l4_kernels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner D from edge vectors using axis-angle representation.

    This approach uses two quaternion charts with NLERP blending to map edge -> +Y,
    which eliminates all singularities (the two charts have complementary singular
    points at +Y and -Y respectively). This avoids both the single-chart Rodrigues
    singularity at -Y and the ZYZ Euler angle singularities at +/-Y.

    The output uses the same real spherical harmonic basis as the Euler-based
    implementation (rotation.py), making this a drop-in replacement.

    Combines edge->Y and gamma rotations into a single quaternion before computing
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
        generators: Optional pre-computed SO(3) generators from get_so3_generators().
               If None, generators are fetched internally (may cause torch.compile
               graph breaks). For optimal torch.compile performance, pre-compute
               generators and pass them here.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
        - wigner_edge_to_y: the edge -> +Y rotation (D @ edge = +Y for l=1)
        - wigner_y_to_edge: the +Y -> edge rotation (D @ +Y = edge for l=1)
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

    # Step 3: Compute quaternion (edge -> +Y) using NLERP-blended two-chart approach
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    # Combined rotation: first edge->Y, then rotate about Y by gamma
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Extract axis-angle from combined quaternion
    axis, angle = quaternion_to_axis_angle(q_combined)

    # Step 6: Get Euler-aligned generators (cached or passed in)
    # These have the Euler transform folded in for l >= 2
    if generators is None:
        generators = get_so3_generators(lmax, dtype, device)

    # Step 7: Compute single Wigner D from combined rotation via matrix_exp
    # Output is directly in Euler basis thanks to Euler-aligned generators
    D = wigner_d_from_axis_angle_batched(
        axis, angle, generators, lmax, l3_l4_kernels=l3_l4_kernels
    )

    # Step 8: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
