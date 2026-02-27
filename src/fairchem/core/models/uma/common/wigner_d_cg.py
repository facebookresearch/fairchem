"""
Wigner-D computation via Clebsch-Gordan recursion.

Uses the recursive formula D^l = D^{l-1} ⊗ D^1 with Wigner 3j symbols.
This provides a unified method for computing Wigner-D matrices for any l
using pure polynomial arithmetic (torch.compile compatible).

Algorithm:
    D^0 = 1 (scalar identity)
    D^1 = R (rotation matrix in spherical harmonic basis)
    D^l = D^{l-1} ⊗ D^1 via Clebsch-Gordan decomposition

The tensor product is computed using Wigner 3j symbols:
    D^l[L,M] = (2l+1) * sum_{a,b,A,B} w3j[a,b,M] * w3j[A,B,L] * D^{l-1}[A,a] * D^1[B,b]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3

from .quaternion_wigner_utils import (
    compute_euler_matching_gamma,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_y_rotation,
)
from .wigner_d_custom_kernels import quaternion_to_rotation_matrix


@dataclass
class CG3jCoefficients:
    """Precomputed Wigner 3j symbols for CG recursion.

    Attributes:
        lmax: Maximum angular momentum
        w3j: Dictionary mapping l -> (2l-1, 3, 2l+1) Wigner 3j tensor
    """

    lmax: int
    w3j: dict[int, torch.Tensor]


def precompute_cg_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CG3jCoefficients:
    """Precompute Wigner 3j symbols for CG recursion.

    Args:
        lmax: Maximum angular momentum to support
        dtype: Data type for coefficient tensors
        device: Device for coefficient tensors

    Returns:
        CG3jCoefficients containing precomputed 3j symbols
    """
    w3j = {}
    for ell in range(2, lmax + 1):
        # wigner_3j(l-1, 1, l) gives symbols for combining D^{l-1} ⊗ D^1 -> D^l
        w3j[ell] = o3.wigner_3j(ell - 1, 1, ell).to(dtype=dtype, device=device)

    return CG3jCoefficients(lmax=lmax, w3j=w3j)


def _cg_combine(
    D_prev: torch.Tensor,
    D_1: torch.Tensor,
    l_out: int,
    w3j: torch.Tensor,
) -> torch.Tensor:
    """Combine D^{l-1} ⊗ D^1 -> D^l via einsum with 3j symbols.

    Args:
        D_prev: (N, 2l-1, 2l-1) batch of D matrices for l-1
        D_1: (N, 3, 3) batch of D^1 matrices
        l_out: Output angular momentum l
        w3j: (2l-1, 3, 2l+1) Wigner 3j symbols

    Returns:
        (N, 2l+1, 2l+1) batch of D^l matrices
    """
    # Tensor product contraction using Clebsch-Gordan coefficients
    # D^l[N,L,M] = w3j[a,b,M] * w3j[A,B,L] * D_prev[N,A,a] * D_1[N,B,b]
    D_out = torch.einsum("abM, ABL, NAa, NBb -> NLM", w3j, w3j, D_prev, D_1)
    return D_out * (2 * l_out + 1)


def quaternion_to_wigner_d_cg(
    q: torch.Tensor,
    lmax: int,
    coeffs: CG3jCoefficients,
) -> torch.Tensor:
    """Compute block-diagonal Wigner-D matrices via CG recursion.

    Args:
        q: (N, 4) quaternions in (w, x, y, z) convention
        lmax: Maximum angular momentum
        coeffs: Precomputed CG3jCoefficients from precompute_cg_coefficients()

    Returns:
        (N, (lmax+1)^2, (lmax+1)^2) block-diagonal Wigner-D matrices
        Each block D^l has shape (2l+1, 2l+1) for l in [0, lmax]
    """
    N = q.shape[0]
    dtype = q.dtype
    device = q.device
    size = (lmax + 1) ** 2

    # Convert quaternion to rotation matrix
    # For l=1, the Wigner-D IS the Cartesian rotation matrix in e3nn convention
    D_1 = quaternion_to_rotation_matrix(q)  # (N, 3, 3)

    # Build D matrices recursively
    D = {}
    D[0] = torch.ones(N, 1, 1, dtype=dtype, device=device)
    D[1] = D_1

    for ell in range(2, lmax + 1):
        D[ell] = _cg_combine(D[ell - 1], D[1], ell, coeffs.w3j[ell])

    # Assemble block-diagonal output
    wigner = torch.zeros(N, size, size, dtype=dtype, device=device)
    start = 0
    for ell in range(lmax + 1):
        end = start + 2 * ell + 1
        wigner[:, start:end, start:end] = D[ell]
        start = end

    return wigner


def axis_angle_wigner_cg(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    coeffs: CG3jCoefficients,
    gamma: Optional[torch.Tensor] = None,
    use_euler_gamma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wigner-D rotation matrices for edge vectors via CG recursion.

    Mirrors the interface of axis_angle_wigner_hybrid but uses
    quaternion_to_wigner_d_cg internally.

    Args:
        edge_distance_vec: (N, 3) edge displacement vectors.
        lmax: Maximum angular momentum.
        coeffs: Precomputed CG3jCoefficients from precompute_cg_coefficients().
        gamma: Optional (N,) SO(2) rotation angles around the edge axis.
               If None, uses random gamma (for SO(2) equivariance during training).
        use_euler_gamma: If True and gamma is None, use -atan2(ex, ez) instead
               of random gamma. This makes output exactly match Euler code.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
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

    # Step 3: Compute quaternion (edge -> +Y)
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)

    # Step 4: Create Y-rotation quaternion and combine with edge->Y
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)

    # Step 5: Compute Wigner D using CG recursion
    D = quaternion_to_wigner_d_cg(q_combined, lmax, coeffs)

    # Step 6: Inverse is transpose (orthogonal matrix)
    D_inv = D.transpose(1, 2).contiguous()

    return D, D_inv
