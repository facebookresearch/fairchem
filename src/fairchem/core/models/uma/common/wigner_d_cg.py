"""
Wigner-D computation via Clebsch-Gordan recursion.

Uses the recursive formula D^l = D^{l-1} x D^1 with Wigner 3j symbols:
    D^0 = 1 (scalar identity)
    D^1 = R (rotation matrix from quaternion)
    D^l = D^{l-1} x D^1 via Clebsch-Gordan decomposition

The tensor product is computed using Wigner 3j symbols with basis
transforms pre-folded in, so the recursion directly produces output in
the correct basis (Cartesian for l=1, Euler-matching for l>=2):
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
    get_so3_generators,
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_y_rotation,
)
from .wigner_d_custom_kernels import quaternion_to_rotation_matrix


@dataclass
class CG3jCoefficients:
    """
    Precomputed Wigner 3j symbols and basis transforms for CG recursion.

    Attributes:
        lmax: Maximum angular momentum
        w3j: Dictionary mapping l -> (2l-1, 3, 2l+1) Wigner 3j tensor
        basis_transforms: Dictionary mapping l -> orthogonal transform matrix
            that converts from e3nn's m-ordering basis to the output basis.
            l=1: P matrix (m-ordering -> Cartesian)
            l>=2: U_euler matrix (m-ordering -> Euler-matching)
    """

    lmax: int
    w3j: dict[int, torch.Tensor]
    basis_transforms: dict[int, torch.Tensor]


def precompute_cg_coefficients(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CG3jCoefficients:
    """
    Precompute Wigner 3j symbols and basis transforms for CG recursion.

    Args:
        lmax: Maximum angular momentum to support
        dtype: Data type for coefficient tensors
        device: Device for coefficient tensors

    Returns:
        CG3jCoefficients containing precomputed 3j symbols and basis transforms
    """
    # Basis transforms from get_so3_generators (cached, loads Jd.pt once)
    # l=1: P matrix (m-ordering -> Cartesian)
    # l>=2: U_euler matrix (m-ordering -> Euler-matching)
    gens = get_so3_generators(lmax, dtype, device)
    basis_transforms = gens["basis_transforms"]

    # Pre-transform w3j to absorb all basis changes. This way the CG
    # recursion directly produces output in the correct basis (Cartesian
    # for l=1, Euler-matching for l>=2) without any post-transform.
    #
    # w3j_t[x,y,z] = T_prev[x,X] * P[y,Y] * w3j_raw[X,Y,Z] * T_l[z,Z]
    #
    # where T_prev converts D_prev indices from its output basis to m-ordering,
    # P converts D_1 indices from Cartesian to m-ordering, and T_l converts
    # the output from m-ordering to the target basis for level l.
    w3j = {}
    P = basis_transforms[1]
    for ell in range(2, lmax + 1):
        w3j_raw = o3.wigner_3j(ell - 1, 1, ell).to(dtype=dtype, device=device)
        T_prev = basis_transforms[ell - 1]
        T_l = basis_transforms[ell]
        w3j[ell] = torch.einsum("xX,yY,XYZ,zZ->xyz", T_prev, P, w3j_raw, T_l)

    return CG3jCoefficients(lmax=lmax, w3j=w3j, basis_transforms=basis_transforms)


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
    """
    Compute block-diagonal Wigner-D matrices via CG recursion.

    The w3j coefficients have basis transforms pre-folded in, so the
    recursion directly produces each block in the output basis (Cartesian
    for l=1, Euler-matching for l>=2), consistent with other Wigner D methods.

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

    # Build D matrices recursively. The w3j coefficients have basis
    # transforms pre-folded in, so D[1] stays in Cartesian and each
    # D[ell] is produced directly in the output basis.
    D = {}
    D[0] = torch.ones(N, 1, 1, dtype=dtype, device=device)

    if lmax >= 1:
        D[1] = quaternion_to_rotation_matrix(q)  # (N, 3, 3) Cartesian

        for ell in range(2, lmax + 1):
            D[ell] = _cg_combine(D[ell - 1], D[1], ell, coeffs.w3j[ell])

    # Assemble block-diagonal output (already in correct basis)
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
