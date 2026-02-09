#!/usr/bin/env python3
"""
Einsum-based Wigner D matrix computation for l=3 and l=4.

Uses tensor contraction with precomputed coefficient tensors, similar to
quaternion_to_wigner_d_l2_einsum() in wigner_d_axis_angle.py.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import math
from itertools import permutations
from typing import Tuple

# Caches for coefficient tensors
_L3_COEFF_TENSOR_CACHE: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
_L4_COEFF_TENSOR_CACHE: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}


def _derive_coefficients_numerical(ell: int) -> Tuple[torch.Tensor, list]:
    """
    Derive Wigner D polynomial coefficients numerically.

    Returns:
        coefficients: Tensor of shape (size, size, n_monomials)
        monomials: List of (a, b, c, d, ...) tuples for each monomial
    """
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        get_so3_generators,
        quaternion_to_axis_angle,
    )

    size = 2 * ell + 1
    degree = 2 * ell

    # Generate all monomials of degree exactly 2l
    # For l=3: degree=6, we need 6 indices in {0,1,2,3} summing to 6
    # For l=4: degree=8, we need 8 indices
    monomials = []

    def generate_monomials(n_vars, total_degree, current=None):
        """Generate all monomials with exactly total_degree."""
        if current is None:
            current = []
        if len(current) == n_vars - 1:
            current.append(total_degree - sum(current))
            if current[-1] >= 0:
                monomials.append(tuple(current))
            return
        for i in range(total_degree - sum(current) + 1):
            generate_monomials(n_vars, total_degree, current + [i])

    generate_monomials(4, degree)
    n_monomials = len(monomials)

    n_samples = n_monomials + 50

    torch.manual_seed(42)
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Build Vandermonde matrix
    X = torch.zeros(n_samples, n_monomials, dtype=torch.float64)
    for i, (a, b, c, d) in enumerate(monomials):
        X[:, i] = (q[:, 0] ** a) * (q[:, 1] ** b) * (q[:, 2] ** c) * (q[:, 3] ** d)

    gens = get_so3_generators(ell, torch.float64, torch.device('cpu'))
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)

    D_ref = torch.zeros(n_samples, size, size, dtype=torch.float64)
    for i in range(n_samples):
        n = axis[i]
        K = n[0] * K_x + n[1] * K_y + n[2] * K_z
        D_ref[i] = torch.linalg.matrix_exp(angle[i] * K)

    coefficients = torch.zeros(size, size, n_monomials, dtype=torch.float64)

    for i in range(size):
        for j in range(size):
            y = D_ref[:, i, j]
            c, _, _, _ = torch.linalg.lstsq(X, y.unsqueeze(1))
            coefficients[i, j, :] = c.squeeze()

    return coefficients, monomials


def _build_l3_coefficient_tensor() -> torch.Tensor:
    """
    Build the (7, 7, 4, 4, 4, 4, 4, 4) coefficient tensor for einsum-based l=3 computation.

    The tensor C satisfies:
        D[i,j] = sum_{a,b,c,d,e,f} C[i,j,a,b,c,d,e,f] * q[a] * q[b] * q[c] * q[d] * q[e] * q[f]

    Coefficients are symmetrized over all permutations of (a, b, c, d, e, f).
    """
    coefficients, monomials = _derive_coefficients_numerical(3)

    size = 7
    C = torch.zeros(size, size, 4, 4, 4, 4, 4, 4, dtype=torch.float64)

    for i in range(size):
        for j in range(size):
            for k, (a, b, c, d) in enumerate(monomials):
                coeff = coefficients[i, j, k].item()
                if abs(coeff) < 1e-12:
                    continue

                # Create the index list: a copies of 0, b copies of 1, etc.
                indices = [0] * a + [1] * b + [2] * c + [3] * d

                # Symmetrize over all permutations
                perms = set(permutations(indices))
                c_per_perm = coeff / len(perms)

                for p in perms:
                    C[i, j, p[0], p[1], p[2], p[3], p[4], p[5]] += c_per_perm

    return C


def _build_l4_coefficient_tensor() -> torch.Tensor:
    """
    Build the (9, 9, 4, 4, 4, 4, 4, 4, 4, 4) coefficient tensor for einsum-based l=4 computation.

    The tensor C satisfies:
        D[i,j] = sum_{a,b,c,d,e,f,g,h} C[i,j,a,b,c,d,e,f,g,h] * q[a] * ... * q[h]

    Coefficients are symmetrized over all permutations of (a, b, c, d, e, f, g, h).
    """
    coefficients, monomials = _derive_coefficients_numerical(4)

    size = 9
    C = torch.zeros(size, size, 4, 4, 4, 4, 4, 4, 4, 4, dtype=torch.float64)

    for i in range(size):
        for j in range(size):
            for k, (a, b, c, d) in enumerate(monomials):
                coeff = coefficients[i, j, k].item()
                if abs(coeff) < 1e-12:
                    continue

                # Create the index list: a copies of 0, b copies of 1, etc.
                indices = [0] * a + [1] * b + [2] * c + [3] * d

                # Symmetrize over all permutations
                perms = set(permutations(indices))
                c_per_perm = coeff / len(perms)

                for p in perms:
                    C[i, j, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]] += c_per_perm

    return C


def _get_l3_coefficient_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get cached l=3 coefficient tensor for einsum computation."""
    key = (dtype, device)
    if key not in _L3_COEFF_TENSOR_CACHE:
        _L3_COEFF_TENSOR_CACHE[key] = _build_l3_coefficient_tensor().to(dtype=dtype, device=device)
    return _L3_COEFF_TENSOR_CACHE[key]


def _get_l4_coefficient_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get cached l=4 coefficient tensor for einsum computation."""
    key = (dtype, device)
    if key not in _L4_COEFF_TENSOR_CACHE:
        _L4_COEFF_TENSOR_CACHE[key] = _build_l4_coefficient_tensor().to(dtype=dtype, device=device)
    return _L4_COEFF_TENSOR_CACHE[key]


def quaternion_to_wigner_d_l3_einsum(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 7x7 l=3 Wigner D matrix using einsum tensor contraction.

    Expresses D as a tensor contraction:
        D[i,j] = C[i,j,a,b,c,d,e,f] * q[a] * q[b] * q[c] * q[d] * q[e] * q[f]

    where C is a precomputed (7,7,4,4,4,4,4,4) coefficient tensor.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3
    """
    C = _get_l3_coefficient_tensor(q.dtype, q.device)

    # Build q⊗q, then (q⊗q)⊗(q⊗q) = q^⊗4, then q^⊗4 ⊗ q^⊗2 = q^⊗6
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)  # (N, 4, 4)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)  # (N, 4, 4, 4, 4)
    q6 = q4.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)  # (N, 4, 4, 4, 4, 4, 4)

    # Contract with coefficient tensor
    D = torch.einsum('nabcdef,ijabcdef->nij', q6, C)

    return D


def quaternion_to_wigner_d_l4_einsum(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 9x9 l=4 Wigner D matrix using einsum tensor contraction.

    Expresses D as a tensor contraction:
        D[i,j] = C[i,j,a,b,c,d,e,f,g,h] * q[a] * ... * q[h]

    where C is a precomputed (9,9,4,4,4,4,4,4,4,4) coefficient tensor.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 9, 9) for l=4
    """
    C = _get_l4_coefficient_tensor(q.dtype, q.device)

    # Build q⊗q, then (q⊗q)⊗(q⊗q) = q^⊗4, then q^⊗4 ⊗ q^⊗4 = q^⊗8
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)  # (N, 4, 4)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)  # (N, 4, 4, 4, 4)
    q8 = q4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
         q4.unsqueeze(-5).unsqueeze(-5).unsqueeze(-5).unsqueeze(-5)  # (N, 4, 4, 4, 4, 4, 4, 4, 4)

    # Contract with coefficient tensor
    D = torch.einsum('nabcdefgh,ijabcdefgh->nij', q8, C)

    return D


if __name__ == "__main__":
    # Test that einsum versions match the polynomial versions
    from wigner_l3_clean import quaternion_to_wigner_d_l3
    from wigner_l4_clean import quaternion_to_wigner_d_l4

    print("Testing einsum versions against polynomial versions...")

    torch.manual_seed(42)
    n_test = 100
    q_raw = torch.randn(n_test, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Test l=3
    print("\nl=3:")
    D_poly = quaternion_to_wigner_d_l3(q)
    D_einsum = quaternion_to_wigner_d_l3_einsum(q)
    max_err = (D_poly - D_einsum).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=3 einsum error too large: {max_err}"
    print("  PASSED")

    # Test l=4
    print("\nl=4:")
    D_poly = quaternion_to_wigner_d_l4(q)
    D_einsum = quaternion_to_wigner_d_l4_einsum(q)
    max_err = (D_poly - D_einsum).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=4 einsum error too large: {max_err}"
    print("  PASSED")

    print("\nAll tests passed!")
