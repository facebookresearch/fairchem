#!/usr/bin/env python3
"""
Sparse einsum-based Wigner D matrix computation for l=3 and l=4.

Uses sparse tensor representation to avoid storing/computing zeros.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import math
from itertools import permutations
from typing import Tuple, List

# Caches for sparse coefficient data
_L3_SPARSE_CACHE: dict[tuple[torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_L4_SPARSE_CACHE: dict[tuple[torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _derive_coefficients_numerical(ell: int) -> Tuple[torch.Tensor, list]:
    """Derive Wigner D polynomial coefficients numerically."""
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        get_so3_generators,
        quaternion_to_axis_angle,
    )

    size = 2 * ell + 1
    degree = 2 * ell

    # Generate all monomials of degree exactly 2l
    monomials = []
    def generate_monomials(n_vars, total_degree, current=None):
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


def _build_sparse_coefficients(ell: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sparse representation of coefficient tensor.

    Returns:
        ij_indices: (N_terms, 2) tensor of (i, j) matrix indices
        q_indices: (N_terms, 2*ell) tensor of quaternion component indices
        coeffs: (N_terms,) tensor of coefficient values

    The Wigner D matrix is computed as:
        D[i,j] = sum over terms where ij_indices matches (i,j) of:
                 coeffs[term] * prod_k q[q_indices[term, k]]
    """
    coefficients, monomials = _derive_coefficients_numerical(ell)
    size = 2 * ell + 1
    degree = 2 * ell

    ij_list = []
    q_idx_list = []
    coeff_list = []

    for i in range(size):
        for j in range(size):
            for k, (a, b, c, d) in enumerate(monomials):
                coeff = coefficients[i, j, k].item()
                if abs(coeff) < 1e-12:
                    continue

                # Create the index list: a copies of 0, b copies of 1, etc.
                indices = tuple([0] * a + [1] * b + [2] * c + [3] * d)

                ij_list.append([i, j])
                q_idx_list.append(indices)
                coeff_list.append(coeff)

    ij_indices = torch.tensor(ij_list, dtype=torch.long)
    q_indices = torch.tensor(q_idx_list, dtype=torch.long)
    coeffs = torch.tensor(coeff_list, dtype=torch.float64)

    return ij_indices, q_indices, coeffs


def _get_l3_sparse_coefficients(dtype: torch.dtype, device: torch.device):
    """Get cached sparse coefficients for l=3."""
    key = (dtype, device)
    if key not in _L3_SPARSE_CACHE:
        ij, q_idx, coeffs = _build_sparse_coefficients(3)
        _L3_SPARSE_CACHE[key] = (
            ij.to(device=device),
            q_idx.to(device=device),
            coeffs.to(dtype=dtype, device=device)
        )
    return _L3_SPARSE_CACHE[key]


def _get_l4_sparse_coefficients(dtype: torch.dtype, device: torch.device):
    """Get cached sparse coefficients for l=4."""
    key = (dtype, device)
    if key not in _L4_SPARSE_CACHE:
        ij, q_idx, coeffs = _build_sparse_coefficients(4)
        _L4_SPARSE_CACHE[key] = (
            ij.to(device=device),
            q_idx.to(device=device),
            coeffs.to(dtype=dtype, device=device)
        )
    return _L4_SPARSE_CACHE[key]


def quaternion_to_wigner_d_l3_sparse(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 7x7 l=3 Wigner D matrix using sparse coefficient representation.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3
    """
    ij_indices, q_indices, coeffs = _get_l3_sparse_coefficients(q.dtype, q.device)
    N = q.shape[0]
    n_terms = coeffs.shape[0]

    # Gather quaternion components for each term: (N, n_terms, 6)
    # q_indices is (n_terms, 6), we need q[:, q_indices] -> (N, n_terms, 6)
    q_gathered = q[:, q_indices]  # (N, n_terms, 6)

    # Compute product of quaternion components for each term: (N, n_terms)
    q_products = q_gathered.prod(dim=-1)

    # Multiply by coefficients: (N, n_terms)
    weighted = q_products * coeffs

    # Scatter-add into output matrix
    # ij_indices is (n_terms, 2)
    D = torch.zeros(N, 7, 7, dtype=q.dtype, device=q.device)

    # Use index_add_ or scatter_add
    # Flatten the output for scatter: (N, 49)
    flat_idx = ij_indices[:, 0] * 7 + ij_indices[:, 1]  # (n_terms,)
    D_flat = D.view(N, 49)

    # weighted is (N, n_terms), flat_idx is (n_terms,)
    # We need to add weighted[:, t] to D_flat[:, flat_idx[t]] for each t
    D_flat.scatter_add_(1, flat_idx.unsqueeze(0).expand(N, -1), weighted)

    return D.view(N, 7, 7)


def quaternion_to_wigner_d_l4_sparse(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 9x9 l=4 Wigner D matrix using sparse coefficient representation.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 9, 9) for l=4
    """
    ij_indices, q_indices, coeffs = _get_l4_sparse_coefficients(q.dtype, q.device)
    N = q.shape[0]
    n_terms = coeffs.shape[0]

    # Gather quaternion components for each term: (N, n_terms, 8)
    q_gathered = q[:, q_indices]  # (N, n_terms, 8)

    # Compute product of quaternion components for each term: (N, n_terms)
    q_products = q_gathered.prod(dim=-1)

    # Multiply by coefficients: (N, n_terms)
    weighted = q_products * coeffs

    # Scatter-add into output matrix
    D = torch.zeros(N, 9, 9, dtype=q.dtype, device=q.device)
    flat_idx = ij_indices[:, 0] * 9 + ij_indices[:, 1]  # (n_terms,)
    D_flat = D.view(N, 81)
    D_flat.scatter_add_(1, flat_idx.unsqueeze(0).expand(N, -1), weighted)

    return D.view(N, 9, 9)


if __name__ == "__main__":
    from wigner_l3_clean import quaternion_to_wigner_d_l3
    from wigner_l4_clean import quaternion_to_wigner_d_l4

    print("Testing sparse versions against polynomial versions...")

    torch.manual_seed(42)
    n_test = 100
    q_raw = torch.randn(n_test, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Test l=3
    print("\nl=3:")
    ij, q_idx, coeffs = _get_l3_sparse_coefficients(torch.float64, torch.device('cpu'))
    print(f"  Sparse terms: {coeffs.shape[0]} (vs 200,704 dense elements)")
    D_poly = quaternion_to_wigner_d_l3(q)
    D_sparse = quaternion_to_wigner_d_l3_sparse(q)
    max_err = (D_poly - D_sparse).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=3 sparse error too large: {max_err}"
    print("  PASSED")

    # Test l=4
    print("\nl=4:")
    ij, q_idx, coeffs = _get_l4_sparse_coefficients(torch.float64, torch.device('cpu'))
    print(f"  Sparse terms: {coeffs.shape[0]} (vs 5,308,416 dense elements)")
    D_poly = quaternion_to_wigner_d_l4(q)
    D_sparse = quaternion_to_wigner_d_l4_sparse(q)
    max_err = (D_poly - D_sparse).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=4 sparse error too large: {max_err}"
    print("  PASSED")

    print("\nAll tests passed!")

    # Quick benchmark
    import time
    print("\n" + "="*60)
    print("Quick benchmark (N=1000):")
    print("="*60)

    q_bench = torch.randn(1000, 4, dtype=torch.float64)
    q_bench = q_bench / q_bench.norm(dim=1, keepdim=True)

    for ell, poly_func, sparse_func in [
        (3, quaternion_to_wigner_d_l3, quaternion_to_wigner_d_l3_sparse),
        (4, quaternion_to_wigner_d_l4, quaternion_to_wigner_d_l4_sparse),
    ]:
        # Warmup
        for _ in range(5):
            _ = poly_func(q_bench)
            _ = sparse_func(q_bench)

        # Time polynomial
        start = time.perf_counter()
        for _ in range(50):
            _ = poly_func(q_bench)
        poly_time = (time.perf_counter() - start) / 50

        # Time sparse
        start = time.perf_counter()
        for _ in range(50):
            _ = sparse_func(q_bench)
        sparse_time = (time.perf_counter() - start) / 50

        print(f"\nl={ell}:")
        print(f"  Polynomial: {poly_time*1000:.3f} ms")
        print(f"  Sparse:     {sparse_time*1000:.3f} ms")
        print(f"  Speedup:    {poly_time/sparse_time:.2f}x")
