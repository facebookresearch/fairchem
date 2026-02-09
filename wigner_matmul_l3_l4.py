#!/usr/bin/env python3
"""
Optimized sparse Wigner D matrix computation using monomial precomputation.

Instead of computing each term separately, we:
1. Precompute all unique monomial products (84 for l=3, 165 for l=4)
2. Use a coefficient matrix to combine them into D matrix elements

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import math
from typing import Tuple, List

# Caches
_L3_MATMUL_CACHE: dict[tuple[torch.dtype, torch.device], Tuple[torch.Tensor, List]] = {}
_L4_MATMUL_CACHE: dict[tuple[torch.dtype, torch.device], Tuple[torch.Tensor, List]] = {}


def _derive_coefficients_numerical(ell: int) -> Tuple[torch.Tensor, list]:
    """Derive Wigner D polynomial coefficients numerically."""
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        get_so3_generators,
        quaternion_to_axis_angle,
    )

    size = 2 * ell + 1
    degree = 2 * ell

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


def _build_matmul_data(ell: int) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    """
    Build coefficient matrix for matmul-based computation.

    Returns:
        C: (size*size, n_monomials) coefficient matrix
        monomials: list of (a, b, c, d) power tuples
    """
    coefficients, monomials = _derive_coefficients_numerical(ell)
    size = 2 * ell + 1

    # Reshape to (size*size, n_monomials)
    C = coefficients.view(size * size, len(monomials))

    return C, monomials


def _get_l3_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=3."""
    key = (dtype, device)
    if key not in _L3_MATMUL_CACHE:
        C, monomials = _build_matmul_data(3)
        _L3_MATMUL_CACHE[key] = (C.to(dtype=dtype, device=device), monomials)
    return _L3_MATMUL_CACHE[key]


def _get_l4_matmul_data(dtype: torch.dtype, device: torch.device):
    """Get cached matmul data for l=4."""
    key = (dtype, device)
    if key not in _L4_MATMUL_CACHE:
        C, monomials = _build_matmul_data(4)
        _L4_MATMUL_CACHE[key] = (C.to(dtype=dtype, device=device), monomials)
    return _L4_MATMUL_CACHE[key]


def quaternion_to_wigner_d_l3_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 7x7 l=3 Wigner D matrix using matmul.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3
    """
    C, monomials = _get_l3_matmul_data(q.dtype, q.device)
    N = q.shape[0]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute powers
    w2, w3, w4, w5, w6 = w*w, w*w*w, w**4, w**5, w**6
    x2, x3, x4, x5, x6 = x*x, x*x*x, x**4, x**5, x**6
    y2, y3, y4, y5, y6 = y*y, y*y*y, y**4, y**5, y**6
    z2, z3, z4, z5, z6 = z*z, z*z*z, z**4, z**5, z**6

    powers = {
        0: {0: torch.ones_like(w), 1: w, 2: w2, 3: w3, 4: w4, 5: w5, 6: w6},
        1: {0: torch.ones_like(x), 1: x, 2: x2, 3: x3, 4: x4, 5: x5, 6: x6},
        2: {0: torch.ones_like(y), 1: y, 2: y2, 3: y3, 4: y4, 5: y5, 6: y6},
        3: {0: torch.ones_like(z), 1: z, 2: z2, 3: z3, 4: z4, 5: z5, 6: z6},
    }

    # Build monomial matrix M: (N, n_monomials)
    M = torch.stack([
        powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
        for a, b, c, d in monomials
    ], dim=1)

    # D_flat = M @ C^T: (N, 49)
    D_flat = M @ C.T

    return D_flat.view(N, 7, 7)


def quaternion_to_wigner_d_l4_matmul(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 9x9 l=4 Wigner D matrix using matmul.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Wigner D matrices of shape (N, 9, 9) for l=4
    """
    C, monomials = _get_l4_matmul_data(q.dtype, q.device)
    N = q.shape[0]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Precompute powers up to 8
    w2 = w*w; w3 = w2*w; w4 = w2*w2; w5 = w4*w; w6 = w4*w2; w7 = w4*w3; w8 = w4*w4
    x2 = x*x; x3 = x2*x; x4 = x2*x2; x5 = x4*x; x6 = x4*x2; x7 = x4*x3; x8 = x4*x4
    y2 = y*y; y3 = y2*y; y4 = y2*y2; y5 = y4*y; y6 = y4*y2; y7 = y4*y3; y8 = y4*y4
    z2 = z*z; z3 = z2*z; z4 = z2*z2; z5 = z4*z; z6 = z4*z2; z7 = z4*z3; z8 = z4*z4

    powers = {
        0: {0: torch.ones_like(w), 1: w, 2: w2, 3: w3, 4: w4, 5: w5, 6: w6, 7: w7, 8: w8},
        1: {0: torch.ones_like(x), 1: x, 2: x2, 3: x3, 4: x4, 5: x5, 6: x6, 7: x7, 8: x8},
        2: {0: torch.ones_like(y), 1: y, 2: y2, 3: y3, 4: y4, 5: y5, 6: y6, 7: y7, 8: y8},
        3: {0: torch.ones_like(z), 1: z, 2: z2, 3: z3, 4: z4, 5: z5, 6: z6, 7: z7, 8: z8},
    }

    # Build monomial matrix M: (N, n_monomials)
    M = torch.stack([
        powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
        for a, b, c, d in monomials
    ], dim=1)

    # D_flat = M @ C^T: (N, 81)
    D_flat = M @ C.T

    return D_flat.view(N, 9, 9)


if __name__ == "__main__":
    from wigner_l3_clean import quaternion_to_wigner_d_l3
    from wigner_l4_clean import quaternion_to_wigner_d_l4

    print("Testing matmul versions against polynomial versions...")

    torch.manual_seed(42)
    n_test = 100
    q_raw = torch.randn(n_test, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Test l=3
    print("\nl=3:")
    C, monomials = _get_l3_matmul_data(torch.float64, torch.device('cpu'))
    print(f"  Coefficient matrix: {C.shape} ({C.shape[0]} outputs × {C.shape[1]} monomials)")
    D_poly = quaternion_to_wigner_d_l3(q)
    D_matmul = quaternion_to_wigner_d_l3_matmul(q)
    max_err = (D_poly - D_matmul).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=3 matmul error too large: {max_err}"
    print("  PASSED")

    # Test l=4
    print("\nl=4:")
    C, monomials = _get_l4_matmul_data(torch.float64, torch.device('cpu'))
    print(f"  Coefficient matrix: {C.shape} ({C.shape[0]} outputs × {C.shape[1]} monomials)")
    D_poly = quaternion_to_wigner_d_l4(q)
    D_matmul = quaternion_to_wigner_d_l4_matmul(q)
    max_err = (D_poly - D_matmul).abs().max().item()
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-10, f"l=4 matmul error too large: {max_err}"
    print("  PASSED")

    print("\nAll tests passed!")

    # Benchmark
    import time
    print("\n" + "="*60)
    print("Benchmark (N=1000):")
    print("="*60)

    q_bench = torch.randn(1000, 4, dtype=torch.float64)
    q_bench = q_bench / q_bench.norm(dim=1, keepdim=True)

    for ell, poly_func, matmul_func in [
        (3, quaternion_to_wigner_d_l3, quaternion_to_wigner_d_l3_matmul),
        (4, quaternion_to_wigner_d_l4, quaternion_to_wigner_d_l4_matmul),
    ]:
        # Warmup
        for _ in range(5):
            _ = poly_func(q_bench)
            _ = matmul_func(q_bench)

        # Time polynomial
        start = time.perf_counter()
        for _ in range(50):
            _ = poly_func(q_bench)
        poly_time = (time.perf_counter() - start) / 50

        # Time matmul
        start = time.perf_counter()
        for _ in range(50):
            _ = matmul_func(q_bench)
        matmul_time = (time.perf_counter() - start) / 50

        print(f"\nl={ell}:")
        print(f"  Polynomial: {poly_time*1000:.3f} ms")
        print(f"  Matmul:     {matmul_time*1000:.3f} ms")
        print(f"  Speedup:    {poly_time/matmul_time:.2f}x")
