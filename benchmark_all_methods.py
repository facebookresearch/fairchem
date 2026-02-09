#!/usr/bin/env python3
"""
Comprehensive benchmark of Wigner D computation methods for l=2, l=3, l=4.

Compares:
1. Direct polynomial (stack-based)
2. Einsum tensor contraction
3. Matmul (sparse coefficient matrix)
4. Ra/Rb complex polynomial
5. Matrix exponential (reference)

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import argparse
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fairchem.core.models.uma.common.wigner_d_axis_angle import (
    get_so3_generators,
    quaternion_to_axis_angle,
    quaternion_to_wigner_d_l2,
    quaternion_to_wigner_d_l2_einsum,
)
from fairchem.core.models.uma.common.wigner_d_quaternion import (
    precompute_wigner_coefficients_symmetric,
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
)

from wigner_l3_clean import quaternion_to_wigner_d_l3
from wigner_l4_clean import quaternion_to_wigner_d_l4
from wigner_einsum_l3_l4 import (
    quaternion_to_wigner_d_l3_einsum,
    quaternion_to_wigner_d_l4_einsum,
)
from wigner_matmul_l3_l4 import (
    quaternion_to_wigner_d_l2_matmul,
    quaternion_to_wigner_d_l3_matmul,
    quaternion_to_wigner_d_l4_matmul,
)

# Cache for Ra/Rb coefficients
_RARB_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], dict] = {}


def _get_rarb_coeffs(ell: int, dtype: torch.dtype, device: torch.device) -> dict:
    """Get cached Ra/Rb coefficients."""
    key = (ell, dtype, device)
    if key not in _RARB_COEFF_CACHE:
        _RARB_COEFF_CACHE[key] = precompute_wigner_coefficients_symmetric(ell, dtype=dtype, device=device)
    return _RARB_COEFF_CACHE[key]


def ra_rb_wigner_d(q: torch.Tensor, ell: int, coeffs: dict) -> torch.Tensor:
    """Compute Wigner D using Ra/Rb real polynomial method."""
    ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
    D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
    # Extract the l block
    start = ell * ell
    end = (ell + 1) * (ell + 1)
    return D_re[:, start:end, start:end]


def ra_rb_wigner_d_complex(q: torch.Tensor, ell: int, coeffs: dict) -> torch.Tensor:
    """Compute Wigner D using Ra/Rb complex polynomial method."""
    ra, rb = quaternion_to_ra_rb(q)
    D = wigner_d_matrix_complex(ra, rb, coeffs)
    # Extract the l block (complex D, take real part for rotation)
    start = ell * ell
    end = (ell + 1) * (ell + 1)
    return D[:, start:end, start:end].real


# Pre-built Ra/Rb real wrappers for torch.compile support
_RARB_WRAPPER_CACHE: dict[tuple[int, torch.dtype, torch.device], callable] = {}


def _make_rarb_wrapper(ell: int, dtype: torch.dtype, device: torch.device) -> callable:
    """Create a compilable wrapper for Ra/Rb real that captures coeffs for a specific device."""
    coeffs = _get_rarb_coeffs(ell, dtype, device)
    def wrapper(q: torch.Tensor) -> torch.Tensor:
        return ra_rb_wigner_d(q, ell, coeffs)
    return wrapper


def _get_rarb_wrapper(ell: int, dtype: torch.dtype, device: torch.device) -> callable:
    """Get a cached compilable wrapper for Ra/Rb real."""
    key = (ell, dtype, device)
    if key not in _RARB_WRAPPER_CACHE:
        _RARB_WRAPPER_CACHE[key] = _make_rarb_wrapper(ell, dtype, device)
    return _RARB_WRAPPER_CACHE[key]


def benchmark_function(func, *args, n_warmup=10, n_iter=100, **kwargs):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(n_warmup):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(n_iter):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / n_iter, result


def matrix_exp_wigner_d(q: torch.Tensor, ell: int, gens: dict) -> torch.Tensor:
    """Compute Wigner D using matrix exponential (batched)."""
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)

    # Batched computation
    n = axis  # (N, 3)
    K = n[:, 0:1, None] * K_x + n[:, 1:2, None] * K_y + n[:, 2:3, None] * K_z
    angle_K = angle[:, None, None] * K
    D = torch.linalg.matrix_exp(angle_K)
    return D


def run_benchmarks(batch_sizes, device, dtype=torch.float64, funcs=None):
    """Run benchmarks for different batch sizes."""
    if funcs is None:
        funcs = {
            'l2_poly': quaternion_to_wigner_d_l2,
            'l2_einsum': quaternion_to_wigner_d_l2_einsum,
            'l2_matmul': quaternion_to_wigner_d_l2_matmul,
            'l2_rarb': _get_rarb_wrapper(2, dtype, device),
            'l3_poly': quaternion_to_wigner_d_l3,
            'l3_einsum': quaternion_to_wigner_d_l3_einsum,
            'l3_matmul': quaternion_to_wigner_d_l3_matmul,
            'l3_rarb': _get_rarb_wrapper(3, dtype, device),
            'l4_poly': quaternion_to_wigner_d_l4,
            'l4_einsum': quaternion_to_wigner_d_l4_einsum,
            'l4_matmul': quaternion_to_wigner_d_l4_matmul,
            'l4_rarb': _get_rarb_wrapper(4, dtype, device),
        }

    print(f"\nDevice: {device}, dtype: {dtype}")
    print("=" * 80)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 60)

        # Generate random quaternions
        torch.manual_seed(42)
        q_raw = torch.randn(batch_size, 4, dtype=dtype, device=device)
        q = q_raw / q_raw.norm(dim=1, keepdim=True)

        results[batch_size] = {}

        # =====================================================================
        # l=2 benchmarks
        # =====================================================================
        print("\n  l=2:")
        ell = 2

        gens = get_so3_generators(ell, dtype, device)

        # Direct polynomial
        t, D_poly = benchmark_function(funcs['l2_poly'], q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l2_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(funcs['l2_einsum'], q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l2_einsum'] = t

        # Matmul
        t, D_matmul = benchmark_function(funcs['l2_matmul'], q)
        print(f"    Matmul:             {t*1000:8.4f} ms")
        results[batch_size]['l2_matmul'] = t

        # Verify matmul matches polynomial
        max_err = (D_poly - D_matmul).abs().max().item()
        print(f"    (matmul verification: {max_err:.2e})")

        # Ra/Rb real
        t, D_rarb = benchmark_function(funcs['l2_rarb'], q)
        print(f"    Ra/Rb real:         {t*1000:8.4f} ms")
        results[batch_size]['l2_rarb'] = t

        # Ra/Rb complex (cannot be compiled)
        coeffs = _get_rarb_coeffs(ell, dtype, device)
        t, D_rarb_c = benchmark_function(ra_rb_wigner_d_complex, q, ell, coeffs)
        print(f"    Ra/Rb complex:      {t*1000:8.4f} ms")
        results[batch_size]['l2_rarb_c'] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]['l2_matexp'] = t

        # =====================================================================
        # l=3 benchmarks
        # =====================================================================
        print("\n  l=3:")
        ell = 3

        gens = get_so3_generators(ell, dtype, device)

        # Direct polynomial
        t, D_poly = benchmark_function(funcs['l3_poly'], q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l3_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(funcs['l3_einsum'], q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l3_einsum'] = t

        # Matmul (sparse)
        t, D_matmul = benchmark_function(funcs['l3_matmul'], q)
        print(f"    Matmul:             {t*1000:8.4f} ms")
        results[batch_size]['l3_matmul'] = t

        # Verify matmul matches polynomial
        max_err = (D_poly - D_matmul).abs().max().item()
        print(f"    (matmul verification: {max_err:.2e})")

        # Ra/Rb real
        t, D_rarb = benchmark_function(funcs['l3_rarb'], q)
        print(f"    Ra/Rb real:         {t*1000:8.4f} ms")
        results[batch_size]['l3_rarb'] = t

        # Ra/Rb complex (cannot be compiled)
        coeffs = _get_rarb_coeffs(ell, dtype, device)
        t, D_rarb_c = benchmark_function(ra_rb_wigner_d_complex, q, ell, coeffs)
        print(f"    Ra/Rb complex:      {t*1000:8.4f} ms")
        results[batch_size]['l3_rarb_c'] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]['l3_matexp'] = t

        # =====================================================================
        # l=4 benchmarks
        # =====================================================================
        print("\n  l=4:")
        ell = 4

        gens = get_so3_generators(ell, dtype, device)

        # Direct polynomial
        t, D_poly = benchmark_function(funcs['l4_poly'], q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l4_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(funcs['l4_einsum'], q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l4_einsum'] = t

        # Matmul (sparse)
        t, D_matmul = benchmark_function(funcs['l4_matmul'], q)
        print(f"    Matmul:             {t*1000:8.4f} ms")
        results[batch_size]['l4_matmul'] = t

        # Verify matmul matches polynomial
        max_err = (D_poly - D_matmul).abs().max().item()
        print(f"    (matmul verification: {max_err:.2e})")

        # Ra/Rb real
        t, D_rarb = benchmark_function(funcs['l4_rarb'], q)
        print(f"    Ra/Rb real:         {t*1000:8.4f} ms")
        results[batch_size]['l4_rarb'] = t

        # Ra/Rb complex (cannot be compiled)
        coeffs = _get_rarb_coeffs(ell, dtype, device)
        t, D_rarb_c = benchmark_function(ra_rb_wigner_d_complex, q, ell, coeffs)
        print(f"    Ra/Rb complex:      {t*1000:8.4f} ms")
        results[batch_size]['l4_rarb_c'] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]['l4_matexp'] = t

    return results


def print_speedup_summary(results):
    """Print speedup summary table."""
    print("\n" + "=" * 140)
    print("SPEEDUP SUMMARY")
    print("=" * 140)

    print("\nSpeedup vs Matrix Exponential (higher = faster than matexp):")
    print("-" * 140)
    print(f"{'Batch':>12} | {'l=2 poly':>8} {'l=2 eins':>8} {'l=2 matm':>8} {'l=2 rarb':>8} {'l=2 rarc':>8} | {'l=3 poly':>8} {'l=3 eins':>8} {'l=3 matm':>8} {'l=3 rarb':>8} {'l=3 rarc':>8} | {'l=4 poly':>8} {'l=4 eins':>8} {'l=4 matm':>8} {'l=4 rarb':>8} {'l=4 rarc':>8}")
    print("-" * 140)

    for batch_size in results:
        row = [f"{batch_size:>12}"]
        for ell in [2, 3, 4]:
            poly_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_poly']
            einsum_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_einsum']
            matmul_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_matmul']
            rarb_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_rarb']
            rarb_c_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_rarb_c']
            row.append(f"{poly_speedup:>8.2f}x")
            row.append(f"{einsum_speedup:>8.2f}x")
            row.append(f"{matmul_speedup:>8.2f}x")
            row.append(f"{rarb_speedup:>8.2f}x")
            row.append(f"{rarb_c_speedup:>8.2f}x")
        print(f"{row[0]} | {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} | {row[6]} {row[7]} {row[8]} {row[9]} {row[10]} | {row[11]} {row[12]} {row[13]} {row[14]} {row[15]}")

    print("\nBest method comparison (ratio to fastest):")
    print("-" * 80)
    print(f"{'Batch':>12} | {'l=2':>20} | {'l=3':>20} | {'l=4':>20}")
    print("-" * 80)

    for batch_size in results:
        best = []
        for ell in [2, 3, 4]:
            times = {
                'poly': results[batch_size][f'l{ell}_poly'],
                'eins': results[batch_size][f'l{ell}_einsum'],
                'matm': results[batch_size][f'l{ell}_matmul'],
                'rarb': results[batch_size][f'l{ell}_rarb'],
                'rarc': results[batch_size][f'l{ell}_rarb_c'],
            }
            fastest = min(times.values())
            winner = [k for k, v in times.items() if v == fastest][0]
            best.append(f"{winner} (1.0x)")
        print(f"{batch_size:>12} | {best[0]:>20} | {best[1]:>20} | {best[2]:>20}")


def run_backward_benchmarks(batch_sizes, device, dtype=torch.float64, funcs=None):
    """Run backward pass benchmarks."""
    if funcs is None:
        funcs = {
            'l2_poly': quaternion_to_wigner_d_l2,
            'l2_einsum': quaternion_to_wigner_d_l2_einsum,
            'l2_matmul': quaternion_to_wigner_d_l2_matmul,
            'l2_rarb': _get_rarb_wrapper(2, dtype, device),
            'l3_poly': quaternion_to_wigner_d_l3,
            'l3_einsum': quaternion_to_wigner_d_l3_einsum,
            'l3_matmul': quaternion_to_wigner_d_l3_matmul,
            'l3_rarb': _get_rarb_wrapper(3, dtype, device),
            'l4_poly': quaternion_to_wigner_d_l4,
            'l4_einsum': quaternion_to_wigner_d_l4_einsum,
            'l4_matmul': quaternion_to_wigner_d_l4_matmul,
            'l4_rarb': _get_rarb_wrapper(4, dtype, device),
        }

    print(f"\n\nBACKWARD PASS (forward + backward)")
    print(f"Device: {device}, dtype: {dtype}")
    print("=" * 80)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 60)

        results[batch_size] = {}

        for ell in [2, 3, 4]:
            print(f"\n  l={ell}:")

            gens = get_so3_generators(ell, dtype, device)
            poly_func = funcs[f'l{ell}_poly']
            einsum_func = funcs[f'l{ell}_einsum']
            matmul_func = funcs[f'l{ell}_matmul']

            # Direct polynomial - forward + backward
            def poly_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = poly_func(q_norm)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(poly_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Direct polynomial:  {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_poly_bwd'] = t

            # Einsum - forward + backward
            def einsum_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = einsum_func(q_norm)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(einsum_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Einsum:             {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_einsum_bwd'] = t

            # Matmul - forward + backward
            def matmul_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = matmul_func(q_norm)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(matmul_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Matmul:             {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_matmul_bwd'] = t

            # Ra/Rb real - forward + backward
            rarb_func = funcs[f'l{ell}_rarb']
            def rarb_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = rarb_func(q_norm)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(rarb_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Ra/Rb real:         {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_rarb_bwd'] = t

            # Ra/Rb complex - forward + backward (cannot be compiled)
            coeffs = _get_rarb_coeffs(ell, dtype, device)
            def rarb_c_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = ra_rb_wigner_d_complex(q_norm, ell, coeffs)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(rarb_c_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Ra/Rb complex:      {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_rarb_c_bwd'] = t

            # Matrix exponential - forward + backward
            def matexp_fwd_bwd():
                q = torch.randn(batch_size, 4, dtype=dtype, device=device, requires_grad=True)
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = matrix_exp_wigner_d(q_norm, ell, gens)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(matexp_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Matrix exponential: {t*1000:8.4f} ms")
            results[batch_size][f'l{ell}_matexp_bwd'] = t

    return results


def print_backward_summary(results):
    """Print backward pass speedup summary."""
    print("\n" + "=" * 140)
    print("BACKWARD PASS SPEEDUP SUMMARY")
    print("=" * 140)

    print("\nSpeedup vs Matrix Exponential (higher = faster than matexp):")
    print("-" * 140)
    print(f"{'Batch':>12} | {'l=2 poly':>8} {'l=2 eins':>8} {'l=2 matm':>8} {'l=2 rarb':>8} {'l=2 rarc':>8} | {'l=3 poly':>8} {'l=3 eins':>8} {'l=3 matm':>8} {'l=3 rarb':>8} {'l=3 rarc':>8} | {'l=4 poly':>8} {'l=4 eins':>8} {'l=4 matm':>8} {'l=4 rarb':>8} {'l=4 rarc':>8}")
    print("-" * 140)

    for batch_size in results:
        row = [f"{batch_size:>12}"]
        for ell in [2, 3, 4]:
            poly_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_poly_bwd']
            einsum_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_einsum_bwd']
            matmul_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_matmul_bwd']
            rarb_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_rarb_bwd']
            rarb_c_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_rarb_c_bwd']
            row.append(f"{poly_speedup:>8.2f}x")
            row.append(f"{einsum_speedup:>8.2f}x")
            row.append(f"{matmul_speedup:>8.2f}x")
            row.append(f"{rarb_speedup:>8.2f}x")
            row.append(f"{rarb_c_speedup:>8.2f}x")
        print(f"{row[0]} | {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} | {row[6]} {row[7]} {row[8]} {row[9]} {row[10]} | {row[11]} {row[12]} {row[13]} {row[14]} {row[15]}")

    print("\nBest method comparison (ratio to fastest):")
    print("-" * 80)
    print(f"{'Batch':>12} | {'l=2':>20} | {'l=3':>20} | {'l=4':>20}")
    print("-" * 80)

    for batch_size in results:
        best = []
        for ell in [2, 3, 4]:
            times = {
                'poly': results[batch_size][f'l{ell}_poly_bwd'],
                'eins': results[batch_size][f'l{ell}_einsum_bwd'],
                'matm': results[batch_size][f'l{ell}_matmul_bwd'],
                'rarb': results[batch_size][f'l{ell}_rarb_bwd'],
                'rarc': results[batch_size][f'l{ell}_rarb_c_bwd'],
            }
            fastest = min(times.values())
            winner = [k for k, v in times.items() if v == fastest][0]
            best.append(f"{winner} (1.0x)")
        print(f"{batch_size:>12} | {best[0]:>20} | {best[1]:>20} | {best[2]:>20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Wigner D computation methods")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile to polynomial, einsum, matmul, and Ra/Rb real functions")
    args = parser.parse_args()

    # Build function dictionary - Ra/Rb real wrappers need device/dtype
    # We'll create them for CPU float64 initially
    dtype = torch.float64
    device = torch.device('cpu')

    funcs = {
        'l2_poly': quaternion_to_wigner_d_l2,
        'l2_einsum': quaternion_to_wigner_d_l2_einsum,
        'l2_matmul': quaternion_to_wigner_d_l2_matmul,
        'l2_rarb': _get_rarb_wrapper(2, dtype, device),
        'l3_poly': quaternion_to_wigner_d_l3,
        'l3_einsum': quaternion_to_wigner_d_l3_einsum,
        'l3_matmul': quaternion_to_wigner_d_l3_matmul,
        'l3_rarb': _get_rarb_wrapper(3, dtype, device),
        'l4_poly': quaternion_to_wigner_d_l4,
        'l4_einsum': quaternion_to_wigner_d_l4_einsum,
        'l4_matmul': quaternion_to_wigner_d_l4_matmul,
        'l4_rarb': _get_rarb_wrapper(4, dtype, device),
    }

    if args.compile:
        print("Applying torch.compile to real-number functions...")
        funcs = {k: torch.compile(v) for k, v in funcs.items()}

    print("=" * 80)
    print("WIGNER D MATRIX COMPUTATION BENCHMARKS")
    print("Direct Polynomial vs Einsum vs Matmul vs Ra/Rb (real & complex) vs Matrix Exponential")
    if args.compile:
        print("(with torch.compile for poly/einsum/matmul/rarb-real)")
    print("=" * 80)

    batch_sizes = [100, 1000, 10000]

    print("\n" + "=" * 80)
    print("FORWARD PASS BENCHMARKS")
    print("=" * 80)

    fwd_results = run_benchmarks(batch_sizes, device=torch.device('cpu'), funcs=funcs)
    print_speedup_summary(fwd_results)

    bwd_results = run_backward_benchmarks(batch_sizes, device=torch.device('cpu'), funcs=funcs)
    print_backward_summary(bwd_results)

    # GPU benchmarks if available
    if torch.cuda.is_available():
        print("\n\n" + "=" * 80)
        print("GPU BENCHMARKS")
        print("=" * 80)

        # Create GPU-specific funcs with CUDA Ra/Rb wrappers
        gpu_device = torch.device('cuda')
        gpu_funcs = {
            'l2_poly': funcs['l2_poly'],
            'l2_einsum': funcs['l2_einsum'],
            'l2_matmul': funcs['l2_matmul'],
            'l2_rarb': _get_rarb_wrapper(2, dtype, gpu_device),
            'l3_poly': funcs['l3_poly'],
            'l3_einsum': funcs['l3_einsum'],
            'l3_matmul': funcs['l3_matmul'],
            'l3_rarb': _get_rarb_wrapper(3, dtype, gpu_device),
            'l4_poly': funcs['l4_poly'],
            'l4_einsum': funcs['l4_einsum'],
            'l4_matmul': funcs['l4_matmul'],
            'l4_rarb': _get_rarb_wrapper(4, dtype, gpu_device),
        }
        if args.compile:
            # Compile the new GPU Ra/Rb wrappers
            for k in ['l2_rarb', 'l3_rarb', 'l4_rarb']:
                gpu_funcs[k] = torch.compile(gpu_funcs[k])

        gpu_fwd = run_benchmarks(batch_sizes, device=gpu_device, funcs=gpu_funcs)
        print_speedup_summary(gpu_fwd)

        gpu_bwd = run_backward_benchmarks(batch_sizes, device=gpu_device, funcs=gpu_funcs)
        print_backward_summary(gpu_bwd)
