#!/usr/bin/env python3
"""
Comprehensive benchmark of Wigner D computation methods for l=2, l=3, l=4.

Compares:
1. Direct polynomial (stack-based)
2. Einsum tensor contraction
3. Matrix exponential (reference)

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

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

from wigner_l3_clean import quaternion_to_wigner_d_l3
from wigner_l4_clean import quaternion_to_wigner_d_l4
from wigner_einsum_l3_l4 import (
    quaternion_to_wigner_d_l3_einsum,
    quaternion_to_wigner_d_l4_einsum,
)
from wigner_matmul_l3_l4 import (
    quaternion_to_wigner_d_l3_matmul,
    quaternion_to_wigner_d_l4_matmul,
)


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


def run_benchmarks(batch_sizes, device, dtype=torch.float64):
    """Run benchmarks for different batch sizes."""
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
        t, D_poly = benchmark_function(quaternion_to_wigner_d_l2, q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l2_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(quaternion_to_wigner_d_l2_einsum, q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l2_einsum'] = t

        # Verify einsum matches polynomial
        max_err = (D_poly - D_einsum).abs().max().item()
        print(f"    (einsum verification: {max_err:.2e})")

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
        t, D_poly = benchmark_function(quaternion_to_wigner_d_l3, q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l3_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(quaternion_to_wigner_d_l3_einsum, q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l3_einsum'] = t

        # Matmul (sparse)
        t, D_matmul = benchmark_function(quaternion_to_wigner_d_l3_matmul, q)
        print(f"    Matmul:             {t*1000:8.4f} ms")
        results[batch_size]['l3_matmul'] = t

        # Verify matmul matches polynomial
        max_err = (D_poly - D_matmul).abs().max().item()
        print(f"    (matmul verification: {max_err:.2e})")

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
        t, D_poly = benchmark_function(quaternion_to_wigner_d_l4, q)
        print(f"    Direct polynomial:  {t*1000:8.4f} ms")
        results[batch_size]['l4_poly'] = t

        # Einsum
        t, D_einsum = benchmark_function(quaternion_to_wigner_d_l4_einsum, q)
        print(f"    Einsum:             {t*1000:8.4f} ms")
        results[batch_size]['l4_einsum'] = t

        # Matmul (sparse)
        t, D_matmul = benchmark_function(quaternion_to_wigner_d_l4_matmul, q)
        print(f"    Matmul:             {t*1000:8.4f} ms")
        results[batch_size]['l4_matmul'] = t

        # Verify matmul matches polynomial
        max_err = (D_poly - D_matmul).abs().max().item()
        print(f"    (matmul verification: {max_err:.2e})")

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]['l4_matexp'] = t

    return results


def print_speedup_summary(results):
    """Print speedup summary table."""
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY")
    print("=" * 80)

    print("\nSpeedup vs Matrix Exponential (higher = faster than matexp):")
    print("-" * 80)
    print(f"{'Batch':>12} | {'l=2 poly':>10} {'l=2 eins':>10} | {'l=3 poly':>10} {'l=3 eins':>10} | {'l=4 poly':>10} {'l=4 eins':>10}")
    print("-" * 80)

    for batch_size in results:
        row = [f"{batch_size:>12}"]
        for ell in [2, 3, 4]:
            poly_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_poly']
            einsum_speedup = results[batch_size][f'l{ell}_matexp'] / results[batch_size][f'l{ell}_einsum']
            row.append(f"{poly_speedup:>10.2f}x")
            row.append(f"{einsum_speedup:>10.2f}x")
        print(f"{row[0]} | {row[1]} {row[2]} | {row[3]} {row[4]} | {row[5]} {row[6]}")

    print("\nPoly vs Einsum (>1 = poly faster, <1 = einsum faster):")
    print("-" * 60)
    print(f"{'Batch':>12} | {'l=2':>15} | {'l=3':>15} | {'l=4':>15}")
    print("-" * 60)

    for batch_size in results:
        ratios = []
        for ell in [2, 3, 4]:
            ratio = results[batch_size][f'l{ell}_einsum'] / results[batch_size][f'l{ell}_poly']
            ratios.append(f"{ratio:.2f}x")
        print(f"{batch_size:>12} | {ratios[0]:>15} | {ratios[1]:>15} | {ratios[2]:>15}")


def run_backward_benchmarks(batch_sizes, device, dtype=torch.float64):
    """Run backward pass benchmarks."""
    print(f"\n\nBACKWARD PASS (forward + backward)")
    print(f"Device: {device}, dtype: {dtype}")
    print("=" * 80)

    results = {}

    poly_funcs = {
        2: quaternion_to_wigner_d_l2,
        3: quaternion_to_wigner_d_l3,
        4: quaternion_to_wigner_d_l4,
    }
    einsum_funcs = {
        2: quaternion_to_wigner_d_l2_einsum,
        3: quaternion_to_wigner_d_l3_einsum,
        4: quaternion_to_wigner_d_l4_einsum,
    }

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 60)

        results[batch_size] = {}

        for ell in [2, 3, 4]:
            print(f"\n  l={ell}:")

            gens = get_so3_generators(ell, dtype, device)

            # Direct polynomial - forward + backward
            poly_func = poly_funcs[ell]
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
            einsum_func = einsum_funcs[ell]
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
    print("\n" + "=" * 80)
    print("BACKWARD PASS SPEEDUP SUMMARY")
    print("=" * 80)

    print("\nSpeedup vs Matrix Exponential (higher = faster than matexp):")
    print("-" * 80)
    print(f"{'Batch':>12} | {'l=2 poly':>10} {'l=2 eins':>10} | {'l=3 poly':>10} {'l=3 eins':>10} | {'l=4 poly':>10} {'l=4 eins':>10}")
    print("-" * 80)

    for batch_size in results:
        row = [f"{batch_size:>12}"]
        for ell in [2, 3, 4]:
            poly_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_poly_bwd']
            einsum_speedup = results[batch_size][f'l{ell}_matexp_bwd'] / results[batch_size][f'l{ell}_einsum_bwd']
            row.append(f"{poly_speedup:>10.2f}x")
            row.append(f"{einsum_speedup:>10.2f}x")
        print(f"{row[0]} | {row[1]} {row[2]} | {row[3]} {row[4]} | {row[5]} {row[6]}")

    print("\nPoly vs Einsum (>1 = poly faster, <1 = einsum faster):")
    print("-" * 60)
    print(f"{'Batch':>12} | {'l=2':>15} | {'l=3':>15} | {'l=4':>15}")
    print("-" * 60)

    for batch_size in results:
        ratios = []
        for ell in [2, 3, 4]:
            ratio = results[batch_size][f'l{ell}_einsum_bwd'] / results[batch_size][f'l{ell}_poly_bwd']
            ratios.append(f"{ratio:.2f}x")
        print(f"{batch_size:>12} | {ratios[0]:>15} | {ratios[1]:>15} | {ratios[2]:>15}")


if __name__ == "__main__":
    print("=" * 80)
    print("WIGNER D MATRIX COMPUTATION BENCHMARKS")
    print("Direct Polynomial vs Einsum vs Matrix Exponential")
    print("=" * 80)

    batch_sizes = [100, 1000, 10000]

    print("\n" + "=" * 80)
    print("FORWARD PASS BENCHMARKS")
    print("=" * 80)

    fwd_results = run_benchmarks(batch_sizes, device=torch.device('cpu'))
    print_speedup_summary(fwd_results)

    bwd_results = run_backward_benchmarks(batch_sizes, device=torch.device('cpu'))
    print_backward_summary(bwd_results)

    # GPU benchmarks if available
    if torch.cuda.is_available():
        print("\n\n" + "=" * 80)
        print("GPU BENCHMARKS")
        print("=" * 80)

        gpu_fwd = run_benchmarks(batch_sizes, device=torch.device('cuda'))
        print_speedup_summary(gpu_fwd)

        gpu_bwd = run_backward_benchmarks(batch_sizes, device=torch.device('cuda'))
        print_backward_summary(gpu_bwd)
