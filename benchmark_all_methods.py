#!/usr/bin/env python3
"""
Benchmark of Wigner D computation methods for l=2, l=3, l=4.

Compares custom kernels (einsum/matmul) vs matrix exponential (reference).

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    get_so3_generators,
    quaternion_to_axis_angle,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    quaternion_to_wigner_d_l2_einsum,
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
    K_x = gens["K_x"][ell]
    K_y = gens["K_y"][ell]
    K_z = gens["K_z"][ell]

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
            "l2_kernel": quaternion_to_wigner_d_l2_einsum,
            "l3_kernel": quaternion_to_wigner_d_l3_matmul,
            "l4_kernel": quaternion_to_wigner_d_l4_matmul,
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

        # Custom kernel (einsum)
        t, D_kernel = benchmark_function(funcs["l2_kernel"], q)
        print(f"    Einsum kernel:      {t*1000:8.4f} ms")
        results[batch_size]["l2_kernel"] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]["l2_matexp"] = t

        # Speedup
        speedup = results[batch_size]["l2_matexp"] / results[batch_size]["l2_kernel"]
        print(f"    Speedup: {speedup:.2f}x")

        # =====================================================================
        # l=3 benchmarks
        # =====================================================================
        print("\n  l=3:")
        ell = 3
        gens = get_so3_generators(ell, dtype, device)

        # Custom kernel (matmul)
        t, D_kernel = benchmark_function(funcs["l3_kernel"], q)
        print(f"    Matmul kernel:      {t*1000:8.4f} ms")
        results[batch_size]["l3_kernel"] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]["l3_matexp"] = t

        # Speedup
        speedup = results[batch_size]["l3_matexp"] / results[batch_size]["l3_kernel"]
        print(f"    Speedup: {speedup:.2f}x")

        # =====================================================================
        # l=4 benchmarks
        # =====================================================================
        print("\n  l=4:")
        ell = 4
        gens = get_so3_generators(ell, dtype, device)

        # Custom kernel (matmul)
        t, D_kernel = benchmark_function(funcs["l4_kernel"], q)
        print(f"    Matmul kernel:      {t*1000:8.4f} ms")
        results[batch_size]["l4_kernel"] = t

        # Matrix exponential
        t, D_matexp = benchmark_function(matrix_exp_wigner_d, q, ell, gens)
        print(f"    Matrix exponential: {t*1000:8.4f} ms")
        results[batch_size]["l4_matexp"] = t

        # Speedup
        speedup = results[batch_size]["l4_matexp"] / results[batch_size]["l4_kernel"]
        print(f"    Speedup: {speedup:.2f}x")

    return results


def print_speedup_summary(results):
    """Print speedup summary table."""
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY (kernel vs matrix exponential)")
    print("=" * 80)

    print(f"\n{'Batch':>12} | {'l=2':>12} | {'l=3':>12} | {'l=4':>12}")
    print("-" * 60)

    for batch_size in results:
        l2_speedup = results[batch_size]["l2_matexp"] / results[batch_size]["l2_kernel"]
        l3_speedup = results[batch_size]["l3_matexp"] / results[batch_size]["l3_kernel"]
        l4_speedup = results[batch_size]["l4_matexp"] / results[batch_size]["l4_kernel"]
        print(
            f"{batch_size:>12} | {l2_speedup:>10.2f}x | {l3_speedup:>10.2f}x | {l4_speedup:>10.2f}x"
        )


def run_backward_benchmarks(batch_sizes, device, dtype=torch.float64, funcs=None):
    """Run backward pass benchmarks."""
    if funcs is None:
        funcs = {
            "l2_kernel": quaternion_to_wigner_d_l2_einsum,
            "l3_kernel": quaternion_to_wigner_d_l3_matmul,
            "l4_kernel": quaternion_to_wigner_d_l4_matmul,
        }

    print("\n\nBACKWARD PASS (forward + backward)")
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
            kernel_func = funcs[f"l{ell}_kernel"]

            # Custom kernel - forward + backward
            def kernel_fwd_bwd():
                q = torch.randn(
                    batch_size, 4, dtype=dtype, device=device, requires_grad=True
                )
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = kernel_func(q_norm)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(kernel_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Kernel:             {t*1000:8.4f} ms")
            results[batch_size][f"l{ell}_kernel_bwd"] = t

            # Matrix exponential - forward + backward
            def matexp_fwd_bwd():
                q = torch.randn(
                    batch_size, 4, dtype=dtype, device=device, requires_grad=True
                )
                q_norm = q / q.norm(dim=1, keepdim=True)
                D = matrix_exp_wigner_d(q_norm, ell, gens)
                loss = D.sum()
                loss.backward()
                return q.grad

            t, _ = benchmark_function(matexp_fwd_bwd, n_warmup=5, n_iter=50)
            print(f"    Matrix exponential: {t*1000:8.4f} ms")
            results[batch_size][f"l{ell}_matexp_bwd"] = t

            # Speedup
            speedup = (
                results[batch_size][f"l{ell}_matexp_bwd"]
                / results[batch_size][f"l{ell}_kernel_bwd"]
            )
            print(f"    Speedup: {speedup:.2f}x")

    return results


def print_backward_summary(results):
    """Print backward pass speedup summary."""
    print("\n" + "=" * 80)
    print("BACKWARD PASS SPEEDUP SUMMARY (kernel vs matrix exponential)")
    print("=" * 80)

    print(f"\n{'Batch':>12} | {'l=2':>12} | {'l=3':>12} | {'l=4':>12}")
    print("-" * 60)

    for batch_size in results:
        l2_speedup = (
            results[batch_size]["l2_matexp_bwd"] / results[batch_size]["l2_kernel_bwd"]
        )
        l3_speedup = (
            results[batch_size]["l3_matexp_bwd"] / results[batch_size]["l3_kernel_bwd"]
        )
        l4_speedup = (
            results[batch_size]["l4_matexp_bwd"] / results[batch_size]["l4_kernel_bwd"]
        )
        print(
            f"{batch_size:>12} | {l2_speedup:>10.2f}x | {l3_speedup:>10.2f}x | {l4_speedup:>10.2f}x"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Wigner D computation methods"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Apply torch.compile to kernel functions"
    )
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="Skip CPU benchmarks (useful when only GPU results are needed)",
    )
    parser.add_argument(
        "--backward-only",
        action="store_true",
        help="Only run forward+backward benchmarks (skip forward-only)",
    )
    args = parser.parse_args()

    dtype = torch.float64

    funcs = {
        "l2_kernel": quaternion_to_wigner_d_l2_einsum,
        "l3_kernel": quaternion_to_wigner_d_l3_matmul,
        "l4_kernel": quaternion_to_wigner_d_l4_matmul,
    }

    if args.compile:
        print("Applying torch.compile to kernel functions...", flush=True)
        funcs = {k: torch.compile(v) for k, v in funcs.items()}
        print("Compilation done.", flush=True)

    print("=" * 80)
    print("WIGNER D MATRIX COMPUTATION BENCHMARKS")
    print("Custom Kernels (einsum/matmul) vs Matrix Exponential")
    if args.compile:
        print("(with torch.compile)")
    print("=" * 80)

    batch_sizes = [10000, 50000, 100000, 200000, 400000]

    # CPU benchmarks (unless --no-cpu)
    if not args.no_cpu:
        if not args.backward_only:
            print("\n" + "=" * 80)
            print("CPU FORWARD PASS BENCHMARKS")
            print("=" * 80)
            fwd_results = run_benchmarks(
                batch_sizes, device=torch.device("cpu"), funcs=funcs
            )
            print_speedup_summary(fwd_results)

        print("\n" + "=" * 80)
        print("CPU FORWARD+BACKWARD BENCHMARKS")
        print("=" * 80)
        bwd_results = run_backward_benchmarks(
            batch_sizes, device=torch.device("cpu"), funcs=funcs
        )
        print_backward_summary(bwd_results)

    # GPU benchmarks if available
    if torch.cuda.is_available():
        if not args.backward_only:
            print("\n\n" + "=" * 80)
            print("GPU FORWARD PASS BENCHMARKS")
            print("=" * 80)
            gpu_fwd = run_benchmarks(
                batch_sizes, device=torch.device("cuda"), funcs=funcs
            )
            print_speedup_summary(gpu_fwd)

        print("\n" + "=" * 80)
        print("GPU FORWARD+BACKWARD BENCHMARKS")
        print("=" * 80)
        gpu_bwd = run_backward_benchmarks(
            batch_sizes, device=torch.device("cuda"), funcs=funcs
        )
        print_backward_summary(gpu_bwd)
