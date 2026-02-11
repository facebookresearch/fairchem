"""
Test and benchmark for Triton Wigner D kernels.

This module provides tests for correctness and performance benchmarking
of the Triton kernels against the PyTorch matmul implementations.

Usage:
    # Run tests (requires GPU)
    python -m fairchem.core.models.uma.common.test_wigner_d_triton

    # Or import and run:
    from fairchem.core.models.uma.common.test_wigner_d_triton import (
        test_all,
        benchmark_all,
    )
    test_all()
    benchmark_all()

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

# Check for Triton availability
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def test_l3_triton_correctness() -> bool:
    """Test l=3 Triton kernel against PyTorch matmul reference."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return True

    from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
        quaternion_to_wigner_d_matmul,
    )
    from fairchem.core.models.uma.common.wigner_d_l3_triton_generated import (
        quaternion_to_wigner_d_l3_triton,
    )

    device = torch.device("cuda")
    dtype = torch.float64

    test_cases = [1, 7, 64, 100, 500, 1000]
    all_passed = True

    print("Testing l=3 Triton kernel correctness...")
    for N in test_cases:
        torch.manual_seed(42 + N)
        q = torch.randn(N, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=1, keepdim=True)

        D_ref = quaternion_to_wigner_d_matmul(q, 3)
        D_triton = quaternion_to_wigner_d_l3_triton(q)

        max_err = (D_ref - D_triton).abs().max().item()
        passed = max_err < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  N={N:5d}: max_err={max_err:.2e} [{status}]")
        all_passed = all_passed and passed

    return all_passed


def test_l4_triton_correctness() -> bool:
    """Test l=4 Triton kernel against PyTorch matmul reference."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return True

    from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
        quaternion_to_wigner_d_matmul,
    )
    from fairchem.core.models.uma.common.wigner_d_l4_triton_generated import (
        quaternion_to_wigner_d_l4_triton,
    )

    device = torch.device("cuda")
    dtype = torch.float64

    test_cases = [1, 7, 64, 100, 500, 1000]
    all_passed = True

    print("Testing l=4 Triton kernel correctness...")
    for N in test_cases:
        torch.manual_seed(42 + N)
        q = torch.randn(N, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=1, keepdim=True)

        D_ref = quaternion_to_wigner_d_matmul(q, 4)
        D_triton = quaternion_to_wigner_d_l4_triton(q)

        max_err = (D_ref - D_triton).abs().max().item()
        passed = max_err < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  N={N:5d}: max_err={max_err:.2e} [{status}]")
        all_passed = all_passed and passed

    return all_passed


def test_hybrid_with_triton() -> bool:
    """Test that hybrid function correctly dispatches to Triton."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return True

    from fairchem.core.models.uma.common.wigner_d_hybrid import (
        axis_angle_wigner_hybrid,
        get_triton_status,
        set_use_triton,
    )

    device = torch.device("cuda")
    dtype = torch.float64
    lmax = 4

    print("Testing hybrid function with Triton enabled...")
    print(f"  Triton status: {get_triton_status()}")

    torch.manual_seed(42)
    edges = torch.randn(100, 3, dtype=dtype, device=device)
    gamma = torch.rand(100, dtype=dtype, device=device) * 6.28

    # Test with Triton enabled
    set_use_triton(True)
    D_triton, D_inv_triton = axis_angle_wigner_hybrid(
        edges, lmax, gamma=gamma, l4_kernel=True
    )

    # Test with Triton disabled (fallback to matmul)
    set_use_triton(False)
    D_matmul, D_inv_matmul = axis_angle_wigner_hybrid(
        edges, lmax, gamma=gamma, l4_kernel=True
    )

    # Re-enable Triton
    set_use_triton(True)

    max_err = (D_triton - D_matmul).abs().max().item()
    passed = max_err < 1e-10
    status = "PASS" if passed else "FAIL"
    print(f"  Triton vs Matmul max_err={max_err:.2e} [{status}]")

    # Verify orthogonality
    identity = torch.eye(D_triton.shape[1], dtype=dtype, device=device)
    ortho_err = (D_triton @ D_inv_triton - identity).abs().max().item()
    ortho_passed = ortho_err < 1e-10
    status = "PASS" if ortho_passed else "FAIL"
    print(f"  Orthogonality error={ortho_err:.2e} [{status}]")

    return passed and ortho_passed


def benchmark_kernel(
    fn: Callable,
    q: torch.Tensor,
    n_warmup: int = 10,
    n_iter: int = 100,
) -> float:
    """Benchmark a kernel function, return time in ms."""
    # Warmup
    for _ in range(n_warmup):
        _ = fn(q)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(q)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_iter * 1000


def benchmark_all(batch_sizes: list[int] | None = None) -> None:
    """Benchmark all Triton kernels against PyTorch matmul."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("SKIP: Triton or CUDA not available")
        return

    from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
        quaternion_to_wigner_d_matmul,
    )
    from fairchem.core.models.uma.common.wigner_d_l3_triton_generated import (
        quaternion_to_wigner_d_l3_triton,
    )
    from fairchem.core.models.uma.common.wigner_d_l4_triton_generated import (
        quaternion_to_wigner_d_l4_triton,
    )

    if batch_sizes is None:
        batch_sizes = [100, 1000, 10000, 50000, 100000]

    device = torch.device("cuda")
    dtype = torch.float64

    print("\nBenchmark: Triton vs PyTorch Matmul")
    print("=" * 90)
    print(f"{'Batch':<10} {'l=3 Matmul':<12} {'l=3 Triton':<12} {'Speedup':<10} "
          f"{'l=4 Matmul':<12} {'l=4 Triton':<12} {'Speedup':<10}")
    print("-" * 90)

    for N in batch_sizes:
        torch.manual_seed(42)
        q = torch.randn(N, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=1, keepdim=True)

        # l=3
        t_l3_matmul = benchmark_kernel(lambda x: quaternion_to_wigner_d_matmul(x, 3), q)
        t_l3_triton = benchmark_kernel(quaternion_to_wigner_d_l3_triton, q)
        speedup_l3 = t_l3_matmul / t_l3_triton

        # l=4
        t_l4_matmul = benchmark_kernel(lambda x: quaternion_to_wigner_d_matmul(x, 4), q)
        t_l4_triton = benchmark_kernel(quaternion_to_wigner_d_l4_triton, q)
        speedup_l4 = t_l4_matmul / t_l4_triton

        print(f"{N:<10} {t_l3_matmul:<12.3f} {t_l3_triton:<12.3f} {speedup_l3:<10.2f}x "
              f"{t_l4_matmul:<12.3f} {t_l4_triton:<12.3f} {speedup_l4:<10.2f}x")

    print()


def test_all() -> bool:
    """Run all tests."""
    results = []

    results.append(("l=3 correctness", test_l3_triton_correctness()))
    results.append(("l=4 correctness", test_l4_triton_correctness()))
    results.append(("hybrid dispatch", test_hybrid_with_triton()))

    print("\n" + "=" * 50)
    print("Test Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("=" * 50)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    import sys

    if not TRITON_AVAILABLE:
        print("Triton not available. Install with: pip install triton")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available. Tests require GPU.")
        sys.exit(1)

    print("Running Wigner D Triton kernel tests...\n")

    if test_all():
        benchmark_all()
        sys.exit(0)
    else:
        sys.exit(1)
