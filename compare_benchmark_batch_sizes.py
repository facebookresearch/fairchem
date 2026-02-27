"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

"""
Performance comparison: any two Wigner D methods head-to-head.

Usage:
    python compare_benchmark_batch_sizes.py euler hybrid
    python compare_benchmark_batch_sizes.py polynomial cg --quick --cpu
    python compare_benchmark_batch_sizes.py matexp hybrid_matexp --lmax 8
    python compare_benchmark_batch_sizes.py hybrid cg --real --compile
"""

import argparse
import time
from pathlib import Path

import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import get_so3_generators
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.wigner_d_cg import (
    axis_angle_wigner_cg,
    precompute_cg_coefficients,
)
from fairchem.core.models.uma.common.wigner_d_hybrid import axis_angle_wigner_hybrid
from fairchem.core.models.uma.common.wigner_d_hybrid_matexp import (
    axis_angle_wigner_hybrid_matexp,
)
from fairchem.core.models.uma.common.wigner_d_matexp import (
    axis_angle_wigner as axis_angle_wigner_matexp,
)
from fairchem.core.models.uma.common.wigner_d_polynomial import (
    axis_angle_wigner_polynomial,
)

ALL_METHODS = ["euler", "hybrid", "matexp", "polynomial", "hybrid_matexp", "cg"]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare any two Wigner D methods head-to-head"
    )
    parser.add_argument(
        "method1",
        choices=ALL_METHODS,
        help="First method (baseline)",
    )
    parser.add_argument(
        "method2",
        choices=ALL_METHODS,
        help="Second method (comparison)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA device")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile to both methods",
    )
    parser.add_argument("--lmax", type=int, default=6, help="Maximum angular momentum")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Data type",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real arithmetic (hybrid/polynomial only)",
    )
    return parser.parse_args()


def sync_if_cuda(device):
    """
    Synchronize CUDA if using GPU to ensure accurate timing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_fn(fn, n_warmup, n_runs, device):
    """
    Benchmark a function with proper CUDA synchronization.
    """
    for _ in range(n_warmup):
        fn()
    sync_if_cuda(device)

    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms


def check_wigner_correctness(W, method_name, n_edges, lmax, test_edges, device, dtype):
    """
    Check Wigner D matrix correctness.
    """
    errors = []

    expected_size = (lmax + 1) ** 2
    if W.shape != (n_edges, expected_size, expected_size):
        errors.append(
            f"wrong shape {W.shape}, "
            f"expected ({n_edges}, {expected_size}, {expected_size})"
        )

    max_ortho_err = 0.0
    offset = 0
    for ell in range(lmax + 1):
        block_size = 2 * ell + 1
        D_l = W[:, offset : offset + block_size, offset : offset + block_size]
        I_expected = torch.eye(block_size, dtype=dtype, device=device).unsqueeze(0)
        DDT = torch.bmm(D_l, D_l.transpose(-2, -1))
        ortho_err = (DDT - I_expected).abs().max().item()
        max_ortho_err = max(max_ortho_err, ortho_err)
        offset += block_size

    if max_ortho_err > 1e-6:
        errors.append(f"orthogonality error = {max_ortho_err:.2e}")

    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    D_l1 = W[:, 1:4, 1:4]
    rotated = torch.bmm(D_l1, test_edges.unsqueeze(-1)).squeeze(-1)
    align_err = (rotated - y_axis).abs().max().item()

    if align_err > 1e-6:
        errors.append(f"l=1 alignment error = {align_err:.2e}")

    if errors:
        print(f"  {method_name}: FAIL - {', '.join(errors)}")
        return False
    else:
        print(
            f"  {method_name}: OK "
            f"(ortho={max_ortho_err:.2e}, align={align_err:.2e})"
        )
        return True


def make_forward_fn(method, lmax, Jd, generators, cg_coeffs, use_real):
    """
    Return a forward function for the given method name.

    The returned callable takes (edges) and returns (W, extra) where
    extra is None for euler (to unify the interface).

    Each method gets its own distinct function definition so that
    torch.compile treats them as separate code objects (avoiding
    recompilation guards on the method string).
    """
    if method == "euler":

        def forward(edges):
            angles = init_edge_rot_euler_angles(edges)
            W = eulers_to_wigner(angles, 0, lmax, Jd)
            return W, None

    elif method == "hybrid":

        def forward(edges):
            return axis_angle_wigner_hybrid(edges, lmax, use_real_arithmetic=use_real)

    elif method == "matexp":

        def forward(edges):
            return axis_angle_wigner_matexp(edges, lmax, generators=generators)

    elif method == "hybrid_matexp":

        def forward(edges):
            return axis_angle_wigner_hybrid_matexp(edges, lmax, generators=generators)

    elif method == "cg":

        def forward(edges):
            return axis_angle_wigner_cg(edges, lmax, cg_coeffs)

    else:  # polynomial

        def forward(edges):
            return axis_angle_wigner_polynomial(
                edges, lmax, use_real_arithmetic=use_real
            )

    return forward


def main():
    """
    Run head-to-head benchmark of two Wigner D methods.
    """
    args = parse_args()

    if args.method1 == args.method2:
        print(f"WARNING: comparing {args.method1} against itself")

    # Determine device
    if args.cpu:
        device = torch.device("cpu")
    elif args.cuda:
        if not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available")
            return
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    lmax = args.lmax
    methods = (args.method1, args.method2)

    # Benchmark parameters
    if args.quick:
        batch_sizes = [100, 1000, 10000]
        n_warmup = 3
        n_runs = 5
    else:
        batch_sizes = [1000, 10000, 50000, 100000, 500000]
        n_warmup = 5
        n_runs = 20

    print("=" * 90)
    print(f"PERFORMANCE: {args.method1} vs {args.method2} by Batch Size")
    if args.compile:
        print("(with torch.compile)")
    if args.real:
        print("(with real arithmetic)")
    print("=" * 90)
    print(f"lmax = {lmax}, dtype = {dtype}, device = {device}")
    print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Pre-compute shared resources
    Jd = None
    if "euler" in methods:
        jd_path = (
            Path(__file__).parent.resolve()
            / "src"
            / "fairchem"
            / "core"
            / "models"
            / "uma"
            / "Jd.pt"
        )
        Jd = torch.load(jd_path, map_location=device, weights_only=True)
        Jd = [J.to(dtype=dtype) for J in Jd]

    generators = None
    if any(m in ("matexp", "hybrid_matexp") for m in methods):
        generators = get_so3_generators(lmax, dtype, device)

    cg_coeffs = None
    if "cg" in methods:
        cg_coeffs = precompute_cg_coefficients(lmax, dtype, device)

    # Build forward functions
    forward1 = make_forward_fn(args.method1, lmax, Jd, generators, cg_coeffs, args.real)
    forward2 = make_forward_fn(args.method2, lmax, Jd, generators, cg_coeffs, args.real)

    if args.compile:
        print("Applying torch.compile to both methods...")
        forward1_fn = torch.compile(forward1, dynamic=True)
        forward2_fn = torch.compile(forward2, dynamic=True)
    else:
        forward1_fn = forward1
        forward2_fn = forward2

    # Column labels
    label1 = args.method1
    label2 = args.method2

    # Correctness checks
    print("CORRECTNESS CHECKS")
    print("-" * 90)

    test_batch = 100
    torch.manual_seed(42)
    test_edges = torch.randn(test_batch, 3, dtype=dtype, device=device)
    test_edges = torch.nn.functional.normalize(test_edges, dim=-1)

    W1, _ = forward1(test_edges)
    ok1 = check_wigner_correctness(
        W1, label1, test_batch, lmax, test_edges, device, dtype
    )

    W2, _ = forward2(test_edges)
    ok2 = check_wigner_correctness(
        W2, label2, test_batch, lmax, test_edges, device, dtype
    )

    if not (ok1 and ok2):
        print("\nWARNING: Some correctness checks failed!")
    print()

    # Table formatting helpers
    col1 = f"{label1} (ms)"
    col2 = f"{label2} (ms)"
    col1_edge = f"{label1} µs/e"
    col2_edge = f"{label2} µs/e"
    header = (
        f"{'Batch':<8} {col1:<15} {col2:<15} "
        f"{'Ratio (1/2)':<12} {col1_edge:<15} {col2_edge:<15}"
    )

    # Forward pass benchmark
    print("FORWARD PASS ONLY")
    print("-" * 90)
    print(header)
    print("-" * 90)

    forward_results = []

    for n_edges in batch_sizes:
        torch.manual_seed(42)
        edges = torch.randn(n_edges, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)

        def fn1(e=edges):
            return forward1_fn(e)

        def fn2(e=edges):
            return forward2_fn(e)

        t1 = benchmark_fn(fn1, n_warmup, n_runs, device)
        t2 = benchmark_fn(fn2, n_warmup, n_runs, device)

        ratio = t1 / t2 if t2 > 0 else float("inf")
        t1_per = t1 / n_edges * 1000
        t2_per = t2 / n_edges * 1000

        forward_results.append((n_edges, t1, t2, ratio))

        print(
            f"{n_edges:<8} {t1:<15.2f} {t2:<15.2f} "
            f"{ratio:<12.2f} {t1_per:<15.3f} {t2_per:<15.3f}"
        )

    print()

    # Forward + Backward pass benchmark
    print("FORWARD + BACKWARD PASS")
    print("-" * 90)
    print(header)
    print("-" * 90)

    backward_results = []

    for n_edges in batch_sizes:
        torch.manual_seed(42)
        edges_base = torch.randn(n_edges, 3, dtype=dtype, device=device)
        edges_base = torch.nn.functional.normalize(edges_base, dim=-1)

        def bwd1(eb=edges_base):
            edges = eb.clone().requires_grad_(True)
            W, _ = forward1_fn(edges)
            torch.autograd.grad(W.sum(), edges)

        def bwd2(eb=edges_base):
            edges = eb.clone().requires_grad_(True)
            W, _ = forward2_fn(edges)
            torch.autograd.grad(W.sum(), edges)

        t1 = benchmark_fn(bwd1, n_warmup, n_runs, device)
        t2 = benchmark_fn(bwd2, n_warmup, n_runs, device)

        ratio = t1 / t2 if t2 > 0 else float("inf")
        t1_per = t1 / n_edges * 1000
        t2_per = t2 / n_edges * 1000

        backward_results.append((n_edges, t1, t2, ratio))

        print(
            f"{n_edges:<8} {t1:<15.2f} {t2:<15.2f} "
            f"{ratio:<12.2f} {t1_per:<15.3f} {t2_per:<15.3f}"
        )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\nRatio interpretation: {label1} time / {label2} time")
    print(f"  > 1.0 means {label2} is faster")
    print(f"  < 1.0 means {label1} is faster")
    print()

    fwd_ratios = [r[3] for r in forward_results]
    bwd_ratios = [r[3] for r in backward_results]

    print("Forward pass:")
    print(f"  Ratio range: {min(fwd_ratios):.2f} - {max(fwd_ratios):.2f}")
    print(f"  Average ratio: {sum(fwd_ratios) / len(fwd_ratios):.2f}")

    print("\nForward+Backward pass:")
    print(f"  Ratio range: {min(bwd_ratios):.2f} - {max(bwd_ratios):.2f}")
    print(f"  Average ratio: {sum(bwd_ratios) / len(bwd_ratios):.2f}")

    print()


if __name__ == "__main__":
    main()
