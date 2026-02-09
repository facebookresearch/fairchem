"""
Performance comparison: Euler vs Axis-Angle Wigner D implementations.

Usage:
    python benchmark_batch_sizes.py              # Auto-detect device
    python benchmark_batch_sizes.py --cpu        # Force CPU
    python benchmark_batch_sizes.py --cuda       # Force CUDA
    python benchmark_batch_sizes.py --quick      # Quick mode (fewer runs)
"""

import argparse
import torch
import time
from pathlib import Path

from fairchem.core.models.uma.common.rotation import init_edge_rot_euler_angles, eulers_to_wigner
from fairchem.core.models.uma.common.wigner_d_axis_angle import (
    axis_angle_wigner_hybrid as axis_angle_wigner,
    get_so3_generators,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Wigner D implementations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA device")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile to both methods")
    parser.add_argument("--lmax", type=int, default=6, help="Maximum angular momentum")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="Data type")
    return parser.parse_args()


def sync_if_cuda(device):
    """Synchronize CUDA if using GPU to ensure accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_fn(fn, n_warmup, n_runs, device):
    """Benchmark a function with proper CUDA synchronization."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    sync_if_cuda(device)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms


def check_wigner_correctness(W, method_name, n_edges, lmax, test_edges, device, dtype):
    """Check Wigner D matrix correctness."""
    errors = []

    # Check shape
    expected_size = (lmax + 1) ** 2
    if W.shape != (n_edges, expected_size, expected_size):
        errors.append(f"wrong shape {W.shape}, expected ({n_edges}, {expected_size}, {expected_size})")

    # Check orthogonality of each l-block
    max_ortho_err = 0.0
    offset = 0
    for l in range(lmax + 1):
        block_size = 2 * l + 1
        D_l = W[:, offset:offset+block_size, offset:offset+block_size]
        I_expected = torch.eye(block_size, dtype=dtype, device=device).unsqueeze(0)
        DDT = torch.bmm(D_l, D_l.transpose(-2, -1))
        ortho_err = (DDT - I_expected).abs().max().item()
        max_ortho_err = max(max_ortho_err, ortho_err)
        offset += block_size

    if max_ortho_err > 1e-6:
        errors.append(f"orthogonality error = {max_ortho_err:.2e}")

    # Check l=1 alignment
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
        print(f"  {method_name}: OK (ortho={max_ortho_err:.2e}, align={align_err:.2e})")
        return True


def main():
    args = parse_args()

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

    # Benchmark parameters
    if args.quick:
        batch_sizes = [100, 1000, 10000]
        n_warmup = 3
        n_runs = 5
    else:
        batch_sizes = [100, 500, 1000, 2000, 5000] + list(range(10000, 150000, 10000)) + list(range(200000,600000,100000))
        n_warmup = 5
        n_runs = 20

    print("=" * 90)
    print("PERFORMANCE: Euler vs Axis-Angle by Batch Size")
    if args.compile:
        print("(with torch.compile)")
    print("=" * 90)
    print(f"lmax = {lmax}, dtype = {dtype}, device = {device}")
    print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load Jd matrices for Euler
    jd_path = Path(__file__).parent.resolve() / "src" / "fairchem" / "core" / "models" / "uma" / "Jd.pt"
    Jd = torch.load(jd_path, map_location=device, weights_only=True)
    Jd = [J.to(dtype=dtype) for J in Jd]

    # Pre-compute generators for axis-angle (avoids torch.compile graph breaks)
    generators = get_so3_generators(lmax, dtype, device)

    # Define functions to be compiled (once, outside the loops)
    def euler_forward(edges):
        angles = init_edge_rot_euler_angles(edges)
        return eulers_to_wigner(angles, 0, lmax, Jd)

    def axis_forward(edges):
        return axis_angle_wigner(edges, lmax, generators=generators)

    # Optionally apply torch.compile
    if args.compile:
        print("Applying torch.compile to both methods...")
        euler_forward_fn = torch.compile(euler_forward)
        axis_forward_fn = torch.compile(axis_forward)
    else:
        euler_forward_fn = euler_forward
        axis_forward_fn = axis_forward

    # Correctness checks
    print("CORRECTNESS CHECKS")
    print("-" * 90)

    test_batch = 100
    torch.manual_seed(42)
    test_edges = torch.randn(test_batch, 3, dtype=dtype, device=device)
    test_edges = torch.nn.functional.normalize(test_edges, dim=-1)

    # Euler
    angles = init_edge_rot_euler_angles(test_edges)
    W_euler = eulers_to_wigner(angles, 0, lmax, Jd)
    euler_ok = check_wigner_correctness(W_euler, "Euler", test_batch, lmax, test_edges, device, dtype)

    # Axis-Angle
    W_axis, _ = axis_angle_wigner(test_edges, lmax)
    axis_ok = check_wigner_correctness(W_axis, "Axis-Angle", test_batch, lmax, test_edges, device, dtype)

    if not (euler_ok and axis_ok):
        print("\nWARNING: Some correctness checks failed!")
    print()

    # Forward pass benchmark
    print("FORWARD PASS ONLY")
    print("-" * 90)
    header = f"{'Batch':<8} {'Euler (ms)':<15} {'AxisAngle (ms)':<16} {'Ratio (E/A)':<12} {'Euler µs/edge':<15} {'AxisAng µs/edge':<15}"
    print(header)
    print("-" * 90)

    forward_results = []

    for n_edges in batch_sizes:
        torch.manual_seed(42)
        edges = torch.randn(n_edges, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)

        # Euler
        def euler_fn():
            return euler_forward_fn(edges)
        t_euler = benchmark_fn(euler_fn, n_warmup, n_runs, device)

        # Axis-Angle
        def axis_fn():
            return axis_forward_fn(edges)
        t_axis = benchmark_fn(axis_fn, n_warmup, n_runs, device)

        ratio = t_euler / t_axis if t_axis > 0 else float('inf')
        euler_per_edge = t_euler / n_edges * 1000  # µs
        axis_per_edge = t_axis / n_edges * 1000

        forward_results.append((n_edges, t_euler, t_axis, ratio))

        print(f"{n_edges:<8} {t_euler:<15.2f} {t_axis:<16.2f} {ratio:<12.2f} {euler_per_edge:<15.3f} {axis_per_edge:<15.3f}")

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

        # Euler
        def euler_bwd_fn():
            edges = edges_base.clone().requires_grad_(True)
            W = euler_forward_fn(edges)
            torch.autograd.grad(W.sum(), edges)
        t_euler = benchmark_fn(euler_bwd_fn, n_warmup, n_runs, device)

        # Axis-Angle
        def axis_bwd_fn():
            edges = edges_base.clone().requires_grad_(True)
            W, _ = axis_forward_fn(edges)
            torch.autograd.grad(W.sum(), edges)
        t_axis = benchmark_fn(axis_bwd_fn, n_warmup, n_runs, device)

        ratio = t_euler / t_axis if t_axis > 0 else float('inf')
        euler_per_edge = t_euler / n_edges * 1000
        axis_per_edge = t_axis / n_edges * 1000

        backward_results.append((n_edges, t_euler, t_axis, ratio))

        print(f"{n_edges:<8} {t_euler:<15.2f} {t_axis:<16.2f} {ratio:<12.2f} {euler_per_edge:<15.3f} {axis_per_edge:<15.3f}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("\nRatio interpretation: E/A = Euler time / AxisAngle time")
    print("  > 1.0 means Axis-Angle is faster")
    print("  < 1.0 means Euler is faster")
    print()

    # Summary statistics
    fwd_ratios = [r[3] for r in forward_results]
    bwd_ratios = [r[3] for r in backward_results]

    print("Forward pass:")
    print(f"  Ratio range: {min(fwd_ratios):.2f} - {max(fwd_ratios):.2f}")
    print(f"  Average ratio: {sum(fwd_ratios)/len(fwd_ratios):.2f}")

    print("\nForward+Backward pass:")
    print(f"  Ratio range: {min(bwd_ratios):.2f} - {max(bwd_ratios):.2f}")
    print(f"  Average ratio: {sum(bwd_ratios)/len(bwd_ratios):.2f}")

    print()


if __name__ == "__main__":
    main()
