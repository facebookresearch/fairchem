"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Benchmark segment_mm implementations: baseline (pytorch loop), fairchem_cpp, nvmath grouped GEMM.
# Measures kernel-level performance for the MOLE segment matmul used in UMA-S-1p2.
#
# Usage:
#     source .venv/bin/activate
#     python benchmarks/bench_segment_mm.py
import argparse
import time

import torch


# ── Approach 1: Baseline PyTorch loop (same as MOLE.forward) ──
def segment_mm_pytorch(A, B, seglen):
    """Pure PyTorch loop: one F.linear per segment."""
    out = []
    off = 0
    for i in range(B.shape[0]):
        m = seglen[i].item()
        out.append(A[off : off + m] @ B[i])
        off += m
    return torch.cat(out)


# ── Approach 2: fairchem_cpp CUDA kernel ──
def segment_mm_fairchem_cpp(A, B, seglen):
    """fairchem_cpp CUDA extension (cuBLAS GemmEx loop in C++)."""
    import fairchem_cpp

    B_cast = B.to(A.dtype)
    return fairchem_cpp.ops.segment_mm(A, B_cast, seglen)


# ── Approach 3: nvmath cuBLAS grouped GEMM ──
_cublas_state = {"handle": None}


def _get_cublas_handle():
    if _cublas_state["handle"] is None:
        import nvmath.bindings.cublas as cublas

        _cublas_state["handle"] = cublas.create()
        cublas.set_stream(
            _cublas_state["handle"], torch.cuda.current_stream().cuda_stream
        )
        cublas.set_math_mode(_cublas_state["handle"], cublas.Math.TF32_TENSOR_OP_MATH)
    return _cublas_state["handle"]


def segment_mm_nvmath_grouped(A, B, seglen):
    """nvmath cuBLAS sgemm_grouped_batched — single kernel for all segments."""
    import numpy as np
    import nvmath.bindings.cublas as cublas

    handle = _get_cublas_handle()
    cublas.set_stream(handle, torch.cuda.current_stream().cuda_stream)

    sizes = seglen.tolist()
    num_seg = len(sizes)
    K = A.shape[1]
    N = B.shape[2]

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(A.shape[0], N, device=A.device, dtype=A.dtype)

    transa = np.zeros(num_seg, dtype=np.int32)
    transb = np.zeros(num_seg, dtype=np.int32)
    m_arr = np.full(num_seg, N, dtype=np.int32)
    n_arr = np.array(sizes, dtype=np.int32)
    k_arr = np.full(num_seg, K, dtype=np.int32)
    alpha_arr = np.full(num_seg, 1.0, dtype=np.float32)
    beta_arr = np.full(num_seg, 0.0, dtype=np.float32)
    lda_arr = np.full(num_seg, N, dtype=np.int32)
    ldb_arr = np.full(num_seg, K, dtype=np.int32)
    ldc_arr = np.full(num_seg, N, dtype=np.int32)
    group_size = np.ones(num_seg, dtype=np.int32)
    a_ptrs = np.empty(num_seg, dtype=np.int64)
    b_ptrs = np.empty(num_seg, dtype=np.int64)
    c_ptrs = np.empty(num_seg, dtype=np.int64)

    start = 0
    for i in range(num_seg):
        a_ptrs[i] = B[i].data_ptr()
        b_ptrs[i] = A[start].data_ptr()
        c_ptrs[i] = C[start].data_ptr()
        start += sizes[i]

    cublas.sgemm_grouped_batched(
        handle,
        transa.ctypes.data,
        transb.ctypes.data,
        m_arr.ctypes.data,
        n_arr.ctypes.data,
        k_arr.ctypes.data,
        alpha_arr.ctypes.data,
        a_ptrs.ctypes.data,
        lda_arr.ctypes.data,
        b_ptrs.ctypes.data,
        ldb_arr.ctypes.data,
        beta_arr.ctypes.data,
        c_ptrs.ctypes.data,
        ldc_arr.ctypes.data,
        num_seg,
        group_size.ctypes.data,
    )
    return C


# ── Approach 4: Padded BMM (pure PyTorch, no ext deps) ──
def segment_mm_padded_bmm(A, B, seglen):
    """Pad ragged segments to max_m, use torch.bmm, extract valid rows."""
    num_seg = B.shape[0]
    K = A.shape[1]
    N = B.shape[2]
    total_rows = A.shape[0]
    sizes = seglen.tolist()
    max_m = max(sizes)

    # Vectorized scatter-based padding
    seg_ids = torch.repeat_interleave(
        torch.arange(num_seg, device=A.device), seglen.to(A.device)
    )
    seg_starts = torch.zeros(num_seg, dtype=torch.long, device=A.device)
    seg_starts[1:] = torch.cumsum(seglen.to(A.device)[:-1], dim=0)
    offsets = torch.arange(total_rows, device=A.device) - seg_starts[seg_ids]
    pad_idx = seg_ids * max_m + offsets

    A_flat = A.new_zeros(num_seg * max_m, K)
    A_flat[pad_idx] = A
    A_padded = A_flat.view(num_seg, max_m, K)

    C_padded = torch.bmm(A_padded, B)
    C_flat = C_padded.reshape(num_seg * max_m, N)
    return C_flat[pad_idx]


def make_inputs(num_segments, atoms_per_segment, K, N, device="cuda"):
    """Create realistic segment_mm inputs.

    Args:
        num_segments: number of systems in batch
        atoms_per_segment: int or list of ints (atoms per system)
        K: input feature dim (sphere_channels)
        N: output feature dim
    """
    if isinstance(atoms_per_segment, int):
        sizes = [atoms_per_segment] * num_segments
    else:
        sizes = list(atoms_per_segment)
    total = sum(sizes)
    seglen = torch.tensor(sizes, dtype=torch.int32)
    A = torch.randn(total, K, device=device)
    B = torch.randn(num_segments, K, N, device=device)
    return A, B, seglen


def check_correctness(implementations, A, B, seglen):
    """Verify all implementations produce the same result."""
    ref = None
    ref_name = None
    for name, fn in implementations.items():
        try:
            result = fn(A, B, seglen)
            torch.cuda.synchronize()
            if ref is None:
                ref = result
                ref_name = name
            else:
                max_diff = (result - ref).abs().max().item()
                rel_diff = max_diff / ref.abs().mean().item()
                status = "PASS" if rel_diff < 1e-3 else "FAIL"
                print(
                    f"  {name} vs {ref_name}: max_diff={max_diff:.2e}, "
                    f"rel_diff={rel_diff:.2e} [{status}]"
                )
        except Exception as e:
            print(f"  {name}: ERROR - {e}")


def benchmark_one(fn, A, B, seglen, warmup=20, iters=100):
    """Benchmark a single implementation. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn(A, B, seglen)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(A, B, seglen)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    return median, p10, p90


def run_benchmark(args):
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    implementations = {
        "pytorch_loop": segment_mm_pytorch,
        "fairchem_cpp": segment_mm_fairchem_cpp,
        "nvmath_grouped": segment_mm_nvmath_grouped,
        "padded_bmm": segment_mm_padded_bmm,
    }

    # UMA-S-1p2 typical dimensions
    K = args.K
    N = args.N

    # Test scenarios: (description, num_segments, atoms_per_segment)
    scenarios = [
        (f"{nsys}sys x {natoms}atoms", nsys, natoms)
        for nsys in args.num_systems
        for natoms in args.atoms_per_system
    ]

    # Also test variable-size segments (realistic batching)
    if args.variable_sizes:
        import random

        random.seed(42)
        for nsys in args.num_systems:
            mean_atoms = args.atoms_per_system[0] if args.atoms_per_system else 50
            sizes = [
                max(1, int(random.gauss(mean_atoms, mean_atoms * 0.3)))
                for _ in range(nsys)
            ]
            scenarios.append(
                (f"{nsys}sys x ~{mean_atoms}atoms (variable)", nsys, sizes)
            )

    print(f"\n{'='*80}")
    print(f"Segment MM Benchmark — K={K}, N={N}, device={device}")
    print(f"{'='*80}")

    for desc, nsys, atoms in scenarios:
        A, B, seglen = make_inputs(nsys, atoms, K, N, device)
        total_rows = A.shape[0]

        print(f"\n--- {desc} (total_rows={total_rows}) ---")

        # Correctness check
        print("Correctness:")
        check_correctness(implementations, A, B, seglen)

        # Performance
        print("Performance (ms, median [p10-p90]):")
        for name, fn in implementations.items():
            try:
                median, p10, p90 = benchmark_one(
                    fn,
                    A,
                    B,
                    seglen,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                print(f"  {name:20s}: {median:8.3f} ms  [{p10:.3f} - {p90:.3f}]")
            except Exception as e:
                print(f"  {name:20s}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark segment_mm implementations")
    parser.add_argument(
        "--K", type=int, default=128, help="Input feature dim (default: 128)"
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Output feature dim (default: 128)"
    )
    parser.add_argument(
        "--num-systems",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Number of systems (segments) to test",
    )
    parser.add_argument(
        "--atoms-per-system",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500],
        help="Atoms per system to test",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--variable-sizes", action="store_true", help="Also test variable segment sizes"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
