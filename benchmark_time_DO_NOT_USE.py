"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Time benchmark for execution backends on Si crystal systems.

Benchmarks forward pass (and optionally backward pass) timing
across different execution modes.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

if TYPE_CHECKING:
    from ase import Atoms


def seed_everywhere(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_si_crystal(num_atoms: int, seed: int = 42) -> Atoms:
    """
    Create a Si diamond crystal with approximately num_atoms atoms.

    Args:
        num_atoms: Target number of atoms (actual will be close multiple of 8)
        seed: Random seed for reproducibility

    Returns:
        ASE Atoms object with Si diamond structure, no PBC
    """
    from ase.build import bulk

    # Si diamond has 2 atoms per primitive cell, 8 per conventional cubic cell
    # Calculate repetitions needed
    atoms_per_cell = 8
    n_cells = max(1, int(round((num_atoms / atoms_per_cell) ** (1 / 3))))

    # Build Si diamond crystal
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    si = si.repeat((n_cells, n_cells, n_cells))

    # Add small random perturbations for realism
    np.random.seed(seed)
    si.positions += np.random.randn(*si.positions.shape) * 0.05

    # Disable PBC
    si.pbc = [False, False, False]

    return si


def make_settings(mode: str, merge_mole: bool = False) -> InferenceSettings:
    """
    Create InferenceSettings for benchmark.

    Args:
        mode: Execution mode (general, umas_fast_pytorch, umas_fast_gpu)
        merge_mole: Whether to merge MoLE experts (False for non-MOLE models)

    Returns:
        InferenceSettings configured for benchmarking
    """
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=merge_mole,
        compile=False,
        external_graph_gen=True,
        internal_graph_gen_version=2,
        execution_mode=mode,
    )


def benchmark_forward(
    predictor: MLIPPredictUnit,
    data: AtomicData,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """
    Benchmark forward pass timing.

    Args:
        predictor: MLIPPredictUnit to benchmark
        data: Input AtomicData batch
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Dict with timing stats (mean, std, min, max in ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = predictor.predict(data.clone())
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = predictor.predict(data.clone())
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "iterations": iterations,
    }


def benchmark_forward_backward(
    predictor: MLIPPredictUnit,
    data: AtomicData,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """
    Benchmark forward + backward pass timing (forces via autograd).

    Args:
        predictor: MLIPPredictUnit to benchmark
        data: Input AtomicData batch (pos will be set to requires_grad)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Dict with timing stats (mean, std, min, max in ms)
    """
    # Warmup
    for _ in range(warmup):
        data_clone = data.clone()
        data_clone.pos.requires_grad = True
        output = predictor.predict(data_clone)
        loss = output["energy"].sum()
        loss.backward()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        data_clone = data.clone()
        data_clone.pos.requires_grad = True

        torch.cuda.synchronize()
        start = time.perf_counter()

        output = predictor.predict(data_clone)
        loss = output["energy"].sum()
        loss.backward()

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Time benchmark for execution backends"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--natoms", type=int, default=2000, help="Number of atoms")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["general", "umas_fast_pytorch", "umas_fast_gpu"],
        help="Execution modes to benchmark",
    )
    parser.add_argument(
        "--merge-mole",
        action="store_true",
        default=False,
        help="Enable merge_mole (for MOLE models)",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        default=False,
        help="Include backward pass (forces) in timing",
    )
    args = parser.parse_args()

    seed_everywhere(args.seed)

    # Create Si crystal
    print(f"Creating Si crystal with ~{args.natoms} atoms (no PBC)...")
    atoms = create_si_crystal(args.natoms, seed=args.seed)
    print(f"  Actual atoms: {len(atoms)}")
    print(f"  PBC: {atoms.pbc.tolist()}")

    # Build graph with first predictor
    data = None
    results = {}

    for mode in args.modes:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")

        try:
            predictor = MLIPPredictUnit(
                args.checkpoint,
                "cuda",
                inference_settings=make_settings(mode, args.merge_mole),
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Build graph once
        if data is None:
            max_neighbors = predictor.model.module.backbone.max_neighbors
            cutoff = predictor.model.module.backbone.cutoff
            print(f"Building graph: max_neighbors={max_neighbors}, cutoff={cutoff}")

            data_obj = AtomicData.from_ase(
                atoms,
                max_neigh=max_neighbors,
                radius=cutoff,
                r_edges=True,
                task_name="omat",
            )
            data_obj.natoms = torch.tensor(len(atoms))
            data_obj.charge = torch.LongTensor([0])
            data_obj.spin = torch.LongTensor([0])
            data = atomicdata_list_to_batch([data_obj]).to("cuda")
            print(f"  Edges: {data.edge_index.shape[1]}")

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        # Run benchmark
        if args.backward:
            print(
                f"Benchmarking forward+backward (warmup={args.warmup}, iter={args.iterations})..."
            )
            stats = benchmark_forward_backward(
                predictor, data, warmup=args.warmup, iterations=args.iterations
            )
        else:
            print(
                f"Benchmarking forward only (warmup={args.warmup}, iter={args.iterations})..."
            )
            stats = benchmark_forward(
                predictor, data, warmup=args.warmup, iterations=args.iterations
            )

        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        stats["peak_memory_gb"] = peak_mem_gb

        results[mode] = stats

        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        print(f"  Min:  {stats['min_ms']:.2f} ms")
        print(f"  Max:  {stats['max_ms']:.2f} ms")
        print(f"  Peak memory: {peak_mem_gb:.3f} GB")

        # Cleanup
        del predictor
        torch.cuda.empty_cache()
        gc.collect()

    # Summary table
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Atoms: {len(atoms)}, Edges: {data.edge_index.shape[1] if data else 'N/A'}")
    print(f"Benchmark: {'forward+backward' if args.backward else 'forward only'}")
    print(f"Settings: merge_mole={args.merge_mole}, tf32=True, compile=False")
    print(f"{'=' * 80}")
    print(
        f"{'Mode':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Peak Mem (GB)':<15} {'Speedup':<10}"
    )
    print(f"{'-' * 80}")

    baseline_time = results.get("general", {}).get("mean_ms", None)

    for mode in args.modes:
        if mode not in results:
            print(f"{mode:<25} {'SKIPPED':<12}")
            continue

        r = results[mode]
        speedup = baseline_time / r["mean_ms"] if baseline_time else 1.0
        speedup_str = f"{speedup:.2f}x" if mode != "general" else "1.00x (baseline)"

        print(
            f"{mode:<25} {r['mean_ms']:<12.2f} {r['std_ms']:<12.2f} "
            f"{r['peak_memory_gb']:<15.3f} {speedup_str:<10}"
        )

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
