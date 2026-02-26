#!/usr/bin/env python
"""
CPU Benchmark: Compare execution backends on CPU.

This compares:
  - general (baseline)
  - general+compile (may fail with Inductor scatter bug)
  - umas_fast_pytorch (block GEMM)
  - umas_fast_pytorch+compile (expected 1.78x speedup)

System: Si FCC crystal atoms, NO PBC
Reference: cpu_feb20_benchmark.md from fairchem_cleanup

Usage:
  CUDA_VISIBLE_DEVICES="" python benchmark_cpu.py --checkpoint /path/to/ckpt.pt
  CUDA_VISIBLE_DEVICES="" python benchmark_cpu.py --checkpoint /path/to/ckpt.pt --threads 32
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


CONFIGS = [
    ("general", False),
    ("general+compile", True),
    ("umas_fast_pytorch", False),
    ("umas_fast_pytorch+compile", True),
]


def make_settings(mode, compile_model):
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    # Parse mode - strip +compile suffix
    exec_mode = mode.replace("+compile", "")

    return InferenceSettings(
        tf32=False,  # CPU doesn't support TF32
        activation_checkpointing=False,
        merge_mole=True,
        compile=compile_model,
        external_graph_gen=True,
        internal_graph_gen_version=2,
        execution_mode=exec_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark for execution backends")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--natoms", type=int, default=1024, help="Approx number of atoms")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--configs", nargs="+", default=None, help="Specific configs to run")
    args = parser.parse_args()

    # Force CPU
    if torch.cuda.is_available():
        print("WARNING: CUDA is available. Set CUDA_VISIBLE_DEVICES='' for pure CPU benchmark")

    # Set threads
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"Using {torch.get_num_threads()} CPU threads")

    # Import after setting threads
    from fairchem.core.datasets.atomic_data import (
        AtomicData,
        atomicdata_list_to_batch,
    )
    from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
    from fairchem.core.units.mlip_unit import MLIPPredictUnit

    # Create atoms deterministically
    seed_everywhere(args.seed)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    atoms.pbc = [False, False, False]
    print(f"Created {len(atoms)} atoms, pbc={atoms.pbc.tolist()}")

    configs = CONFIGS
    if args.configs:
        configs = [(name, compile) for name, compile in CONFIGS if name in args.configs]

    results = {}
    baseline_time = None
    data = None

    print(f"\n{'=' * 70}")
    print(f"CPU Benchmark: {len(atoms)} atoms, warmup={args.warmup}, iters={args.iters}")
    print(f"{'=' * 70}")

    for mode, compile_model in configs:
        print(f"\n[{mode}]")
        print("-" * 40)

        try:
            settings = make_settings(mode, compile_model)
            predictor = MLIPPredictUnit(
                args.checkpoint,
                "cpu",
                inference_settings=settings,
            )
        except Exception as e:
            print(f"  SKIP: Failed to load model: {e}")
            results[mode] = {"status": "FAILED", "error": str(e)}
            continue

        # Build graph once using first predictor's backbone params
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
            data_obj.pos.requires_grad = True
            data = atomicdata_list_to_batch([data_obj])
            data = data.to("cpu")
            print(f"Graph edges: {data.edge_index.shape[1]}")

        # Warmup
        print(f"  Warmup ({args.warmup} iters)...", end=" ", flush=True)
        try:
            for _ in range(args.warmup):
                _ = predictor.model(data)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            results[mode] = {"status": "FAILED", "error": str(e)}
            continue

        # Benchmark
        print(f"  Benchmark ({args.iters} iters)...", end=" ", flush=True)
        try:
            start = time.perf_counter()
            for _ in range(args.iters):
                result = predictor.model(data)
            elapsed = (time.perf_counter() - start) / args.iters * 1000
            print(f"done")

            # Record baseline
            if baseline_time is None:
                baseline_time = elapsed

            speedup = baseline_time / elapsed
            # Energy is nested under task head, e.g. omat_energy/energy
            energy = result.get("omat_energy", {}).get("energy", torch.tensor([float('nan')]))
            results[mode] = {
                "status": "OK",
                "time_ms": elapsed,
                "speedup": speedup,
                "energy": energy.item(),
            }

            print(f"  Time: {elapsed:.1f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Energy: {energy.item():.6f}")

        except Exception as e:
            print(f"FAILED: {e}")
            results[mode] = {"status": "FAILED", "error": str(e)}
            continue

        # Clean up to avoid memory accumulation
        del predictor

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<30} {'Time (ms)':<12} {'Speedup':<10} {'Status':<10}")
    print("-" * 70)

    for mode, compile_model in configs:
        if mode in results:
            r = results[mode]
            if r["status"] == "OK":
                print(f"{mode:<30} {r['time_ms']:<12.1f} {r['speedup']:<10.2f}x {'OK':<10}")
            else:
                print(f"{mode:<30} {'---':<12} {'---':<10} {'FAILED':<10}")

    print(f"\n{'=' * 70}")
    print("Expected: umas_fast_pytorch+compile should be ~1.78x faster than general")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
