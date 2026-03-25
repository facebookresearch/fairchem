"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Full UMA-S-1p2 inference benchmark comparing MOLE backends.
# Loads the model with each mole_layer_type (pytorch, dgl, grouped_gemm)
# and measures end-to-end inference throughput on batched inputs.
#
# Usage:
#     source .venv/bin/activate
#     python benchmarks/bench_mole_inference.py --num-systems 4 8 16 32
import argparse
import gc
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_single_gpu():
    """Initialize minimal distributed env for single-GPU inference."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def make_batched_input(
    predictor,
    num_systems,
    atoms_per_system,
    variable=False,
    seed=42,
    dataset_name="omat",
):
    """Create a batched AtomicData input with num_systems systems.

    Args:
        variable: If True, sample atom counts from a distribution around
            atoms_per_system (range: 0.3x to 1.7x) to simulate realistic
            variable-size batching.
    """
    import random

    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms

    max_neighbors = predictor.model.module.backbone.max_neighbors
    cutoff = predictor.model.module.backbone.cutoff

    rng = random.Random(seed)
    data_list = []
    sizes = []
    for _ in range(num_systems):
        if variable:
            natoms = max(4, int(rng.gauss(atoms_per_system, atoms_per_system * 0.4)))
        else:
            natoms = atoms_per_system
        sizes.append(natoms)
        atoms = get_fcc_crystal_by_num_atoms(natoms)
        data_object = AtomicData.from_ase(
            atoms,
            max_neigh=max_neighbors,
            radius=cutoff,
            r_edges=True,
            task_name=dataset_name,
        )
        data_object.natoms = torch.tensor(len(atoms))
        data_object.charge = torch.LongTensor([0])
        data_object.spin = torch.LongTensor([0])
        data_object.pos.requires_grad = True
        data_list.append(data_object)

    return atomicdata_list_to_batch(data_list), sizes


def benchmark_predictor(predictor, data, warmup=10, iters=50):
    """Benchmark a predictor, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        predictor.predict(data)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict(data)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    mean = np.mean(times)
    return median, p10, p90, mean


def _patch_mole_modules():
    """Patch installed mole/mole_utils modules with source tree versions.

    The installed package may not have grouped_gemm support. We patch
    only the MOLE modules from our source tree so the rest of the
    installed fairchem remains compatible with the checkpoint format.
    """
    import importlib
    import sys

    src = "/home/misko/fairchem_nvmath/src"
    mole_path = f"{src}/fairchem/core/models/uma/nn/mole.py"
    utils_path = f"{src}/fairchem/core/models/uma/nn/mole_utils.py"

    # Reload mole.py from source
    spec = importlib.util.spec_from_file_location(
        "fairchem.core.models.uma.nn.mole", mole_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fairchem.core.models.uma.nn.mole"] = mod
    spec.loader.exec_module(mod)

    # Reload mole_utils.py from source
    spec2 = importlib.util.spec_from_file_location(
        "fairchem.core.models.uma.nn.mole_utils", utils_path
    )
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["fairchem.core.models.uma.nn.mole_utils"] = mod2
    spec2.loader.exec_module(mod2)

    # Also patch escn_moe since it imports from mole/mole_utils at import time
    moe_path = f"{src}/fairchem/core/models/uma/escn_moe.py"
    spec3 = importlib.util.spec_from_file_location(
        "fairchem.core.models.uma.escn_moe", moe_path
    )
    mod3 = importlib.util.module_from_spec(spec3)
    sys.modules["fairchem.core.models.uma.escn_moe"] = mod3
    spec3.loader.exec_module(mod3)


# Patch before any fairchem imports
_patch_mole_modules()


def load_predictor(checkpoint, mole_layer_type):
    """Load a fresh MLIPPredictUnit with the given mole_layer_type."""
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=False,  # keep MOLE active so backend matters
        compile=False,
        external_graph_gen=True,  # pre-compute graph to isolate model time
    )

    # Pass mole_layer_type as a backbone override
    overrides = {"backbone": {"moe_layer_type": mole_layer_type}}

    logging.info(f"Loading model with mole_layer_type={mole_layer_type}")
    predictor = MLIPPredictUnit(
        checkpoint,
        device="cuda",
        inference_settings=settings,
        overrides=overrides,
    )
    return predictor


def _run_scenario(args, scenario_name, backends, variable):
    """Run one scenario (fixed or variable) and return results dict."""
    results = {}

    for backend in backends:
        try:
            predictor = load_predictor(args.checkpoint, backend)
        except Exception as e:
            logging.error(f"Failed to load {backend}: {e}")
            continue

        results[backend] = {}

        for nsys in args.num_systems:
            data, sizes = make_batched_input(
                predictor, nsys, args.atoms_per_system, variable=variable
            )
            total_atoms = data.natoms.sum().item()
            size_desc = (
                f"{sizes}" if variable and nsys <= 8 else f"~{np.mean(sizes):.0f}"
            )

            logging.info(
                f"Benchmarking {backend}: {nsys} systems, "
                f"sizes={size_desc}, total={total_atoms} atoms"
            )

            try:
                median, p10, p90, mean = benchmark_predictor(
                    predictor, data, warmup=args.warmup, iters=args.iters
                )
                results[backend][nsys] = (median, p10, p90, mean)
                logging.info(
                    f"  {backend} {nsys}sys: {median:.2f} ms " f"[{p10:.2f}-{p90:.2f}]"
                )
            except Exception as e:
                logging.error(f"  {backend} {nsys}sys: FAILED - {e}")

        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _print_results(scenario_name, backends, num_systems, results):
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print(f"RESULTS — {scenario_name} (median ms)")
    print(f"{'='*80}")

    header = f"{'Systems':>10s}"
    for backend in backends:
        if backend in results:
            header += f"  {backend:>20s}"
    print(header)
    print("-" * len(header))

    for nsys in num_systems:
        row = f"{nsys:>10d}"
        baseline_median = None
        for backend in backends:
            if backend not in results or nsys not in results[backend]:
                row += f"  {'N/A':>20s}"
                continue
            median, p10, p90, _ = results[backend][nsys]
            if baseline_median is None:
                baseline_median = median
            speedup = baseline_median / median if median > 0 else 0
            row += f"  {median:7.2f} ({speedup:.2f}x)"
        print(row)


def run_benchmark(args):
    setup_single_gpu()
    torch.set_float32_matmul_precision("high")

    backends = args.backends

    print(f"\n{'='*80}")
    print("UMA-S-1p2 Full Inference Benchmark")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backends: {backends}")
    print(f"Systems: {args.num_systems}, Atoms/system: {args.atoms_per_system}")
    print(f"{'='*80}")

    # Fixed-size scenario
    fixed_results = _run_scenario(
        args,
        f"Fixed {args.atoms_per_system} atoms/sys",
        backends,
        variable=False,
    )
    _print_results(
        f"Fixed {args.atoms_per_system} atoms/system",
        backends,
        args.num_systems,
        fixed_results,
    )

    # Variable-size scenario
    var_results = _run_scenario(
        args,
        f"Variable ~{args.atoms_per_system} atoms/sys",
        backends,
        variable=True,
    )
    _print_results(
        f"Variable ~{args.atoms_per_system} atoms/system",
        backends,
        args.num_systems,
        var_results,
    )

    print()


def main():
    default_ckpt = (
        "/home/misko/.cache/fairchem/models--facebook--UMA/"
        "snapshots/9e0d80ebc07f0c777e14d53781e1a7dcb2fd8561/"
        "checkpoints/uma-s-1p2.pt"
    )

    parser = argparse.ArgumentParser(
        description="Benchmark UMA-S-1p2 inference with different MOLE backends"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to UMA-S-1p2 checkpoint",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch", "dgl", "grouped_gemm"],
        help="MOLE backends to compare",
    )
    parser.add_argument(
        "--num-systems",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Batch sizes (number of systems)",
    )
    parser.add_argument(
        "--atoms-per-system",
        type=int,
        default=50,
        help="Atoms per system",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
