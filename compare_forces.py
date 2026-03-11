"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import timeit

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.datasets.common_structures import (
    get_fcc_crystal_by_num_atoms,
    get_mixed_batch_systems,
    FCC_ELEMENTS,
)
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

MODES = [
    "general",
    "umas_fast_pytorch",
    "umas_fast_gpu",
]

# Default elements for mixed batch testing (different element per system)
DEFAULT_MIXED_ELEMENTS = ["Cu", "Ag", "Au", "Al", "Ni", "Pd", "Pt", "Pb"]


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_settings(mode, compile_model=False, merge_mole=True):
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=merge_mole,
        compile=compile_model,
        external_graph_gen=True,
        internal_graph_gen_version=2,
        execution_mode=mode,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare energy and forces across execution backends"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="benchmark_logs", help="Output dir")
    parser.add_argument("--natoms", type=int, default=2000, help="Number of atoms (single system mode)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Execution modes to test (default: all)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile",
    )
    # Mixed batch arguments
    parser.add_argument(
        "--mixed-batch",
        type=str,
        default=None,
        help="Comma-separated atom counts for mixed batch (e.g., '500,1000,2000'). "
             "Each system gets a different element.",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=None,
        help="Comma-separated elements for mixed batch (e.g., 'Cu,Ag,Au'). "
             "Must match --mixed-batch length. Default: cycles through FCC metals.",
    )
    # Benchmark timing arguments
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run timing benchmark to measure QPS",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--timeiters",
        type=int,
        default=10,
        help="Number of timed iterations (default: 10)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats (default: 3)",
    )
    parser.add_argument(
        "--moe-layer-type",
        type=str,
        default="pytorch",
        choices=["pytorch", "dgl"],
        help="MOLE layer type: 'pytorch' (default) or 'dgl' (uses fairchem_cpp segment_mm)",
    )
    args = parser.parse_args()

    modes = args.modes if args.modes else MODES
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everywhere(args.seed)

    # Determine if we're in mixed batch mode or single system mode
    if args.mixed_batch:
        natoms_list = [int(x.strip()) for x in args.mixed_batch.split(",")]
        if args.elements:
            elements = [x.strip() for x in args.elements.split(",")]
        else:
            # Cycle through default elements
            elements = [DEFAULT_MIXED_ELEMENTS[i % len(DEFAULT_MIXED_ELEMENTS)] 
                       for i in range(len(natoms_list))]
        
        atoms_list = get_mixed_batch_systems(natoms_list, elements=elements, seed=args.seed, pbc=False)
        total_atoms = sum(len(a) for a in atoms_list)
        print(f"Mixed batch mode: {len(atoms_list)} systems")
        for i, (atoms, elem) in enumerate(zip(atoms_list, elements)):
            print(f"  System {i}: {len(atoms)} atoms, element={elem}")
        print(f"  Total atoms: {total_atoms}")
    else:
        # Single system mode (original behavior)
        atoms = get_fcc_crystal_by_num_atoms(args.natoms)
        atoms.pbc = [False, False, False]
        atoms_list = [atoms]
        elements = ["C"]
        print(f"Single system mode: {len(atoms)} atoms, pbc={atoms.pbc.tolist()}")

    results = {}
    data = None

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"MOE layer type: {args.moe_layer_type}")
        print(f"{'=' * 60}")

        # Disable merge_mole for multi-system batches (not supported)
        use_merge_mole = len(atoms_list) == 1
        # Build overrides for moe_layer_type
        overrides = {"backbone": {"moe_layer_type": args.moe_layer_type}}
        try:
            predictor = MLIPPredictUnit(
                args.checkpoint,
                "cuda",
                inference_settings=make_settings(mode, args.compile, merge_mole=use_merge_mole),
                overrides=overrides,
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Build graph once using first predictor's backbone params
        if data is None:
            max_neighbors = predictor.model.module.backbone.max_neighbors
            cutoff = predictor.model.module.backbone.cutoff
            print(f"Building graph: max_neighbors={max_neighbors}, cutoff={cutoff}")
            
            data_objects = []
            total_edges = 0
            for atoms in atoms_list:
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
                data_objects.append(data_obj)
                total_edges += data_obj.edge_index.shape[1]
            
            data = atomicdata_list_to_batch(data_objects)
            print(f"Batched {len(data_objects)} systems, total edges: {total_edges}")

        # Run prediction
        torch.cuda.reset_peak_memory_stats()
        output = predictor.predict(data.clone())
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        energy = output["energy"].detach().cpu()
        forces = output["forces"].detach().cpu().numpy()

        # Benchmark timing if requested
        qps = None
        if args.benchmark:
            print(f"  Running benchmark: {args.warmup} warmup, {args.timeiters} iters x {args.repeats} repeats")
            
            def timefunc():
                predictor.predict(data.clone())
                torch.cuda.synchronize()
            
            # Warmup
            for _ in range(args.warmup):
                timefunc()
            
            # Timed runs
            times = timeit.repeat(timefunc, number=args.timeiters, repeat=args.repeats)
            avg_time = np.mean(times)
            std_time = np.std(times)
            qps = args.timeiters / avg_time
            print(f"  Timing: {times} (mean={avg_time:.3f}s, std={std_time:.3f}s)")
            print(f"  QPS: {qps:.2f}")

        results[mode] = {
            "energy": energy,
            "forces": forces,
            "peak_memory_gb": peak_mem_gb,
            "qps": qps,
        }

        # Save forces
        outpath = os.path.join(args.output_dir, f"{mode}_forces.npy")
        np.save(outpath, forces)
        
        # Print energy (handle both single and multi-system cases)
        if energy.numel() == 1:
            print(f"  Energy: {energy.item():.6f}")
        else:
            print(f"  Energies: {energy.tolist()}")
            print(f"  Total energy: {energy.sum().item():.6f}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Peak memory: {peak_mem_gb:.3f} GB")
        print(f"  Saved: {outpath}")

        del predictor
        torch.cuda.empty_cache()

    # Comparison table
    if "general" not in results:
        print("\nWARNING: 'general' baseline not available, skipping comparison.")
        return 0

    baseline_forces = results["general"]["forces"]
    baseline_energy = results["general"]["energy"]

    print(f"\n{'=' * 80}")
    print("Comparison vs baseline (general)")
    print(f"{'=' * 80}")
    print(
        f"{'Mode':<25} {'Energy Diff':<15} {'Force MAE':<15} "
        f"{'Force Max Err':<15} {'Peak Mem (GB)':<15}"
    )
    print(f"{'-' * 80}")

    for mode in modes:
        if mode not in results:
            print(f"{mode:<25} {'SKIPPED':<15}")
            continue

        r = results[mode]
        peak_mem = r["peak_memory_gb"]

        if mode == "general":
            print(
                f"{'general (baseline)':<25} {'---':<15} {'---':<15} "
                f"{'---':<15} {peak_mem:<15.3f}"
            )
            continue

        # Handle both single and multi-system energy comparisons
        if baseline_energy.numel() == 1:
            energy_diff = abs(r["energy"].item() - baseline_energy.item())
        else:
            # For multi-system, compute max per-system energy difference
            energy_diff = torch.abs(r["energy"] - baseline_energy).max().item()
        
        force_diff = np.abs(r["forces"] - baseline_forces)
        force_mae = np.mean(force_diff)
        force_max = np.max(force_diff)

        print(
            f"{mode:<25} {energy_diff:<15.6e} {force_mae:<15.6e} "
            f"{force_max:<15.6e} {peak_mem:<15.3f}"
        )

    print(f"{'=' * 80}")

    # Pass/fail summary with tolerances
    print(f"\n{'=' * 80}")
    print("Pass/Fail Summary (energy atol=1e-4, forces atol=1e-3)")
    print(f"{'=' * 80}")

    all_pass = True
    for mode in modes:
        if mode == "general" or mode not in results:
            continue

        r = results[mode]
        # Handle both single and multi-system energy comparisons
        if baseline_energy.numel() == 1:
            energy_diff = abs(r["energy"].item() - baseline_energy.item())
        else:
            energy_diff = torch.abs(r["energy"] - baseline_energy).max().item()
        force_max = np.max(np.abs(r["forces"] - baseline_forces))

        energy_ok = energy_diff < 1e-4
        forces_ok = force_max < 1e-3

        status = "PASS" if (energy_ok and forces_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        details = []
        if not energy_ok:
            details.append(f"energy_diff={energy_diff:.2e}")
        if not forces_ok:
            details.append(f"force_max_err={force_max:.2e}")

        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"  {mode:<25} {status}{detail_str}")

    print(f"{'=' * 80}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
