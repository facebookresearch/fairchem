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

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.datasets.common_structures import (
    get_fcc_crystal_by_num_atoms,
)
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

MODES = [
    "general",
    "umas_fast_pytorch",
    "umas_fast_gpu",
]


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_settings(mode, compile_model=False):
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
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
    parser.add_argument("--natoms", type=int, default=2000, help="Number of atoms")
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
    args = parser.parse_args()

    modes = args.modes if args.modes else MODES
    os.makedirs(args.output_dir, exist_ok=True)

    # Create atoms deterministically, without PBC
    seed_everywhere(args.seed)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    atoms.pbc = [False, False, False]
    print(f"Created {len(atoms)} atoms, pbc={atoms.pbc.tolist()}")

    results = {}
    data = None

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")

        try:
            predictor = MLIPPredictUnit(
                args.checkpoint,
                "cuda",
                inference_settings=make_settings(mode, args.compile),
            )
        except Exception as e:
            print(f"  SKIP: {e}")
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
            print(f"Graph edges: {data.edge_index.shape[1]}")

        # Run prediction
        torch.cuda.reset_peak_memory_stats()
        output = predictor.predict(data.clone())
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        energy = output["energy"].detach().cpu()
        forces = output["forces"].detach().cpu().numpy()

        results[mode] = {
            "energy": energy,
            "forces": forces,
            "peak_memory_gb": peak_mem_gb,
        }

        # Save forces
        outpath = os.path.join(args.output_dir, f"{mode}_forces.npy")
        np.save(outpath, forces)
        print(f"  Energy: {energy.item():.6f}")
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

        energy_diff = abs(r["energy"].item() - baseline_energy.item())
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
        energy_diff = abs(r["energy"].item() - baseline_energy.item())
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
