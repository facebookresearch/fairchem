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
    "triton_scatter_add",
    "triton_atomic",
    "triton_pytorch_bwd",
    "triton_recompute",
        "triton_so2_and_rotate_in",
    "triton_so2_and_rotate_in_outfused",
    "triton_rotate_in",
    "triton_so2_and_rotate_in_all_fused",
    "triton_so2_and_rotate_in_emit_unified_radial"
]


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_settings(mode):
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=True,
        internal_graph_gen_version=2,
        execution_mode=mode,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare forces across execution modes"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="benchmark_logs")
    parser.add_argument("--natoms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create atoms deterministically, without PBC
    seed_everywhere(args.seed)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    atoms.pbc = [False, False, False]
    print(f"Created {len(atoms)} atoms, pbc={atoms.pbc.tolist()}")

    forces_dict = {}
    data = None

    for mode in MODES:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print(f"{'='*50}")

        predictor = MLIPPredictUnit(
            args.checkpoint,
            "cuda",
            inference_settings=make_settings(mode),
        )

        # Build graph once using first predictor's backbone params
        if data is None:
            max_neighbors = predictor.model.module.backbone.max_neighbors
            cutoff = predictor.model.module.backbone.cutoff
            print(f"Building graph: max_neighbors={max_neighbors}, " f"cutoff={cutoff}")
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

        output = predictor.predict(data.clone())
        forces = output["forces"].detach().cpu().numpy()
        forces_dict[mode] = forces

        outpath = os.path.join(args.output_dir, f"{mode}_forces.npy")
        np.save(outpath, forces)
        print(f"Saved forces shape={forces.shape} -> {outpath}")

        del predictor
        torch.cuda.empty_cache()

    # Print MAE comparison table
    baseline = forces_dict["general"]
    print(f"\n{'='*60}")
    print("Force MAE vs baseline (general)")
    print(f"{'='*60}")
    print(f"{'Mode':<25} {'MAE (eV/A)':<15} {'Max Error':<15}")
    print(f"{'-'*55}")
    for mode in MODES:
        if mode == "general":
            print(f"{'general (baseline)':<25} {'---':<15} {'---':<15}")
            continue
        diff = np.abs(forces_dict[mode] - baseline)
        mae = np.mean(diff)
        max_err = np.max(diff)
        print(f"{mode:<25} {mae:<15.6e} {max_err:<15.6e}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
