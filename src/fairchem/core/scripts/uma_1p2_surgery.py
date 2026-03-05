"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

UMA 1.2 Checkpoint Surgery Script

This script performs checkpoint surgery on UMA 1.2 to fix omat_rattle dataset support.
Stage 1: Add omat_rattle support (matches notebook)
Stage 2: Remove all omat_rattle mentions entirely
"""

from __future__ import annotations

import argparse
import os

import torch

TASKS = ["oc20", "oc22", "oc25", "omat", "odac", "omc", "omol"]


def add_omat_rattle_support(checkpoint):
    """Stage 1: Add omat_rattle support (matches notebook)."""
    dataset_mapping = {
        "oc20": "oc20",
        "oc22": "oc22",
        "oc25": "oc25",
        "omol": "omol",
        "omat": "omat",
        "omat_rattle": "omat",
        "odac": "odac",
        "omc": "omc",
    }

    del checkpoint.model_config["backbone"]["dataset_list"]
    checkpoint.model_config["backbone"]["dataset_mapping"] = dataset_mapping

    del checkpoint.model_config["heads"]["energyandforcehead"]["dataset_names"]
    checkpoint.model_config["heads"]["energyandforcehead"]["dataset_mapping"] = (
        dataset_mapping
    )

    checkpoint.model_state_dict[
        "backbone.dataset_embedding.dataset_emb_dict.omat_rattle.weight"
    ] = checkpoint.model_state_dict[
        "backbone.dataset_embedding.dataset_emb_dict.omat.weight"
    ].clone()

    checkpoint.ema_state_dict[
        "module.backbone.dataset_embedding.dataset_emb_dict.omat_rattle.weight"
    ] = checkpoint.ema_state_dict[
        "module.backbone.dataset_embedding.dataset_emb_dict.omat.weight"
    ].clone()

    checkpoint.model_config["model_id"] = "UMA-1.2"
    return checkpoint


def remove_omat_rattle(checkpoint):
    """Stage 2: Remove all omat_rattle mentions from config and weights.

    Key insight: In stage1, omat_rattle maps to 'omat' (shares omat's expert).
    The unique targets in stage1 are: ['oc20', 'oc22', 'oc25', 'odac', 'omat', 'omc', 'omol']
    These map to expert indices 0-6 via _build_expert_mapping (sorted unique values).

    However, the original checkpoint weights have 8 expert slots (indices 0-7),
    where index 7 is UNUSED/untrained. When we remove omat_rattle and have only
    7 datasets, the model expects 7 expert slots, so we must resize weights by
    removing the unused index 7.
    """
    dataset_mapping = {
        "oc20": "oc20",
        "oc22": "oc22",
        "oc25": "oc25",
        "omol": "omol",
        "omat": "omat",
        "odac": "odac",
        "omc": "omc",
    }

    checkpoint.model_config["backbone"]["dataset_mapping"] = dataset_mapping
    checkpoint.model_config["heads"]["energyandforcehead"]["dataset_mapping"] = (
        dataset_mapping
    )

    # Remove backbone embeddings for omat_rattle
    del checkpoint.model_state_dict[
        "backbone.dataset_embedding.dataset_emb_dict.omat_rattle.weight"
    ]
    del checkpoint.ema_state_dict[
        "module.backbone.dataset_embedding.dataset_emb_dict.omat_rattle.weight"
    ]

    # Resize head expert weights from 8 to 7 by removing UNUSED index 7
    # The original checkpoint has 8 slots, with index 7 never used during training.
    # Stage1 mapping: 0=oc20, 1=oc22, 2=oc25, 3=odac, 4=omat(+omat_rattle), 5=omc, 6=omol, 7=UNUSED
    # Stage2 mapping: same indices 0-6, just without omat_rattle entry
    head_keys = [
        "output_heads.energyandforcehead.head.energy_block.0.weights",
        "output_heads.energyandforcehead.head.energy_block.2.weights",
        "output_heads.energyandforcehead.head.energy_block.4.weights",
    ]
    for key in head_keys:
        w = checkpoint.model_state_dict[key]
        checkpoint.model_state_dict[key] = w[:7]  # keep indices 0-6, remove index 7

        ema_key = "module." + key
        w = checkpoint.ema_state_dict[ema_key]
        checkpoint.ema_state_dict[ema_key] = w[:7]  # keep indices 0-6, remove index 7

    # Remove omat_rattle tasks from tasks_config
    checkpoint.tasks_config = [
        t for t in checkpoint.tasks_config if "omat_rattle" not in t.get("datasets", [])
    ]

    # Add single atom support
    checkpoint.model_config["backbone"]["supports_single_atoms"] = True
    checkpoint.model_config["model_id"] = "UMA-1.2"
    return checkpoint


def create_test_systems():
    """Create PBC and non-PBC H2O test systems."""
    from ase.build import molecule

    # Non-PBC (aperiodic) H2O
    h2o_nopbc = molecule("H2O")
    h2o_nopbc.info["charge"] = 0
    h2o_nopbc.info["spin"] = 1

    # PBC H2O
    h2o_pbc = molecule("H2O")
    h2o_pbc.set_cell([10.0, 10.0, 10.0])
    h2o_pbc.set_pbc(True)

    return {"nopbc": h2o_nopbc, "pbc": h2o_pbc}


def compare_checkpoints(stage1_path: str, stage2_path: str):
    """Compare stage1 and stage2 checkpoints across all tasks."""
    import numpy as np

    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    systems = create_test_systems()

    print(
        f"{'Task':<8} {'System':<8} {'E1':<14} {'E2':<14} {'E_abs':<12} {'E_rel':<12} {'F_max_abs':<12} {'F_max_rel':<12}"
    )
    print("-" * 100)

    all_match = True
    for task in TASKS:
        calc1 = FAIRChemCalculator.from_model_checkpoint(stage1_path, task_name=task)
        calc2 = FAIRChemCalculator.from_model_checkpoint(stage2_path, task_name=task)

        for sys_name, atoms in systems.items():
            atoms1 = atoms.copy()
            atoms2 = atoms.copy()
            atoms1.calc = calc1
            atoms2.calc = calc2

            e1 = atoms1.get_potential_energy()
            e2 = atoms2.get_potential_energy()
            f1 = atoms1.get_forces()
            f2 = atoms2.get_forces()

            e_abs = abs(e1 - e2)
            e_rel = e_abs / abs(e1) if e1 != 0 else 0
            f_abs = np.abs(f1 - f2).max()
            f_rel = f_abs / np.abs(f1).max() if np.abs(f1).max() != 0 else 0

            e_match = e_abs < 1e-5
            f_match = f_abs < 1e-5

            print(
                f"{task:<8} {sys_name:<8} {e1:<14.6f} {e2:<14.6f} {e_abs:<12.2e} {e_rel:<12.2e} {f_abs:<12.2e} {f_rel:<12.2e}"
            )

            if not (e_match and f_match):
                all_match = False

    return all_match


def uma_1p2_surgery(checkpoint_path: str, output_dir: str) -> tuple[str, str]:
    """Perform checkpoint surgery and output stage1 and stage2 checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Add omat_rattle support
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint = add_omat_rattle_support(checkpoint)
    stage1_path = os.path.join(output_dir, "inference_ckpt_stage1.pt")
    torch.save(checkpoint, stage1_path)
    print(f"Stage 1 saved to {stage1_path}")

    # Stage 2: Remove omat_rattle entirely
    checkpoint = remove_omat_rattle(checkpoint)
    stage2_path = os.path.join(output_dir, "inference_ckpt_stage2.pt")
    torch.save(checkpoint, stage2_path)
    print(f"Stage 2 saved to {stage2_path}")

    return stage1_path, stage2_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform checkpoint surgery on UMA 1.2"
    )
    parser.add_argument("--checkpoint-in", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    stage1_path, stage2_path = uma_1p2_surgery(args.checkpoint_in, args.output_dir)
    print("\n=== Self-test: stage1 vs stage1 ===")
    self_match = compare_checkpoints(stage1_path, stage1_path)
    print(f"Self-test match: {self_match}")

    print("\n=== Comparing stage1 vs stage2 ===")
    all_match = compare_checkpoints(stage1_path, stage2_path)
    print(f"\nAll outputs match: {all_match}")
