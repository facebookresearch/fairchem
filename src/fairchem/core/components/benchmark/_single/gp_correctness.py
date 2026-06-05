"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GP correctness runner: compares single-GPU reference against
all-gather and all-to-all graph parallel modes.

Usage:
    fairchem -c configs/uma/correctness/gp-correctness.yaml job=local_8gpu
"""

from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import torch

from fairchem.core.common import distutils
from fairchem.core.components.runner import Runner
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ENERGY_TOL = 5e-4
FORCE_TOL = 1e-4
STRESS_TOL = 5e-4


def _seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GPCorrectnessRunner(Runner):
    def __init__(
        self,
        model_checkpoint: str,
        natoms_list: list[int] | None = None,
        seed: int = 42,
        device: str = "cuda",
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa: B008
        overrides: dict | None = None,
    ):
        self.model_checkpoint = model_checkpoint
        self.natoms_list = natoms_list or [10, 50, 100]
        self.seed = seed
        self.device = device
        self.inference_settings = inference_settings
        self.overrides = overrides or {}

    def run(self) -> None:
        run_dir = self.job_config.metadata.results_dir
        os.makedirs(run_dir, exist_ok=True)

        rank = distutils.get_rank()
        world_size = distutils.get_world_size()

        gp_modes = [
            ("allgather", {}),
            (
                "a2a_spatial",
                {"use_all_to_all_gp": True, "gp_partition_strategy": "spatial"},
            ),
            (
                "a2a_index_split",
                {"use_all_to_all_gp": True, "gp_partition_strategy": "index_split"},
            ),
        ]

        results = []

        for num_atoms in self.natoms_list:
            _seed_everywhere(self.seed)
            atoms = get_fcc_crystal_by_num_atoms(num_atoms)
            data = AtomicData.from_ase(atoms, task_name="omat")

            # Reference: single-rank prediction (no GP overrides)
            _seed_everywhere(self.seed)
            ref_predictor = MLIPPredictUnit(
                self.model_checkpoint,
                self.device,
                overrides=self.overrides,
                inference_settings=self.inference_settings,
            )
            ref_result = ref_predictor.predict(data)
            ref_energy = ref_result["energy"].detach().cpu()
            ref_forces = ref_result["forces"].detach().cpu()
            ref_stress = ref_result["stress"].detach().cpu()
            del ref_predictor
            torch.cuda.empty_cache()

            for mode_name, gp_overrides in gp_modes:
                merged = {**self.overrides}
                if "backbone" not in merged:
                    merged["backbone"] = {}
                merged["backbone"].update(gp_overrides)

                _seed_everywhere(self.seed)
                predictor = MLIPPredictUnit(
                    self.model_checkpoint,
                    self.device,
                    overrides=merged,
                    inference_settings=self.inference_settings,
                )
                gp_result = predictor.predict(data)
                gp_energy = gp_result["energy"].detach().cpu()
                gp_forces = gp_result["forces"].detach().cpu()
                gp_stress = gp_result["stress"].detach().cpu()
                del predictor
                torch.cuda.empty_cache()

                energy_diff = torch.abs(gp_energy - ref_energy).max().item()
                forces_diff = torch.abs(gp_forces - ref_forces).max().item()
                stress_diff = torch.abs(gp_stress - ref_stress).max().item()

                energy_ok = energy_diff < ENERGY_TOL
                forces_ok = forces_diff < FORCE_TOL
                stress_ok = stress_diff < STRESS_TOL
                all_ok = energy_ok and forces_ok and stress_ok

                result = {
                    "num_atoms": num_atoms,
                    "world_size": world_size,
                    "mode": mode_name,
                    "energy_diff": energy_diff,
                    "forces_diff": forces_diff,
                    "stress_diff": stress_diff,
                    "energy_ok": energy_ok,
                    "forces_ok": forces_ok,
                    "stress_ok": stress_ok,
                    "pass": all_ok,
                }
                results.append(result)

                status = "PASS" if all_ok else "FAIL"
                logger.info(
                    f"[{status}] {mode_name} | {num_atoms} atoms | "
                    f"ws={world_size} | "
                    f"energy_diff={energy_diff:.2e} | "
                    f"forces_diff={forces_diff:.2e} | "
                    f"stress_diff={stress_diff:.2e}"
                )

        # Write results JSON (rank 0 only)
        if rank == 0:
            output = {
                "world_size": world_size,
                "model": self.model_checkpoint,
                "tolerances": {
                    "energy": ENERGY_TOL,
                    "forces": FORCE_TOL,
                    "stress": STRESS_TOL,
                },
                "results": results,
            }
            results_path = os.path.join(run_dir, "correctness_results.json")
            with open(results_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results written to {results_path}")

            # Summary
            n_pass = sum(1 for r in results if r["pass"])
            n_total = len(results)
            logger.info(f"Summary: {n_pass}/{n_total} passed")

            if n_pass < n_total:
                failures = [r for r in results if not r["pass"]]
                for f in failures:
                    logger.error(
                        f"FAILED: {f['mode']} | {f['num_atoms']} atoms | "
                        f"energy={f['energy_diff']:.2e} "
                        f"forces={f['forces_diff']:.2e} "
                        f"stress={f['stress_diff']:.2e}"
                    )
                raise AssertionError(
                    f"{n_total - n_pass}/{n_total} GP correctness checks failed"
                )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
