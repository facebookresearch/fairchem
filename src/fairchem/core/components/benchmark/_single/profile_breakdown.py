"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

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


class OverheadProfileRunner(Runner):
    """
    Profile the forward pass overhead breakdown for A2A vs BL graph parallel.

    Uses torch.profiler with key_averages() to measure time in each phase:
    graph generation, per-layer communication, edge computation, etc.
    """

    def __init__(
        self,
        model_checkpoint: str | list[str] = "uma-s-1p2",
        natoms_list: list[int] | None = None,
        device: str = "cuda",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        warmup_steps: int = 10,
        measure_steps: int = 20,
        dataset_name: str = "oc20",
    ):
        self.model_checkpoint = model_checkpoint
        self.natoms_list = natoms_list or [4000]
        self.device = device
        self.overrides = overrides or {}
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.dataset_name = dataset_name

        if inference_settings is None:
            self.inference_settings = inference_settings_default()
        else:
            self.inference_settings = inference_settings

    def run(self) -> Any:
        run_dir = self.job_config.metadata.results_dir
        os.makedirs(run_dir, exist_ok=True)

        rank = distutils.get_rank()
        world_size = distutils.get_world_size()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        checkpoint = self.model_checkpoint
        if isinstance(checkpoint, list):
            checkpoint = checkpoint[0]

        is_a2a = self.overrides.get("backbone", {}).get("use_all_to_all_gp", False)
        mode = "a2a" if is_a2a else "bl"

        logging.info(
            f"Rank {rank}: Profiling {mode} at {world_size} GPUs, "
            f"atoms_list={self.natoms_list}"
        )

        from_ase_kwargs = {"task_name": self.dataset_name}

        for num_atoms in self.natoms_list:
            atoms = get_fcc_crystal_by_num_atoms(num_atoms)
            data = AtomicData.from_ase(atoms, **from_ase_kwargs)

            predictor = MLIPPredictUnit(
                checkpoint,
                self.device,
                overrides=self.overrides,
                inference_settings=self.inference_settings,
            )

            logging.info(
                f"Rank {rank}: num_atoms={num_atoms}, warming up "
                f"{self.warmup_steps} steps..."
            )

            for _ in range(self.warmup_steps):
                predictor.predict(data)
                dist.barrier()

            logging.info(f"Rank {rank}: Measuring {self.measure_steps} steps...")

            from torch.profiler import ProfilerActivity, profile, schedule

            all_timings = []

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=1, warmup=2, active=self.measure_steps, repeat=1
                ),
                record_shapes=False,
                with_stack=False,
            ) as prof:
                for step in range(3 + self.measure_steps):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    predictor.predict(data)
                    end_event.record()
                    torch.cuda.synchronize()
                    step_ms = start_event.elapsed_time(end_event)
                    if step >= 3:
                        all_timings.append(step_ms)
                    dist.barrier()
                    prof.step()

            avg_step_ms = np.mean(all_timings)
            std_step_ms = np.std(all_timings)

            key_avgs = prof.key_averages()

            # Categories to extract
            categories = {
                "generate_graph": [],
                "a2a_partition": [],
                "a2a_collect": [],
                "allgather_collect": [],
                "SO2Conv": [],
                "edgewise": [],
                "atomwise": [],
                "message passing": [],
                "layer_radial_emb": [],
                "sparse_index_exchange": [],
                "fused_index_exchange": [],
                "a2a_collect_p2p": [],
                "a2a_collect_async": [],
                "local_edges": [],
                "boundary_edges": [],
                "final_barrier": [],
            }

            for evt in key_avgs:
                for cat_key in categories:
                    if cat_key.lower() in evt.key.lower():
                        categories[cat_key].append(
                            {
                                "key": evt.key,
                                "cpu_time_ms": evt.cpu_time_total / 1000,
                                "cuda_time_ms": evt.cuda_time_total / 1000,
                                "count": evt.count,
                            }
                        )

            result = {
                "mode": mode,
                "world_size": world_size,
                "num_atoms": num_atoms,
                "atoms_per_rank": (
                    num_atoms // world_size if world_size > 1 else num_atoms
                ),
                "avg_step_ms": round(avg_step_ms, 2),
                "std_step_ms": round(std_step_ms, 2),
                "all_step_timings_ms": [round(t, 2) for t in all_timings],
                "breakdown": {},
            }

            for cat_key, events in categories.items():
                if events:
                    total_cuda = sum(e["cuda_time_ms"] for e in events)
                    total_cpu = sum(e["cpu_time_ms"] for e in events)
                    total_count = sum(e["count"] for e in events)
                    result["breakdown"][cat_key] = {
                        "per_step_cuda_ms": round(total_cuda / self.measure_steps, 3),
                        "per_step_cpu_ms": round(total_cpu / self.measure_steps, 3),
                        "total_count": total_count,
                        "count_per_step": total_count // self.measure_steps,
                    }

            if rank == 0:
                logging.info("=" * 70)
                logging.info(
                    f"PROFILE: {mode} | {world_size} GPUs | "
                    f"{num_atoms} atoms "
                    f"({num_atoms // max(world_size, 1)} per rank)"
                )
                logging.info(f"Step time: {avg_step_ms:.2f} ± {std_step_ms:.2f} ms")
                logging.info("=" * 70)
                logging.info(
                    f"{'Category':<30} {'CUDA ms/step':>14} "
                    f"{'CPU ms/step':>14} {'Calls/step':>12}"
                )
                logging.info("-" * 70)
                for cat_key in sorted(
                    result["breakdown"],
                    key=lambda k: result["breakdown"][k]["per_step_cuda_ms"],
                    reverse=True,
                ):
                    d = result["breakdown"][cat_key]
                    pct = (
                        d["per_step_cuda_ms"] / avg_step_ms * 100
                        if avg_step_ms > 0
                        else 0
                    )
                    logging.info(
                        f"{cat_key:<30} "
                        f"{d['per_step_cuda_ms']:>10.3f} ms "
                        f"{d['per_step_cpu_ms']:>10.3f} ms "
                        f"{d['count_per_step']:>8}x  "
                        f"({pct:.1f}%)"
                    )
                logging.info("=" * 70)

                logging.info("\n--- Top 30 CUDA operations ---")
                logging.info(
                    key_avgs.table(
                        sort_by="cuda_time_total",
                        row_limit=30,
                    )
                )

                output_file = os.path.join(
                    run_dir,
                    f"profile_{mode}_{world_size}gpu_{num_atoms}atoms.json",
                )
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                logging.info(f"Saved profile to {output_file}")

            del predictor
            torch.cuda.empty_cache()

        logging.info(f"Rank {rank}: Profiling complete.")

    def save_state(self, checkpoint_location, is_preemption=False):
        return False

    def load_state(self, checkpoint_location):
        return
