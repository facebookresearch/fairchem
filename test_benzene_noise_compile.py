"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Same as test_benzene_noise.py, but with compile=True. Verifies that
varying num_edges across frames does NOT trigger recompiles when B
is held fixed at 1.

Run with TORCH_LOGS=recompiles to see any recompile events.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from ase.build import molecule

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def build_batch(atoms):
    a = atoms.copy()
    a.info = {"charge": 0, "spin": 1}
    data = AtomicData.from_ase(
        a, task_name="omol", r_edges=True, radius=6.0, max_neigh=300
    )
    return atomicdata_list_to_batch([data])


def time_predict(predictor, batch):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = predictor.predict(batch)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=False,
        compile=True,
        external_graph_gen=False,
        execution_mode="umas_fast_gpu_mixed",
    )
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    base = molecule("C6H6")
    base.info = {"charge": 0, "spin": 1}
    base.center(vacuum=4.0)
    rng = np.random.default_rng(0)

    print("compile=True  execution_mode=umas_fast_gpu_mixed")
    print(f"benzene: {len(base)} atoms\n")

    # Warmup so phase-1 frame 0 isn't dominated by initial compile
    warm = build_batch(base)
    print("warmup (initial compile)...")
    t0 = time.perf_counter()
    _ = time_predict(predictor, warm)
    print(f"warmup wall: {time.perf_counter() - t0:.2f}s")
    _ = time_predict(predictor, warm)

    print(f"\n{'phase':<8} {'frame':<6} {'noise':<8} {'edges':<8} {'wall (ms)':<12}")

    def run_phase(name, sigma):
        atoms = base.copy()
        for i in range(10):
            atoms.positions = atoms.positions + rng.normal(
                0.0, sigma, size=atoms.positions.shape
            )
            batch = build_batch(atoms)
            wall = time_predict(predictor, batch)
            n_edges = int(batch.edge_index.shape[1])
            print(
                f"{name:<8} {i:<6} {sigma:<8.2f} {n_edges:<8} "
                f"{wall * 1000:<12.3f}"
            )

    run_phase("phase1", 0.1)
    run_phase("phase2", 1.0)


if __name__ == "__main__":
    main()
