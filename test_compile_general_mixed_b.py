"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Question: does torch.compile + execution_mode="general" tolerate
mixed batch sizes the same way umas_fast_gpu_mixed does?
"""

from __future__ import annotations

import time

import numpy as np
import torch
from ase.build import bulk

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from test_helpers import GRAPH_GEN, make_atomic_data, make_settings


def make_fcc(element, num_atoms=200, seed=42):
    atoms = bulk(element, "fcc", a=3.8)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(atoms), num_atoms, replace=False)
    a = atoms[indices]
    a.info = {"charge": 0, "spin": 0}
    return a


def main():
    settings = make_settings(compile=True, execution_mode="general")
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    elements = ["Cu", "Si", "Al", "Ge", "Pd", "Au", "Ni", "Pt"]
    seed = 0

    def make_batch(b):
        nonlocal seed
        data_list = []
        for _ in range(b):
            data_list.append(
                make_atomic_data(
                    make_fcc(elements[seed % len(elements)], 200, seed),
                    task_name="omat",
                )
            )
            seed += 1
        return atomicdata_list_to_batch(data_list)

    sequence = [1, 2, 3, 4, 5, 4, 2, 1, 8, 4]
    print(f"compile=True  graph_gen={GRAPH_GEN}  execution_mode=general")
    print(f"sequence: {sequence}\n")
    print(f"{'#':<3} {'B':<3} {'wall (s)':<10} {'note':<30}")

    seen_b: set[int] = set()
    for i, b in enumerate(sequence):
        batch = make_batch(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predictor.predict(batch)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        first = b not in seen_b
        seen_b.add(b)
        note = "first time at this B" if first else "B repeats"
        print(f"{i:<3} {b:<3} {wall:<10.4f} {note:<30}")


if __name__ == "__main__":
    main()
