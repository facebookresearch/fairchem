"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Atom-count growth timing test. Same execution backend
(``umas_fast_gpu_mixed``, compile=False), but each frame appends another
benzene molecule to the system so num_atoms and num_edges both grow.
Demonstrates dynamic-shape handling as system size scales.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from ase import Atoms
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


def append_benzene(atoms, rng, spacing=6.0):
    new_mol = molecule("C6H6")
    n = len(atoms) // len(new_mol)
    offset = np.array([(n + 1) * spacing, 0.0, 0.0])
    new_mol.translate(offset)
    new_mol.rotate(
        rng.uniform(0, 360),
        v=rng.normal(size=3),
        center="COM",
    )
    combined = Atoms(
        symbols=list(atoms.symbols) + list(new_mol.symbols),
        positions=np.vstack([atoms.positions, new_mol.positions]),
    )
    combined.center(vacuum=4.0)
    return combined


def main():
    settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        execution_mode="umas_fast_gpu_mixed",
    )
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    rng = np.random.default_rng(0)
    atoms = molecule("C6H6")
    atoms.center(vacuum=4.0)

    print("compile=False  execution_mode=umas_fast_gpu_mixed")
    print("growing system: +1 benzene each frame\n")

    # Warmup so frame 0 isn't dominated by lazy init
    warm = build_batch(atoms)
    _ = time_predict(predictor, warm)
    _ = time_predict(predictor, warm)

    print(f"{'frame':<6} {'natoms':<8} {'edges':<8} {'wall (ms)':<12}")

    for i in range(15):
        batch = build_batch(atoms)
        wall = time_predict(predictor, batch)
        n_edges = int(batch.edge_index.shape[1])
        print(f"{i:<6} {len(atoms):<8} {n_edges:<8} {wall * 1000:<12.3f}")
        atoms = append_benzene(atoms, rng)


if __name__ == "__main__":
    main()
