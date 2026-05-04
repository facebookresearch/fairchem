"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test (C): use torch._dynamo.mark_dynamic to skip the B=1 specialization
recompile cycle.

Subtlety: MLIPPredictUnit._run_inference calls
``data.to(device).clone()`` before model(...), creating fresh tensors.
Marks set on the user-side batch are LOST. To take effect, the marks
must be applied to the GPU-resident clone, right before model(data).
We monkey-patch _run_inference to do exactly that.
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


B_INDEXED_DIM0 = (
    "cell", "charge", "natoms", "nedges", "pbc", "spin", "dataset",
)
ATOM_INDEXED_DIM0 = (
    "atomic_numbers", "batch", "fixed", "pos", "tags",
)
EDGE_INDEXED_DIM0 = ("cell_offsets",)
EDGE_INDEXED_DIM1 = ("edge_index",)


def mark_dyn(data):
    for key in B_INDEXED_DIM0 + ATOM_INDEXED_DIM0 + EDGE_INDEXED_DIM0:
        try:
            t = data[key]
        except KeyError:
            continue
        if torch.is_tensor(t) and t.ndim >= 1:
            torch._dynamo.mark_dynamic(t, 0)
    for key in EDGE_INDEXED_DIM1:
        try:
            t = data[key]
        except KeyError:
            continue
        if torch.is_tensor(t) and t.ndim >= 2:
            torch._dynamo.mark_dynamic(t, 1)


def install_mark_patch(predictor):
    """Monkey-patch _run_inference to mark_dynamic on the GPU clone
    just before the compiled forward."""
    orig = predictor._run_inference

    def patched(self_, data, undo_element_references):
        mark_dyn(data)
        return orig.__func__(self_, data, undo_element_references) \
            if hasattr(orig, "__func__") else orig(data, undo_element_references)

    # bind so calls work via instance method semantics
    import types
    predictor._run_inference = types.MethodType(
        lambda self_, data, undo_element_references: (
            mark_dyn(data) or orig(data, undo_element_references)
        ),
        predictor,
    )


def main():
    settings = make_settings(compile=True)
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    install_mark_patch(predictor)

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

    print(f"graph_gen={GRAPH_GEN}")
    print(f"calling {len(sequence)} times with batch sizes: {sequence}")
    print("(mark_dynamic applied to GPU-resident data inside _run_inference)\n")
    print(f"{'#':<3} {'B':<3} {'wall (s)':<10} {'note':<30}")

    seen_b: set[int] = set()
    for i, b in enumerate(sequence):
        batch = make_batch(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = predictor.predict(batch)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        first_seen = b not in seen_b
        seen_b.add(b)
        note = "first time at this B" if first_seen else "B repeats"
        print(f"{i:<3} {b:<3} {wall:<10.4f} {note:<30}")


if __name__ == "__main__":
    main()
