"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

B>1 mixed-batch test: collate N systems with different compositions into
a single AtomicData via atomicdata_list_to_batch, then run one
predictor.predict(batch) call. Compare per-system predictions between
eager and compile=True.

This stresses the segment_mm path at B>1 (no B=1 fast path on this
branch) and the multi-system mole_sizes routing.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

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


def make_mol_box(mol_name, n=20, seed=42, box=10.0):
    rng = np.random.default_rng(seed)
    template = molecule(mol_name)
    pos, sym = [], []
    for _ in range(n):
        offset = rng.random(3) * box
        pos.extend(template.get_positions() + offset)
        sym.extend(template.get_chemical_symbols())
    return Atoms(symbols=sym, positions=pos, cell=[box] * 3, pbc=True)


def split_predictions(preds, batch):
    """Split batched predictions back to per-system."""
    out = []
    for i in range(int(batch.natoms.numel())):
        sys = {}
        for key, val in preds.items():
            if not torch.is_tensor(val):
                continue
            if val.shape[0] == batch.natoms.numel():
                sys[key] = val[i : i + 1]
            elif val.shape[0] == int(batch.batch.numel()):
                mask = batch.batch == i
                sys[key] = val[mask]
        out.append(sys)
    return out


def run_b_gt_1(systems_atoms_task, compile_flag):
    settings = make_settings(compile=compile_flag, tf32=False)
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    # Build per-system AtomicData and collate into one batched object.
    data_list = [
        make_atomic_data(atoms, task_name=task)
        for atoms, task in systems_atoms_task
    ]
    batch = atomicdata_list_to_batch(data_list)

    preds = predictor.predict(batch)
    torch.cuda.synchronize()

    return preds, batch


def main():
    # MIXED BATCH: B=4 systems with different compositions, processed
    # together in ONE forward call.
    # NOTE: All systems in a batch must use the same task_name for
    # uma-s-1p1's MOLE-coefficient routing to be consistent. We use 4
    # different FCC metals (all task=omat).
    systems_metals = [
        (make_fcc("Cu", 200, 0), "omat"),
        (make_fcc("Si", 200, 1), "omat"),
        (make_fcc("Al", 200, 2), "omat"),
        (make_fcc("Ge", 200, 3), "omat"),
    ]

    print(f"=== B=4 mixed-element FCC batch  (graph_gen={GRAPH_GEN}) ===")
    print(f"systems: {[sorted(set(a.get_chemical_symbols())) for a, _ in systems_metals]}")
    print(f"natoms per system: {[len(a) for a, _ in systems_metals]}")
    print()

    print("running eager ...")
    preds_e, batch_e = run_b_gt_1(systems_metals, compile_flag=False)
    eager_split = split_predictions(preds_e, batch_e)
    print(f"  eager output keys: {list(preds_e.keys())}")
    print(f"  energy shape: {preds_e['energy'].shape}")
    print(f"  forces shape: {preds_e['forces'].shape}")
    for i, p in enumerate(eager_split):
        print(f"  system {i}: E={p['energy'].item():.4f}  "
              f"|F|max={p['forces'].abs().max().item():.4e}")
    print()

    print("running compile=True ...")
    preds_c, batch_c = run_b_gt_1(systems_metals, compile_flag=True)
    compile_split = split_predictions(preds_c, batch_c)
    for i, p in enumerate(compile_split):
        print(f"  system {i}: E={p['energy'].item():.4f}  "
              f"|F|max={p['forces'].abs().max().item():.4e}")
    print()

    print("=== diff per system ===")
    print(f"  {'#':<2} {'system':<22} {'|dE|':<14} {'|dF|max':<14} {'|dF|mean':<14}")
    fail = False
    for i in range(len(systems_metals)):
        sym = sorted(set(systems_metals[i][0].get_chemical_symbols()))
        dE = float((compile_split[i]["energy"] - eager_split[i]["energy"]).abs().item())
        dF = (compile_split[i]["forces"] - eager_split[i]["forces"]).abs()
        dF_max = float(dF.max().item())
        dF_mean = float(dF.mean().item())
        bad = dF_max > 1e-2 or np.isnan(dF_max) or np.isnan(dE)
        marker = "  ❌" if bad else ""
        if bad:
            fail = True
        print(f"  [{i}] {str(sym):<22} {dE:<14.4e} {dF_max:<14.4e} "
              f"{dF_mean:<14.4e}{marker}")

    print()
    if fail:
        print("FAIL")
    else:
        print("PASS: B=4 mixed-element batch matches eager within bf16 precision")


if __name__ == "__main__":
    main()
