"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Verify compile=True correctness across DIFFERENT compositions sequentially.
Each call uses a different element (Cu, Si, Ge, ...) and a different
small-molecule type (H2O, CH4, NH3, ...). Compare compile vs eager
energies and forces system-by-system.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
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


def run(systems, compile_flag):
    settings = make_settings(compile=compile_flag, tf32=False)
    ckpt = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = MLIPPredictUnit(ckpt, "cuda", inference_settings=settings)

    out = []
    for atoms, task in systems:
        # Fresh AtomicData per call so each forward is independent.
        data = make_atomic_data(atoms, task_name=task)
        preds = predictor.predict(data)
        torch.cuda.synchronize()
        out.append({
            "energy": float(preds["energy"].detach().cpu().to(torch.float64).item()),
            "forces": preds["forces"].detach().cpu().to(torch.float64).numpy(),
        })
    return out


def main():
    # Mix of materials (omat) and molecules (omol). Each consecutive call
    # has a different composition.
    systems = [
        (make_fcc("Cu", 200, 0), "omat"),
        (make_mol_box("H2O", 20, 1), "omol"),
        (make_fcc("Si", 200, 2), "omat"),
        (make_mol_box("CH4", 20, 3), "omol"),
        (make_fcc("Al", 200, 4), "omat"),
        (make_mol_box("NH3", 20, 5), "omol"),
        (make_fcc("Ge", 200, 6), "omat"),
        (make_fcc("Pd", 200, 7), "omat"),
    ]

    print(f"running {len(systems)} diverse systems  (graph_gen={GRAPH_GEN})\n")
    print("=== eager ===")
    eager = run(systems, compile_flag=False)
    for i, (s, t) in enumerate(systems):
        elements = sorted(set(s.get_chemical_symbols()))
        print(f"  [{i}] {elements} natoms={len(s):4d}  E={eager[i]['energy']:.4f}  "
              f"|F|max={np.abs(eager[i]['forces']).max():.4e}")

    print("\n=== compile=True ===")
    compiled = run(systems, compile_flag=True)
    for i, (s, t) in enumerate(systems):
        elements = sorted(set(s.get_chemical_symbols()))
        print(f"  [{i}] {elements} natoms={len(s):4d}  E={compiled[i]['energy']:.4f}  "
              f"|F|max={np.abs(compiled[i]['forces']).max():.4e}")

    print("\n=== diff (compile - eager) per system ===")
    print(f"  {'#':<2} {'system':<28} {'|dE|':<14} {'|dF|max':<14} {'|dF|mean':<14}")
    fail = False
    for i, (s, t) in enumerate(systems):
        elements = sorted(set(s.get_chemical_symbols()))
        dE = abs(compiled[i]["energy"] - eager[i]["energy"])
        dF = np.abs(compiled[i]["forces"] - eager[i]["forces"])
        dF_max = dF.max()
        dF_mean = dF.mean()
        bad = dF_max > 1e-2 or np.isnan(dF_max) or np.isnan(dE)
        if bad:
            fail = True
        marker = "  ❌" if bad else ""
        print(f"  [{i}] {str(elements):<28} {dE:<14.4e} {dF_max:<14.4e} {dF_mean:<14.4e}{marker}")

    print()
    if fail:
        print("FAIL: at least one system has |dF|max > 1e-2 or NaN")
    else:
        print("PASS: all diverse systems match eager within bf16 precision")


if __name__ == "__main__":
    main()
