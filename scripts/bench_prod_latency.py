"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Production-regime latency: re-measure compacted checkpoints the way a real
# deployment (and the collaborator's eSEN profile) runs inference --
#   * fast backend (execution_mode="umas_fast_pytorch"), TF32 on
#   * activation_checkpointing OFF (recompute-free forward)
#   * conserving E + autograd F
#   * warmups + median over timed iters, peak GPU memory
#   * swept across system sizes (small -> large) to expose the fixed-cost floor
#
# This corrects scripts/bench_md_latency.py, which used inference_settings_default
# (TF32 off, activation checkpointing ON, auto backend) -- a non-production regime
# that inflates the compute slice channel pruning targets.
#
#   python scripts/bench_prod_latency.py dense.pt a.pt b.pt --natoms 256 2048 5120
# =============================================================================
import argparse
import dataclasses
import math
import statistics
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule as get_molecule

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.units.mlip_unit.api.inference import inference_settings_turbo
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit


def build_water_box(nwaters, density_gcc=1.0):
    """
    Condensed-phase-density water cluster (nonperiodic, no cell) so it feeds the
    molecule graph path. Grid spacing set to the ~1 g/cc water O-O spacing, giving
    realistic (bulk-like) neighbor density; nonperiodic so it matches the
    collaborator's synthetic-molecule inputs.
    """
    mass_g = nwaters * 18.015 / 6.022e23
    box_a = ((mass_g / density_gcc) * 1e24) ** (1 / 3)
    side = math.ceil(nwaters ** (1 / 3))
    spacing = box_a / side
    a = Atoms()  # no cell, pbc False -> molecule path
    placed = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if placed >= nwaters:
                    break
                w = get_molecule("H2O").copy()
                w.rotate((placed * 137.5) % 360, "z")
                w.rotate((placed * 71.3) % 360, "x")
                w.translate(np.array([i, j, k]) * spacing)
                a += w
                placed += 1
    a.info["charge"] = 0
    a.info["spin"] = 1
    return a


def make_pu(ckpt, device, compile_fwd):
    s = inference_settings_turbo()
    # turbo enables compile by default; the conserving E+F path stalls under
    # inductor, so keep it off unless explicitly requested.
    s = dataclasses.replace(
        s,
        compile=compile_fwd,
        merge_mole=False,  # not a MOLE model
        auto_add_default_untrained_tasks=False,
        execution_mode="umas_fast_pytorch",
    )
    return MLIPPredictUnit(ckpt, device=device, inference_settings=s)


def time_ckpt(ckpt, atoms, device, warmup, iters, compile_fwd):
    pu = make_pu(ckpt, device, compile_fwd)
    data = atomicdata_list_to_batch(
        [
            AtomicData.from_ase(
                atoms,
                r_edges=False,
                radius=6,
                max_neigh=300,
                r_data_keys=["spin", "charge"],
                molecule_cell_size=120.0,
                task_name="omol",
            )
        ]
    )
    for _ in range(warmup):
        pu.predict(data, undo_element_references=False)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pu.predict(data, undo_element_references=False)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    del pu
    if device == "cuda":
        torch.cuda.empty_cache()
    return statistics.median(ts), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="first is baseline (dense)")
    ap.add_argument("--natoms", type=int, nargs="+", default=[256, 2048, 5120])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} turbo(fast_pytorch,tf32,no-actckpt) compile={args.compile} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    for target in args.natoms:
        atoms = build_water_box(max(1, round(target / 3)))
        base_ms = None
        print(f"\n=== {len(atoms)} atoms ===")
        print(f"{'ckpt':<40} {'ms':>9} {'speedup':>8} {'peak GiB':>9}")
        for c in args.ckpts:
            name = c.rsplit("/", 1)[-1].replace("_inference.pt", "")
            try:
                ms, peak = time_ckpt(
                    c, atoms, device, args.warmup, args.iters, args.compile
                )
            except torch.OutOfMemoryError:
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"{name:<40} {'OOM':>9} {'--':>8} {'--':>9}")
                continue
            if base_ms is None:
                base_ms = ms
            spd = f"{base_ms / ms:.2f}x" if base_ms else "--"
            print(f"{name:<40} {ms:>9.2f} {spd:>8} {peak:>9.3f}")


if __name__ == "__main__":
    main()
