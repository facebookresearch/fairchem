"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Real ASE MD latency validation: drive VelocityVerlet dynamics with the
# FAIRChemCalculator (positions evolve via actual forces, graph rebuilt each
# step) and time per-step wall-clock for dense vs the compacted (re-healed)
# sphere model. Also reports energy drift as a stability sanity check.
#
#   python scripts/bench_md_latency.py <dense.pt> <compacted.pt> [--natoms 200] [--steps 40]
# =============================================================================
import argparse
import dataclasses
import math
import statistics
import time

import numpy as np
import torch
from ase import Atoms, units
from ase.build import molecule as get_molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit


def build_system(nmols, spacing=10.0):
    # Well-separated grid (>= vdW + cutoff clearance) so the initial geometry has
    # no clashes -> forces are physical and the dense/compacted trajectories can be
    # compared as an actual energy-stability sanity, not a violent relaxation.
    names = ["CH3CH2OH", "CH3COOH", "C6H6", "CH3CN"]
    side = math.ceil(nmols ** (1 / 3))
    a = None
    for i in range(nmols):
        m = get_molecule(names[i % len(names)]).copy()
        m.translate(
            [
                (i % side) * spacing,
                (i // side % side) * spacing,
                i // (side * side) * spacing,
            ]
        )
        a = m if a is None else a + m
    a.info["charge"] = 0
    a.info["spin"] = 1
    return a


def build_water_box(nwaters, density_gcc=1.0):
    # Condensed-phase (liquid-like) system at physical density: this is the regime
    # where per-step cost is dominated by the neural compute (edge count scales with
    # N at fixed density), so channel compaction's FLOP savings actually show up.
    # Water grid at the target density with random orientation ~ amorphous ice.
    mass_g = nwaters * 18.015 / 6.022e23
    vol_cm3 = mass_g / density_gcc
    box_a = (vol_cm3 * 1e24) ** (1 / 3)  # cm -> Angstrom
    side = math.ceil(nwaters ** (1 / 3))
    spacing = box_a / side
    a = Atoms(pbc=True, cell=[box_a, box_a, box_a])
    placed = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if placed >= nwaters:
                    break
                w = get_molecule("H2O").copy()
                # deterministic pseudo-random tumble (no RNG: reproducible)
                ang = (placed * 137.5) % 360
                w.rotate(ang, "z")
                w.rotate((placed * 71.3) % 360, "x")
                w.translate(np.array([i, j, k]) * spacing + spacing / 2 - box_a / 2)
                a += w
                placed += 1
    a.center()
    a.info["charge"] = 0
    a.info["spin"] = 1
    return a


def make_calc(ckpt, device):
    settings = dataclasses.replace(
        inference_settings_default(), auto_add_default_untrained_tasks=False
    )
    pu = MLIPPredictUnit(ckpt, device=device, inference_settings=settings)
    return FAIRChemCalculator(pu, task_name="omol")


def run_md(calc, atoms, steps, device):
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=150)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    dyn.run(3)  # warmup (lazy init / compile / first graph)
    e0 = atoms.get_potential_energy()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        dyn.run(1)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    e1 = atoms.get_potential_energy()
    return statistics.median(times), e0, e1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dense")
    ap.add_argument("compacted")
    ap.add_argument(
        "--natoms", type=int, nargs="+", default=[200, 560, 1000], help="approx sizes"
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument(
        "--system",
        choices=["gas", "water"],
        default="water",
        help="gas = 10A-spaced molecule grid (overhead-bound); "
        "water = condensed-phase liquid box at ~1 g/cc (compute-bound)",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build calculators once; reuse across sizes (graph rebuilt per system anyway).
    dense_calc = make_calc(args.dense, device)
    comp_calc = make_calc(args.compacted, device)
    print(
        f"device={device} steps={args.steps} system={args.system} "
        "(VelocityVerlet 0.5fs, 150K)"
    )
    print(
        f"{'natoms':>7} {'dense ms':>9} {'comp ms':>9} {'speedup':>8} {'drift(d/c) eV':>18}"
    )
    for target in args.natoms:
        if args.system == "water":
            atoms = build_water_box(max(1, round(target / 3)))  # 3 atoms/water
        else:
            atoms = build_system(max(1, round(target / 9)))  # ~9 atoms/molecule
        td, e0d, e1d = run_md(dense_calc, atoms, args.steps, device)
        tc, e0c, e1c = run_md(comp_calc, atoms, args.steps, device)
        print(
            f"{len(atoms):>7} {td:>9.2f} {tc:>9.2f} {td / tc:>7.2f}x "
            f"{e1d - e0d:>+8.3f} / {e1c - e0c:>+8.3f}"
        )


if __name__ == "__main__":
    main()
