#!/usr/bin/env python3
"""
Force correctness comparison for UMA backends.

Two modes:
  --generate     Run the `general` backend (no compile) to produce the
                 gold-standard reference PKL for a 2000-atom non-PBC FCC system.
                 This only needs to be done once.

  --backend X    Run backend X and compare forces/energy against the gold PKL.
                 Prints PASS or FAIL.

Examples:
    # Generate gold standard (once, on any device):
    python compare_forces.py --generate --device cpu

    # Check a backend:
    python compare_forces.py --backend umas_fast_cpu --device cpu
    python compare_forces.py --backend umas_fast_gpu --device cuda --compile
"""

from __future__ import annotations

import argparse
import sys

import torch

from bench_common import (
    attach_calculator,
    compare,
    load_gold,
    load_predictor,
    log,
    make_system,
    print_comparison,
    save_gold,
)


def generate_gold(device: str) -> None:
    """Run general backend without compile, save reference PKL."""
    log.info("Generating gold standard with general backend (no compile) on %s", device)
    predictor = load_predictor(backend="general", compile=False, device=device)
    atoms = make_system()
    attach_calculator(atoms, predictor)

    # Single-point calculation
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    log.info("Energy: %.6f eV", energy)
    log.info("Forces shape: %s, max abs: %.6e", forces.shape, abs(forces).max())

    save_gold(energy=float(energy), forces=forces, natoms=len(atoms))


def check_backend(backend: str, compile: bool, device: str) -> bool:
    """Run backend, compare against gold, return True if PASS."""
    gold = load_gold()

    predictor = load_predictor(backend=backend, compile=compile, device=device)
    atoms = make_system(natoms=gold.natoms)
    attach_calculator(atoms, predictor)

    log.info("Running single-point with backend=%s compile=%s device=%s", backend, compile, device)

    # Warmup for compile
    if compile:
        log.info("Warmup pass (compile)...")
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        log.info("Warmup done.")

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()

    passed, details = compare(energy, forces, gold)
    print_comparison(details)
    return passed


def main():
    parser = argparse.ArgumentParser(description="UMA force correctness comparison")
    parser.add_argument("--generate", action="store_true", help="Generate gold standard PKL")
    parser.add_argument("--backend", type=str, default=None, help="Backend to test")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    if args.generate:
        generate_gold(args.device)
        return

    if args.backend is None:
        parser.error("Must specify --generate or --backend <name>")

    passed = check_backend(args.backend, args.compile, args.device)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
