#!/usr/bin/env python3
"""
MD QPS benchmark for UMA backends.

Runs Langevin dynamics on a 2000-atom non-PBC FCC carbon system at 400K
and measures queries-per-second (each MD step = one query).

Usage:
    python check_md_qps.py --backend umas_fast_cpu --device cpu --warmup 5 --steps 50
    python check_md_qps.py --backend umas_fast_gpu --device cuda --compile --warmup 15 --steps 200
"""

from __future__ import annotations

import argparse
import time

import torch
from ase import units
from ase.md.langevin import Langevin

from bench_common import (
    attach_calculator,
    load_predictor,
    log,
    make_system,
)


def run_md_benchmark(
    backend: str,
    compile: bool,
    device: str,
    warmup: int,
    steps: int,
) -> float:
    """Run Langevin MD and return average QPS."""
    predictor = load_predictor(backend=backend, compile=compile, device=device)
    atoms = make_system()
    attach_calculator(atoms, predictor)

    dyn = Langevin(
        atoms,
        timestep=0.1 * units.fs,
        temperature_K=400,
        friction=0.001 / units.fs,
    )

    # Warmup
    log.info("Warmup: %d steps", warmup)
    t0 = time.perf_counter()
    dyn.run(warmup)
    warmup_time = time.perf_counter() - t0
    log.info("Warmup done in %.2f s (%.2f QPS)", warmup_time, warmup / max(warmup_time, 1e-9))

    if device == "cuda":
        torch.cuda.synchronize()
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        log.info("Peak GPU memory: %.2f GB", mem_gb)

    # Measured run
    log.info("Measured run: %d steps", steps)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    dyn.run(steps)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    qps = steps / elapsed
    log.info("Elapsed: %.3f s", elapsed)
    log.info("MD QPS: %.2f", qps)
    log.info("ns/day: %.2f", qps * 0.1e-3 * 86400)  # timestep_ps * 86400

    return qps


def main():
    parser = argparse.ArgumentParser(description="UMA MD QPS benchmark")
    parser.add_argument("--backend", type=str, required=True, help="Execution backend")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup MD steps")
    parser.add_argument("--steps", type=int, default=50, help="Measured MD steps")
    args = parser.parse_args()

    qps = run_md_benchmark(args.backend, args.compile, args.device, args.warmup, args.steps)
    log.info("RESULT backend=%s compile=%s device=%s md_qps=%.2f", args.backend, args.compile, args.device, qps)


if __name__ == "__main__":
    main()
