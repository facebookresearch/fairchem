#!/usr/bin/env python3
"""
Static QPS benchmark for UMA backends.

Repeatedly runs predict() on the same AtomicData input (no dynamics,
no position updates). Measures raw inference throughput by calling the
predict unit directly (bypasses ASE calculator caching).

Usage:
    python check_static_qps.py --backend umas_fast_cpu --device cpu --warmup 3 --iters 50
    python check_static_qps.py --backend umas_fast_gpu --device cuda --compile --warmup 15 --iters 200
"""

from __future__ import annotations

import argparse
import time

import torch

from bench_common import (
    load_predictor,
    log,
    make_system,
    TASK_NAME,
)
from fairchem.core.datasets.atomic_data import AtomicData


def run_static_benchmark(
    backend: str,
    compile: bool,
    device: str,
    warmup: int,
    iters: int,
) -> float:
    """Repeatedly run predict() on same data, return average QPS."""
    predictor = load_predictor(backend=backend, compile=compile, device=device)
    atoms = make_system()

    # Convert atoms to AtomicData once
    data = AtomicData.from_ase(atoms, task_name=TASK_NAME)

    # Warmup
    log.info("Warmup: %d iterations", warmup)
    t0 = time.perf_counter()
    for i in range(warmup):
        predictor.predict(data)
    warmup_time = time.perf_counter() - t0
    log.info("Warmup done in %.2f s (%.2f QPS)", warmup_time, warmup / max(warmup_time, 1e-9))

    if device == "cuda":
        torch.cuda.synchronize()
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        log.info("Peak GPU memory: %.2f GB", mem_gb)

    # Measured run
    log.info("Measured run: %d iterations", iters)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(iters):
        predictor.predict(data)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    qps = iters / elapsed
    log.info("Elapsed: %.3f s", elapsed)
    log.info("Static QPS: %.2f", qps)

    return qps


def main():
    parser = argparse.ArgumentParser(description="UMA static QPS benchmark")
    parser.add_argument("--backend", type=str, required=True, help="Execution backend")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations")
    args = parser.parse_args()

    qps = run_static_benchmark(args.backend, args.compile, args.device, args.warmup, args.iters)
    log.info("RESULT backend=%s compile=%s device=%s static_qps=%.2f", args.backend, args.compile, args.device, qps)


if __name__ == "__main__":
    main()
