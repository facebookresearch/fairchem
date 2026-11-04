"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fairchem.core.common import distutils
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit


def summarize(samples: list[float]) -> dict:
    """
    Summarize synchronized wall-clock timings in milliseconds.
    """
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] * 1000,
        "samples": len(samples),
    }


def main():
    """
    Benchmark one parallel input mode in a fresh process.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode", choices=["checked", "fixed", "baseline"], required=True
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--execution-mode", choices=["general", "umas_fast_gpu"], default="general"
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(0)
    atoms = get_fcc_crystal_by_num_atoms(args.atoms)
    input_digest = hashlib.sha256(
        atoms.numbers.tobytes() + atoms.positions.tobytes() + atoms.cell.array.tobytes()
    ).hexdigest()
    data = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="omat", r_edges=False)]
    )
    settings = InferenceSettings(
        merge_mole=True,
        tf32=args.tf32,
        compile=args.compile,
        activation_checkpointing=False,
        external_graph_gen=False,
        auto_add_default_untrained_tasks=False,
        execution_mode=args.execution_mode,
    )
    kwargs = {}
    if args.mode != "baseline":
        kwargs["assume_fixed_input"] = args.mode == "fixed"
    predictor = ParallelMLIPPredictUnit(
        args.checkpoint,
        device=args.device,
        inference_settings=settings,
        num_workers=args.workers,
        num_workers_per_node=args.workers,
        **kwargs,
    )
    try:
        # Warm beyond the model's Dynamo recompile limit before timing. Some
        # PyTorch versions retrace gradient-context wrappers during early calls.
        warmup = max(args.warmup, 64) if args.compile else args.warmup
        for step in range(warmup):
            data.pos[0, 0] += 1e-5 if step % 2 == 0 else -1e-5
            predictor.predict(data.clone())
        if not predictor.inference_settings.merge_mole:
            raise RuntimeError("Benchmark unexpectedly fell back from merged inference")
        if predictor.inference_settings.compile != args.compile:
            raise RuntimeError("Benchmark did not retain the requested compile setting")
        elapsed = []
        for step in range(args.steps):
            data.pos[0, 0] += 1e-5 if step % 2 == 0 else -1e-5
            # The baseline moves its argument to CUDA in place. Fresh CPU
            # requests keep the transfer workload identical across all modes.
            request = data.clone()
            start = time.perf_counter()
            result = predictor.predict(request)
            # Predictions return CPU tensors; this includes the GPU-to-CPU
            # synchronization and the actual distributed forward/backward work.
            elapsed.append(time.perf_counter() - start)
        output = {
            "mode": args.mode,
            "device": args.device,
            "atoms": len(atoms),
            "requested_atoms": args.atoms,
            "input_digest": input_digest,
            "workers": args.workers,
            "compile": args.compile,
            "warmup": warmup,
            "tf32": args.tf32,
            "execution_mode": args.execution_mode,
            "effective_execution_mode": str(
                predictor.inference_settings.execution_mode
            ),
            "nitro": predictor.inference_settings.nitro,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "checkpoint": args.checkpoint,
            "geometry": summarize(elapsed),
            "geometry_samples_ms": [value * 1000 for value in elapsed],
            "energy": result["energy"].tolist(),
        }
        if args.mode == "checked":
            elapsed = []
            for _ in range(min(args.steps, 20)):
                data.tags[0] = 1 - data.tags[0]
                request = data.clone()
                start = time.perf_counter()
                predictor.predict(request)
                elapsed.append(time.perf_counter() - start)
            output["metadata_refresh"] = summarize(elapsed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2), flush=True)
    finally:
        distutils.cleanup_gp_ray()


if __name__ == "__main__":
    main()
