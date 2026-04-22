"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fairchem.core.components.runner import Runner
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)

if TYPE_CHECKING:
    from fairchem.core.components.benchmark.systems import BenchmarkSystem

logger = logging.getLogger(__name__)

# High-precision settings used for the gold-standard baseline.
BASELINE_SETTINGS = InferenceSettings(
    tf32=False,
    activation_checkpointing=False,
    merge_mole=False,
    compile=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
    execution_mode="general",
    base_precision_dtype=torch.float64,
)

BASELINE_CACHE_FILE = "baseline_cache.json"


def _baseline_cache_key(
    checkpoint: str,
    systems: list[BenchmarkSystem],
    device: str,
    seed: int,
) -> str:
    """
    Produce a deterministic hash from the inputs that affect baseline results.
    """
    key_data = {
        "checkpoint": checkpoint,
        "systems": [{"name": s.name, "num_atoms": len(s.atoms)} for s in systems],
        "device": device,
        "seed": seed,
        "baseline_settings": str(BASELINE_SETTINGS),
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()


def _save_baseline_cache(
    cache_path: str,
    cache_key: str,
    baselines: dict[str, InferenceResult],
) -> None:
    """
    Save baseline InferenceResult dicts to a JSON cache file.
    """
    serialized: dict[str, Any] = {}
    for name, result in baselines.items():
        entry: dict[str, Any] = {
            "energy": result.energy,
            "forces": result.forces.tolist(),
        }
        if result.stress is not None:
            entry["stress"] = result.stress.tolist()
        serialized[name] = entry

    with open(cache_path, "w") as f:
        json.dump(
            {"cache_key": cache_key, "baselines": serialized},
            f,
            indent=2,
        )


def _load_baseline_cache(
    cache_path: str,
    expected_key: str,
) -> dict[str, InferenceResult] | None:
    """
    Load cached baselines if the cache file exists and the key matches.

    Returns None on missing file or key mismatch.
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("cache_key") != expected_key:
        return None

    baselines: dict[str, InferenceResult] = {}
    for name, entry in data["baselines"].items():
        stress = None
        if "stress" in entry:
            stress = np.array(entry["stress"], dtype=np.float64)
        baselines[name] = InferenceResult(
            energy=float(entry["energy"]),
            forces=np.array(entry["forces"], dtype=np.float64),
            stress=stress,
        )
    return baselines


@dataclass
class InferenceResult:
    """
    Predictions and optional performance metrics from a single inference run.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None
    qps: float | None = None
    wall_time_seconds: float | None = None
    peak_gpu_memory_mb: float | None = None
    warmup_time_seconds: float | None = None


def run_inference(
    checkpoint: str,
    system: BenchmarkSystem,
    inference_settings: InferenceSettings,
    device: str = "cuda",
    seed: int = 42,
    warmup_iters: int = 0,
    timed_iters: int = 1,
) -> InferenceResult:
    """
    Run inference on a single system, optionally measuring performance.

    When warmup_iters=0 and timed_iters=1 (defaults), this runs a single
    inference pass and returns predictions only. With higher values, it
    measures throughput (QPS) and peak GPU memory.

    Args:
        checkpoint: Model name (e.g. "uma-s-1p2") or path.
        system: BenchmarkSystem to evaluate.
        inference_settings: InferenceSettings to use.
        device: Device for inference.
        seed: Random seed for determinism.
        warmup_iters: Number of warmup iterations (not timed).
        timed_iters: Number of timed iterations.

    Returns:
        InferenceResult with predictions and optional perf metrics.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(checkpoint):
        from fairchem.core.calculate.pretrained_mlip import (
            pretrained_checkpoint_path_from_name,
        )

        checkpoint = pretrained_checkpoint_path_from_name(checkpoint)

    predictor = MLIPPredictUnit(
        checkpoint, device, inference_settings=inference_settings
    )
    data = AtomicData.from_ase(system.atoms, task_name=system.task_name)

    is_cuda = device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    warmup_time = 0.0
    if warmup_iters > 0:
        warmup_start = time.perf_counter()
        for _ in range(warmup_iters):
            predictor.predict(data)
            if is_cuda:
                torch.cuda.synchronize()
        warmup_time = time.perf_counter() - warmup_start

    # Timed iterations
    timed_start = time.perf_counter()
    for _ in range(timed_iters):
        preds = predictor.predict(data)
        if is_cuda:
            torch.cuda.synchronize()
    wall_time = time.perf_counter() - timed_start

    # Extract predictions
    energy = float(preds["energy"].detach().cpu().to(torch.float64).item())
    forces = preds["forces"].detach().cpu().to(torch.float64).numpy()
    stress = None
    if "stress" in preds:
        stress = preds["stress"].detach().cpu().to(torch.float64).numpy()

    # Performance metrics (only when measuring)
    qps = None
    peak_mem = None
    if timed_iters > 1 or warmup_iters > 0:
        qps = timed_iters / wall_time if wall_time > 0 else 0.0
        if is_cuda:
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    # Cleanup
    del predictor
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return InferenceResult(
        energy=energy,
        forces=forces,
        stress=stress,
        qps=qps,
        wall_time_seconds=wall_time if qps is not None else None,
        peak_gpu_memory_mb=peak_mem,
        warmup_time_seconds=warmup_time if warmup_iters > 0 else None,
    )


def compare_results(
    baseline: InferenceResult,
    candidate: InferenceResult,
) -> dict[str, Any]:
    """
    Compare candidate predictions against baseline, computing error metrics.

    Args:
        baseline: Gold-standard reference result.
        candidate: Result from a candidate configuration.

    Returns:
        Dict with accuracy errors and performance metrics.
    """
    metrics: dict[str, Any] = {}

    # Energy error
    metrics["energy_abs_error"] = abs(baseline.energy - candidate.energy)

    # Force errors
    force_diff = np.abs(baseline.forces - candidate.forces)
    metrics["force_mae"] = float(np.mean(force_diff))
    metrics["force_max_error"] = float(np.max(force_diff))

    # Stress errors
    if baseline.stress is not None and candidate.stress is not None:
        stress_diff = np.abs(baseline.stress - candidate.stress)
        metrics["stress_mae"] = float(np.mean(stress_diff))
        metrics["stress_max_error"] = float(np.max(stress_diff))

    # Performance metrics (pass through from candidate)
    if candidate.qps is not None:
        metrics["qps"] = candidate.qps
    if candidate.wall_time_seconds is not None:
        metrics["wall_time_seconds"] = candidate.wall_time_seconds
    if candidate.peak_gpu_memory_mb is not None:
        metrics["peak_gpu_memory_mb"] = candidate.peak_gpu_memory_mb
    if candidate.warmup_time_seconds is not None:
        metrics["warmup_time_seconds"] = candidate.warmup_time_seconds

    return metrics


def format_report_table(
    results: dict[str, dict[str, Any]],
) -> str:
    """
    Format benchmark results as a human-readable table.

    Args:
        results: Dict of {system_name: metrics}.

    Returns:
        Formatted string table.
    """
    header = (
        f"{'System':<20} {'E err(eV)':>12} "
        f"{'F MAE':>12} {'F max':>12} {'QPS':>10} "
        f"{'GPU MB':>10} {'Warmup(s)':>10}"
    )
    lines = [header, "-" * len(header)]

    def _fmt(metrics: dict, key: str, fmt: str = ".6f") -> str:
        v = metrics.get(key)
        return f"{v:{fmt}}" if v is not None else "N/A"

    for sys_name, m in results.items():
        if "error" in m:
            lines.append(f"{sys_name:<20} {m['error']:>12}")
            continue
        lines.append(
            f"{sys_name:<20} "
            f"{_fmt(m, 'energy_abs_error'):>12} "
            f"{_fmt(m, 'force_mae'):>12} "
            f"{_fmt(m, 'force_max_error'):>12} "
            f"{_fmt(m, 'qps', '.2f'):>10} "
            f"{_fmt(m, 'peak_gpu_memory_mb', '.0f'):>10} "
            f"{_fmt(m, 'warmup_time_seconds', '.2f'):>10}"
        )

    return "\n".join(lines)


class PerfCheckRunner(Runner):
    """
    Benchmark a single inference configuration against a fp64 baseline.

    Runs high-precision fp64 baseline inference on default test systems,
    then runs the given inference_settings on the same systems and reports
    accuracy error and performance metrics.

    Usage via fairchem CLI:
        fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml
        fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml \
            runner.inference_settings.execution_mode=umas_fast_gpu
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        warmup_iters: int = 10,
        timed_iters: int = 50,
        seed: int = 42,
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa: B008
    ):
        from fairchem.core.components.benchmark.systems import (
            get_default_benchmark_systems,
        )

        self.checkpoint = checkpoint
        self.systems = get_default_benchmark_systems(seed=seed)
        self.inference_settings = inference_settings
        self.device = device
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.seed = seed

    def run(self) -> dict:
        """
        Run the benchmark: baseline then candidate on each system.

        Returns:
            Dict with baseline summary, candidate results, and settings.
        """
        output_dir = self.job_config.metadata.results_dir
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Run baseline (fp64, no optimizations) on each system
        # Cache lives in run_dir (stable across runs), not results_dir (per-run)
        cache_dir = self.job_config.run_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, BASELINE_CACHE_FILE)
        cache_key = _baseline_cache_key(
            self.checkpoint, self.systems, self.device, self.seed
        )
        baselines = _load_baseline_cache(cache_path, cache_key)

        if baselines is not None:
            logger.warning(
                "Using cached baseline results from %s. "
                "Delete this file to force recomputation.",
                cache_path,
            )
        else:
            logger.info("Running baseline inference (fp64)...")
            baselines = {}
            for system in self.systems:
                logger.info(
                    "  Baseline: %s (%d atoms)",
                    system.name,
                    len(system.atoms),
                )
                baselines[system.name] = run_inference(
                    checkpoint=self.checkpoint,
                    system=system,
                    inference_settings=BASELINE_SETTINGS,
                    device=self.device,
                    seed=self.seed,
                )
            _save_baseline_cache(cache_path, cache_key, baselines)

        baseline_summary = {
            name: {
                "energy": result.energy,
                "num_atoms": result.forces.shape[0],
            }
            for name, result in baselines.items()
        }

        # Step 2: Evaluate the candidate config
        logger.info("Evaluating config: %s", self.inference_settings)
        results: dict[str, dict[str, Any]] = {}
        for system in self.systems:
            logger.info("  %s (%d atoms)", system.name, len(system.atoms))
            try:
                candidate = run_inference(
                    checkpoint=self.checkpoint,
                    system=system,
                    inference_settings=self.inference_settings,
                    device=self.device,
                    seed=self.seed,
                    warmup_iters=self.warmup_iters,
                    timed_iters=self.timed_iters,
                )
                results[system.name] = compare_results(
                    baselines[system.name], candidate
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or isinstance(
                    e, torch.cuda.OutOfMemoryError
                ):
                    logger.warning("  OOM on %s", system.name)
                    results[system.name] = {"error": "OOM"}
                else:
                    raise

        # Step 3: Format and log results
        table = format_report_table(results)
        logger.info("Benchmark results:\n%s", table)

        # Step 4: Save JSON report
        full_report = {
            "baseline": baseline_summary,
            "inference_settings": str(self.inference_settings),
            "results": results,
        }
        report_path = os.path.join(output_dir, "benchmark_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info("Report saved to %s", report_path)

        return full_report

    def save_state(self, _):
        return

    def load_state(self, _):
        return
