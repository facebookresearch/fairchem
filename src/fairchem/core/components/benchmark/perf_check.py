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
    Predictions from a single inference call.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None


@dataclass
class PerfMetrics:
    """
    Aggregate timing/memory measurements for a batch of inference iterations.
    """

    qps: float | None = None
    wall_time_seconds: float | None = None
    peak_gpu_memory_mb: float | None = None
    warmup_time_seconds: float | None = None


def run_inference(
    checkpoint: str,
    systems: list[BenchmarkSystem],
    inference_settings: InferenceSettings,
    device: str = "cuda",
    seed: int = 42,
    warmup_iters: int = 0,
    timed_iters: int = 1,
) -> tuple[list[InferenceResult], PerfMetrics]:
    """
    Run inference over a sequence of (possibly distinct) systems.

    The list length must equal ``warmup_iters + timed_iters``: iterations
    rotate through it, with ``systems[:warmup_iters]`` used for warmup and
    ``systems[warmup_iters:]`` used for the timed phase. ``AtomicData`` is
    built up-front (outside the timed window). Per-iteration predictions are
    extracted inside the timed window so memory usage stays bounded — this is
    consistent across baseline and candidate calls so QPS comparisons remain
    meaningful.

    With ``warmup_iters=0`` and ``timed_iters=1`` (the baseline shape), no
    perf metrics are reported.

    Args:
        checkpoint: Model name (e.g. "uma-s-1p2") or path.
        systems: BenchmarkSystems to evaluate, one per iteration.
        inference_settings: InferenceSettings to use.
        device: Device for inference.
        seed: Random seed for determinism.
        warmup_iters: Number of warmup iterations (not timed).
        timed_iters: Number of timed iterations.

    Returns:
        Tuple of (per-timed-iteration predictions, aggregate perf metrics).
    """
    if len(systems) != warmup_iters + timed_iters:
        raise ValueError(
            f"systems length {len(systems)} != warmup_iters + timed_iters "
            f"= {warmup_iters + timed_iters}"
        )

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
    data_list = [
        AtomicData.from_ase(s.atoms, task_name=s.task_name) for s in systems
    ]

    is_cuda = device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    warmup_time = 0.0
    if warmup_iters > 0:
        warmup_start = time.perf_counter()
        for i in range(warmup_iters):
            predictor.predict(data_list[i])
            if is_cuda:
                torch.cuda.synchronize()
        warmup_time = time.perf_counter() - warmup_start

    # Timed iterations
    timed_results: list[InferenceResult] = []
    timed_start = time.perf_counter()
    for j in range(timed_iters):
        preds = predictor.predict(data_list[warmup_iters + j])
        if is_cuda:
            torch.cuda.synchronize()
        energy = float(preds["energy"].detach().cpu().to(torch.float64).item())
        forces = preds["forces"].detach().cpu().to(torch.float64).numpy()
        stress = None
        if "stress" in preds:
            stress = preds["stress"].detach().cpu().to(torch.float64).numpy()
        timed_results.append(
            InferenceResult(energy=energy, forces=forces, stress=stress)
        )
    wall_time = time.perf_counter() - timed_start

    # Performance metrics (only when measuring)
    measure_perf = timed_iters > 1 or warmup_iters > 0
    qps = (timed_iters / wall_time) if (measure_perf and wall_time > 0) else None
    peak_mem = (
        (torch.cuda.max_memory_allocated() / (1024**2))
        if (measure_perf and is_cuda)
        else None
    )

    metrics = PerfMetrics(
        qps=qps,
        wall_time_seconds=wall_time if qps is not None else None,
        peak_gpu_memory_mb=peak_mem,
        warmup_time_seconds=warmup_time if warmup_iters > 0 else None,
    )

    # Cleanup
    del predictor
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return timed_results, metrics


def compare_results(
    baselines: list[InferenceResult],
    candidates: list[InferenceResult],
    perf: PerfMetrics | None = None,
) -> dict[str, Any]:
    """
    Aggregate accuracy errors over a sequence of baseline/candidate pairs and
    pass through perf metrics.

    Args:
        baselines: Per-iteration fp64 reference results.
        candidates: Per-iteration candidate results (same length as baselines).
        perf: Optional aggregate perf metrics from the candidate run.

    Returns:
        Dict with mean and max error metrics plus any perf fields.
    """
    if len(baselines) != len(candidates):
        raise ValueError(
            f"baselines/candidates length mismatch: "
            f"{len(baselines)} vs {len(candidates)}"
        )

    energy_errs = np.array(
        [abs(b.energy - c.energy) for b, c in zip(baselines, candidates)]
    )
    force_maes = np.array(
        [
            float(np.mean(np.abs(b.forces - c.forces)))
            for b, c in zip(baselines, candidates)
        ]
    )
    force_max_errs = np.array(
        [
            float(np.max(np.abs(b.forces - c.forces)))
            for b, c in zip(baselines, candidates)
        ]
    )

    metrics: dict[str, Any] = {
        "n_systems": len(baselines),
        "energy_abs_error_mean": float(np.mean(energy_errs)),
        "energy_abs_error_max": float(np.max(energy_errs)),
        "force_mae_mean": float(np.mean(force_maes)),
        "force_max_error": float(np.max(force_max_errs)),
    }

    have_stress = all(
        b.stress is not None and c.stress is not None
        for b, c in zip(baselines, candidates)
    )
    if have_stress:
        stress_maes = np.array(
            [
                float(np.mean(np.abs(b.stress - c.stress)))
                for b, c in zip(baselines, candidates)
            ]
        )
        stress_max_errs = np.array(
            [
                float(np.max(np.abs(b.stress - c.stress)))
                for b, c in zip(baselines, candidates)
            ]
        )
        metrics["stress_mae_mean"] = float(np.mean(stress_maes))
        metrics["stress_max_error"] = float(np.max(stress_max_errs))

    if perf is not None:
        if perf.qps is not None:
            metrics["qps"] = perf.qps
        if perf.wall_time_seconds is not None:
            metrics["wall_time_seconds"] = perf.wall_time_seconds
        if perf.peak_gpu_memory_mb is not None:
            metrics["peak_gpu_memory_mb"] = perf.peak_gpu_memory_mb
        if perf.warmup_time_seconds is not None:
            metrics["warmup_time_seconds"] = perf.warmup_time_seconds

    return metrics


def format_report_table(
    results: dict[str, dict[str, Any]],
) -> str:
    """
    Format benchmark results as a human-readable table.

    Args:
        results: Dict of {archetype_name: metrics}.

    Returns:
        Formatted string table.
    """
    header = (
        f"{'Archetype':<20} {'N':>4} {'E err mean':>12} "
        f"{'F MAE mean':>12} {'F max':>12} {'QPS':>10} "
        f"{'GPU MB':>10} {'Warmup(s)':>10}"
    )
    lines = [header, "-" * len(header)]

    def _fmt(metrics: dict, key: str, fmt: str = ".6f") -> str:
        v = metrics.get(key)
        return f"{v:{fmt}}" if v is not None else "N/A"

    for arch_name, m in results.items():
        if "error" in m:
            lines.append(f"{arch_name:<20} {m['error']:>12}")
            continue
        n = m.get("n_systems")
        lines.append(
            f"{arch_name:<20} "
            f"{(str(n) if n is not None else '-'):>4} "
            f"{_fmt(m, 'energy_abs_error_mean'):>12} "
            f"{_fmt(m, 'force_mae_mean'):>12} "
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
            archetype_of,
            get_default_benchmark_systems,
        )

        self.checkpoint = checkpoint
        self.inference_settings = inference_settings
        self.device = device
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.seed = seed

        # Build one distinct variant per inference call. Each archetype
        # contributes (warmup_iters + timed_iters) variants; the timed slice
        # is what gets fp64 baselines and accuracy comparisons.
        n_per_archetype = warmup_iters + timed_iters
        flat = get_default_benchmark_systems(
            seed=seed, n_per_archetype=n_per_archetype
        )
        self.archetype_variants: dict[str, list[BenchmarkSystem]] = {}
        for system in flat:
            self.archetype_variants.setdefault(archetype_of(system.name), []).append(
                system
            )

    def run(self) -> dict:
        """
        Run the benchmark: per-variant fp64 baselines then candidate over
        every variant per archetype, with accuracy aggregated per archetype.

        Returns:
            Dict with baseline summary, per-archetype results, and settings.
        """
        output_dir = self.job_config.metadata.results_dir
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Run baseline (fp64, no optimizations) on every timed variant.
        # Cache lives in run_dir (stable across runs), not results_dir (per-run).
        cache_dir = self.job_config.run_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, BASELINE_CACHE_FILE)

        # Only the timed slice needs fp64 reference; warmup variants are not
        # included in accuracy comparisons.
        timed_variants_by_archetype: dict[str, list[BenchmarkSystem]] = {
            arch: variants[self.warmup_iters :]
            for arch, variants in self.archetype_variants.items()
        }
        all_timed = [v for vs in timed_variants_by_archetype.values() for v in vs]

        cache_key = _baseline_cache_key(
            self.checkpoint, all_timed, self.device, self.seed
        )
        baselines = _load_baseline_cache(cache_path, cache_key)

        if baselines is not None:
            logger.warning(
                "Using cached baseline results from %s. "
                "Delete this file to force recomputation.",
                cache_path,
            )
        else:
            logger.info(
                "Running baseline inference (fp64) on %d systems...", len(all_timed)
            )
            baselines = {}
            for variant in all_timed:
                logger.info(
                    "  Baseline: %s (%d atoms)",
                    variant.name,
                    len(variant.atoms),
                )
                results, _ = run_inference(
                    checkpoint=self.checkpoint,
                    systems=[variant],
                    inference_settings=BASELINE_SETTINGS,
                    device=self.device,
                    seed=self.seed,
                )
                baselines[variant.name] = results[0]
            _save_baseline_cache(cache_path, cache_key, baselines)

        baseline_summary = {
            arch: {
                "n_systems": len(variants),
                "natoms": len(variants[0].atoms),
            }
            for arch, variants in timed_variants_by_archetype.items()
        }

        # Step 2: Evaluate the candidate config per archetype.
        logger.info("Evaluating config: %s", self.inference_settings)
        results: dict[str, dict[str, Any]] = {}
        for arch, variants in self.archetype_variants.items():
            logger.info(
                "  %s: %d variants (%d atoms each)",
                arch,
                len(variants),
                len(variants[0].atoms),
            )
            timed_baselines = [baselines[v.name] for v in variants[self.warmup_iters :]]
            try:
                cand_results, perf = run_inference(
                    checkpoint=self.checkpoint,
                    systems=variants,
                    inference_settings=self.inference_settings,
                    device=self.device,
                    seed=self.seed,
                    warmup_iters=self.warmup_iters,
                    timed_iters=self.timed_iters,
                )
                results[arch] = compare_results(timed_baselines, cand_results, perf)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or isinstance(
                    e, torch.cuda.OutOfMemoryError
                ):
                    logger.warning("  OOM on %s", arch)
                    results[arch] = {"error": "OOM"}
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
