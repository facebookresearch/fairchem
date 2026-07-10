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
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import OmegaConf

from fairchem.core.components.runner import Runner
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)

if TYPE_CHECKING:
    from fairchem.core.components.benchmark.systems import (
        BenchmarkSystem,
        SystemPool,
    )

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
MIXED_BASELINE_CACHE_FILE = "mixed_baseline_cache.json"
MIXED_REPORT_FILE = "mixed_benchmark_report.json"
GRID_BASELINE_CACHE_FILE = "grid_baseline_cache.json"
GRID_REPORT_FILE = "grid_benchmark_report.json"
DEFAULT_BATCH_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_GRID_SIZES: tuple[int, ...] = (10, 20, 40, 80, 160, 320)
DEFAULT_GRID_BATCHES: tuple[int, ...] = (4, 8, 16, 32, 64)


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


def build_batch_schedule(
    pool_size: int,
    batch_sizes: list[int] | tuple[int, ...],
    num_steps: int,
    seed: int = 42,
    max_reroll: int = 32,
) -> list[tuple[int, tuple[int, ...]]]:
    """
    Build a deterministic schedule of (batch_size, system_indices) pairs.

    Round-robins through ``batch_sizes`` and samples ``batch_size`` pool indices
    (with replacement) per step. Guarantees no two adjacent steps share the same
    (batch_size, sorted_indices) multiset, by rerolling up to ``max_reroll``
    times. If the pool is degenerate (single system, batch_size==1), reroll
    cannot succeed; this is intentional and the caller should diversify the
    pool.

    Args:
        pool_size: Number of systems available in the SystemPool.
        batch_sizes: Batch sizes to cycle through.
        num_steps: Total number of (warmup + timed) steps to produce.
        seed: Seed for the deterministic RNG.
        max_reroll: Max attempts before accepting a duplicate adjacent batch.

    Returns:
        List of (batch_size, indices_tuple) entries of length ``num_steps``.
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if not batch_sizes:
        raise ValueError("batch_sizes must be non-empty")
    if num_steps <= 0:
        return []

    rng = np.random.default_rng(seed)
    sizes = list(batch_sizes)
    schedule: list[tuple[int, tuple[int, ...]]] = []
    last_key: tuple[int, tuple[int, ...]] | None = None

    for step in range(num_steps):
        bsz = sizes[step % len(sizes)]
        for _ in range(max_reroll):
            indices = tuple(int(i) for i in rng.integers(0, pool_size, size=bsz))
            key = (bsz, tuple(sorted(indices)))
            if key != last_key:
                break
        else:
            # Reroll budget exhausted; accept duplicate (pool too narrow).
            pass
        schedule.append((bsz, indices))
        last_key = (bsz, tuple(sorted(indices)))

    return schedule


@dataclass
class BatchTiming:
    """
    Aggregated timing for one batch size across the timed phase.
    """

    batch_size: int
    n_steps: int
    total_seconds: float
    samples_per_sec: float
    atoms_per_sec: float
    peak_gpu_memory_mb: float | None = None


@dataclass
class MixedInferenceResult:
    """
    Output of ``run_mixed_inference``: per-system predictions plus per-batch-size
    timings. Per-system predictions are populated only from the timed phase.
    """

    per_system: dict[str, InferenceResult]
    per_batch_size: dict[int, BatchTiming]
    warmup_seconds: float
    total_timed_seconds: float
    peak_gpu_memory_mb: float | None = None
    oom_batch_sizes: list[int] = None  # populated by runner if oom_policy=skip


def _split_batched_predictions(
    preds: dict[str, torch.Tensor],
    natoms_per_system: list[int],
) -> list[dict[str, torch.Tensor]]:
    """
    Split outputs from ``MLIPPredictUnit.predict`` (collate_predictions returns
    concatenated atom-level / per-system tensors) back into per-system dicts.
    """
    n_systems = len(natoms_per_system)
    splits: list[dict[str, torch.Tensor]] = [{} for _ in range(n_systems)]
    total_atoms = int(sum(natoms_per_system))

    for prop, tensor in preds.items():
        t = tensor.detach()
        if t.dim() == 0 or t.shape[0] == n_systems:
            # system-level (energy, stress per-system): one row per system
            for i in range(n_systems):
                splits[i][prop] = t[i] if t.dim() > 0 else t
        elif t.shape[0] == total_atoms:
            # atom-level (forces): split by natoms
            offsets = np.cumsum([0, *natoms_per_system]).tolist()
            for i in range(n_systems):
                splits[i][prop] = t[offsets[i] : offsets[i + 1]]
        else:
            # Unknown layout - give every system the full tensor; caller decides.
            for i in range(n_systems):
                splits[i][prop] = t
    return splits


def _to_inference_result(pred: dict[str, torch.Tensor]) -> InferenceResult:
    """
    Convert a per-system prediction dict to an InferenceResult (fp64 numpy).
    """
    energy_t = pred["energy"].detach().cpu().to(torch.float64)
    energy = float(
        energy_t.item() if energy_t.dim() == 0 else energy_t.reshape(()).item()
    )
    forces = pred["forces"].detach().cpu().to(torch.float64).numpy()
    stress = None
    if "stress" in pred:
        stress = pred["stress"].detach().cpu().to(torch.float64).numpy()
    return InferenceResult(energy=energy, forces=forces, stress=stress)


def run_mixed_inference(
    predict_unit: MLIPPredictUnit,
    pool: SystemPool,
    schedule: list[tuple[int, tuple[int, ...]]],
    warmup_steps: int,
    device: str = "cuda",
    oom_policy: str = "abort",
) -> MixedInferenceResult:
    """
    Run a mixed-batch benchmark using a pre-built predict unit and pre-materialized
    schedule.

    Walks ``schedule[:warmup_steps]`` untimed, then walks the remainder while
    measuring per-batch-size throughput. Per-system predictions are taken from
    the last time each system appears in the timed phase.

    Args:
        predict_unit: A ready ``MLIPPredictUnit``.
        pool: The ``SystemPool`` referenced by schedule indices.
        schedule: Output of ``build_batch_schedule``.
        warmup_steps: First ``warmup_steps`` entries are warmup (untimed).
        device: "cuda" or "cpu".
        oom_policy: "abort" (re-raise) or "skip" (record batch size as OOM and
            continue with the rest of the schedule).

    Returns:
        MixedInferenceResult.
    """
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    if oom_policy not in {"abort", "skip"}:
        raise ValueError(f"oom_policy must be 'abort' or 'skip', got {oom_policy!r}")

    is_cuda = device == "cuda" and torch.cuda.is_available()

    # Pre-build AtomicData once per pool entry; collation is cheap relative to fwd.
    atomic_data = [
        AtomicData.from_ase(s.atoms, task_name=s.task_name) for s in pool.systems
    ]

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    def _step(indices: tuple[int, ...]) -> dict[str, torch.Tensor]:
        batch = atomicdata_list_to_batch([atomic_data[i] for i in indices])
        out = predict_unit.predict(batch)
        if is_cuda:
            torch.cuda.synchronize()
        return out

    # Warmup
    warmup_start = time.perf_counter()
    for bsz, indices in schedule[:warmup_steps]:
        try:
            _step(indices)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if _is_oom(e) and oom_policy == "skip":
                logger.warning("OOM during warmup at batch_size=%d", bsz)
                continue
            raise
    warmup_seconds = time.perf_counter() - warmup_start

    # Timed phase
    per_batch_steps: dict[int, list[float]] = {}
    per_batch_atoms: dict[int, int] = {}
    per_system_pred: dict[str, InferenceResult] = {}
    oom_sizes: set[int] = set()

    timed_phase_start = time.perf_counter()
    for bsz, indices in schedule[warmup_steps:]:
        if bsz in oom_sizes:
            continue
        natoms = [int(atomic_data[i].natoms.item()) for i in indices]
        try:
            t0 = time.perf_counter()
            preds = _step(indices)
            dt = time.perf_counter() - t0
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if _is_oom(e) and oom_policy == "skip":
                logger.warning(
                    "OOM at batch_size=%d; skipping remaining %d-batches", bsz, bsz
                )
                oom_sizes.add(bsz)
                if is_cuda:
                    torch.cuda.empty_cache()
                continue
            raise

        per_batch_steps.setdefault(bsz, []).append(dt)
        per_batch_atoms[bsz] = per_batch_atoms.get(bsz, 0) + sum(natoms)

        # Capture per-system predictions; later steps overwrite earlier ones,
        # which is fine - we just need *some* prediction per system to compare.
        split = _split_batched_predictions(preds, natoms)
        for slot, i in enumerate(indices):
            per_system_pred[pool.systems[i].name] = _to_inference_result(split[slot])

    total_timed = time.perf_counter() - timed_phase_start

    per_batch_size: dict[int, BatchTiming] = {}
    for bsz, durations in per_batch_steps.items():
        total = float(sum(durations))
        n = len(durations)
        per_batch_size[bsz] = BatchTiming(
            batch_size=bsz,
            n_steps=n,
            total_seconds=total,
            samples_per_sec=(n * bsz) / total if total > 0 else 0.0,
            atoms_per_sec=per_batch_atoms[bsz] / total if total > 0 else 0.0,
        )

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if is_cuda else None

    return MixedInferenceResult(
        per_system=per_system_pred,
        per_batch_size=per_batch_size,
        warmup_seconds=warmup_seconds,
        total_timed_seconds=total_timed,
        peak_gpu_memory_mb=peak_mem,
        oom_batch_sizes=sorted(oom_sizes),
    )


def _is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower()
    )


def _mixed_baseline_cache_key(
    checkpoint: str,
    pool: SystemPool,
    device: str,
    seed: int,
    batch_sizes: tuple[int, ...],
) -> str:
    """
    Cache key for the per-system fp64 baselines used by the mixed runner.

    Independent from ``_baseline_cache_key`` so the two runners never alias
    each other's cache files. ``batch_sizes`` is included because changing them
    can change which OOM-skipped systems were probed (does not affect baseline
    correctness but keeps the cache scoped to a single benchmark config).
    """
    key_data = {
        "checkpoint": checkpoint,
        "pool": pool.signature(),
        "device": device,
        "seed": seed,
        "batch_sizes": list(batch_sizes),
        "baseline_settings": str(BASELINE_SETTINGS),
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def format_mixed_report_table(
    per_batch_size: dict[int, BatchTiming],
    per_system_errors: dict[str, dict[str, Any]],
) -> str:
    """
    Format the mixed-batch report as two stacked tables: per-batch-size throughput
    and per-system accuracy errors.
    """
    lines: list[str] = []

    perf_header = (
        f"{'Batch':>6} {'Steps':>6} {'Total(s)':>10} " f"{'samp/s':>10} {'atoms/s':>12}"
    )
    lines.append("Throughput by batch size:")
    lines.append(perf_header)
    lines.append("-" * len(perf_header))
    for bsz in sorted(per_batch_size):
        t = per_batch_size[bsz]
        lines.append(
            f"{t.batch_size:>6d} {t.n_steps:>6d} {t.total_seconds:>10.3f} "
            f"{t.samples_per_sec:>10.2f} {t.atoms_per_sec:>12.2f}"
        )

    lines.append("")
    acc_header = f"{'System':<28} {'E err(eV)':>12} {'F MAE':>12} {'F max':>12}"
    lines.append("Per-system error vs fp64 baseline:")
    lines.append(acc_header)
    lines.append("-" * len(acc_header))
    for name in sorted(per_system_errors):
        m = per_system_errors[name]
        if "error" in m:
            lines.append(f"{name:<28} {m['error']:>12}")
            continue
        lines.append(
            f"{name:<28} {m.get('energy_abs_error', float('nan')):>12.6f} "
            f"{m.get('force_mae', float('nan')):>12.6f} "
            f"{m.get('force_max_error', float('nan')):>12.6f}"
        )

    return "\n".join(lines)


class MixedPerfCheckRunner(Runner):
    """
    Benchmark mixed-size batched inference against per-system fp64 baselines.

    Builds a ``SystemPool`` of diverse systems (multiple UMA tasks, multiple
    size buckets), then runs a deterministic schedule of batches drawn from
    {4, 8, 16, 32, 64, 128, 256}-size buckets. Ground-truth predictions are
    computed once per pool entry at fp64 (re-using ``run_inference`` with
    ``BASELINE_SETTINGS``) and cached on disk, so subsequent runs only pay the
    benchmark cost.

    Usage:
        fairchem -c configs/uma/benchmark/perf_check/mixed_benchmark.yaml
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        batch_sizes: list[int] | tuple[int, ...] = DEFAULT_BATCH_SIZES,
        warmup_steps: int = 20,
        timed_steps: int = 200,
        seed: int = 42,
        pool_size_buckets: tuple[int, ...] = (20, 100, 500),
        pool_n_per_bucket: int = 2,
        pool_tasks: tuple[str, ...] = ("oc20", "omat", "omol", "odac", "omc"),
        oom_policy: str = "skip",
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa: B008
        overrides: dict | None = None,
    ):
        from fairchem.core.components.benchmark.systems import (
            get_diverse_benchmark_pool,
        )

        self.checkpoint = checkpoint
        self.device = device
        self.batch_sizes = tuple(int(b) for b in batch_sizes)
        self.warmup_steps = int(warmup_steps)
        self.timed_steps = int(timed_steps)
        self.seed = int(seed)
        self.oom_policy = oom_policy
        self.inference_settings = inference_settings
        # Model-construction overrides forwarded to the candidate predict unit
        # (e.g. {"backbone": {"moe_layer_type": "nvmath"}} to exercise the
        # nvmath/cuBLAS MOLEDGL path instead of the default pytorch MOLE path).
        # The fp64 accuracy baseline intentionally ignores these so both the
        # baseline and the nvmath candidate are scored against the same gold
        # reference. Normalize OmegaConf containers (passed via the Hydra CLI)
        # to plain Python so the run report stays JSON-serializable.
        if OmegaConf.is_config(overrides):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        self.overrides = overrides
        self.pool = get_diverse_benchmark_pool(
            seed=self.seed,
            size_buckets=tuple(pool_size_buckets),
            n_per_bucket=int(pool_n_per_bucket),
            tasks=tuple(pool_tasks),
        )

    def _baselines(self, cache_dir: str) -> dict[str, InferenceResult]:
        cache_path = os.path.join(cache_dir, MIXED_BASELINE_CACHE_FILE)
        cache_key = _mixed_baseline_cache_key(
            self.checkpoint, self.pool, self.device, self.seed, self.batch_sizes
        )
        cached = _load_baseline_cache(cache_path, cache_key)
        if cached is not None:
            logger.warning(
                "Using cached mixed baseline results from %s. "
                "Delete this file to force recomputation.",
                cache_path,
            )
            return cached

        logger.info(
            "Running per-system fp64 baselines for %d pool entries...",
            len(self.pool),
        )
        baselines: dict[str, InferenceResult] = {}
        for system in self.pool.systems:
            logger.info("  Baseline: %s (%d atoms)", system.name, len(system.atoms))
            baselines[system.name] = run_inference(
                checkpoint=self.checkpoint,
                system=system,
                inference_settings=BASELINE_SETTINGS,
                device=self.device,
                seed=self.seed,
            )
        _save_baseline_cache(cache_path, cache_key, baselines)
        return baselines

    def _build_predict_unit(self) -> MLIPPredictUnit:
        checkpoint = self.checkpoint
        if not os.path.exists(checkpoint):
            from fairchem.core.calculate.pretrained_mlip import (
                pretrained_checkpoint_path_from_name,
            )

            checkpoint = pretrained_checkpoint_path_from_name(checkpoint)
        return MLIPPredictUnit(
            checkpoint,
            self.device,
            overrides=self.overrides,
            inference_settings=self.inference_settings,
        )

    def run(self) -> dict:
        output_dir = self.job_config.metadata.results_dir
        os.makedirs(output_dir, exist_ok=True)
        cache_dir = self.job_config.run_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Step 1: per-system fp64 baselines (cached).
        baselines = self._baselines(cache_dir)

        # Step 2: deterministic schedule (warmup + timed).
        total_steps = self.warmup_steps + self.timed_steps
        schedule = build_batch_schedule(
            pool_size=len(self.pool),
            batch_sizes=self.batch_sizes,
            num_steps=total_steps,
            seed=self.seed,
        )

        # Step 3: benchmark.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        predict_unit = self._build_predict_unit()
        try:
            mixed = run_mixed_inference(
                predict_unit=predict_unit,
                pool=self.pool,
                schedule=schedule,
                warmup_steps=self.warmup_steps,
                device=self.device,
                oom_policy=self.oom_policy,
            )
        finally:
            del predict_unit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 4: per-system error vs cached baseline.
        per_system_errors: dict[str, dict[str, Any]] = {}
        for name, baseline in baselines.items():
            candidate = mixed.per_system.get(name)
            if candidate is None:
                per_system_errors[name] = {"error": "not benchmarked"}
                continue
            per_system_errors[name] = compare_results(baseline, candidate)

        # Step 5: report.
        table = format_mixed_report_table(mixed.per_batch_size, per_system_errors)
        logger.info("Mixed benchmark results:\n%s", table)

        report = {
            "baseline": {
                name: {"energy": r.energy, "num_atoms": r.forces.shape[0]}
                for name, r in baselines.items()
            },
            "inference_settings": str(self.inference_settings),
            "overrides": self.overrides,
            "batch_sizes": list(self.batch_sizes),
            "warmup_steps": self.warmup_steps,
            "timed_steps": self.timed_steps,
            "warmup_seconds": mixed.warmup_seconds,
            "total_timed_seconds": mixed.total_timed_seconds,
            "peak_gpu_memory_mb": mixed.peak_gpu_memory_mb,
            "oom_batch_sizes": mixed.oom_batch_sizes,
            "per_batch_size": {
                bsz: {
                    "n_steps": t.n_steps,
                    "total_seconds": t.total_seconds,
                    "samples_per_sec": t.samples_per_sec,
                    "atoms_per_sec": t.atoms_per_sec,
                }
                for bsz, t in mixed.per_batch_size.items()
            },
            "per_system": per_system_errors,
        }
        report_path = os.path.join(output_dir, MIXED_REPORT_FILE)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Mixed report saved to %s", report_path)
        return report

    def save_state(self, _):
        return

    def load_state(self, _):
        return


@dataclass
class GridCellTiming:
    """
    Forward-only timing for a single (system size, batch size) grid cell.
    """

    size: int
    batch: int
    reps: int
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    atoms_per_s: float
    mean_atoms_per_batch: float
    peak_gpu_memory_mb: float | None = None
    oom: bool = False


def _grid_baseline_cache_key(
    checkpoint: str,
    systems: list[BenchmarkSystem],
    device: str,
    seed: int,
    sizes: tuple[int, ...],
    variants_per_size: int,
    jitter: float,
    task: str,
) -> str:
    """
    Cache key for the per-system fp64 baselines used by the grid runner.

    Independent of the candidate backend (``overrides`` / ``use_grouped_gemm``)
    so pytorch and dgl runs share one baseline, but scoped to the pool geometry
    so a different grid cannot alias a stale baseline.
    """
    key_data = {
        "checkpoint": checkpoint,
        "systems": [{"name": s.name, "num_atoms": len(s.atoms)} for s in systems],
        "device": device,
        "seed": seed,
        "sizes": list(sizes),
        "variants_per_size": variants_per_size,
        "jitter": jitter,
        "task": task,
        "baseline_settings": str(BASELINE_SETTINGS),
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def _time_grid_cell(
    predict_unit: MLIPPredictUnit,
    atomic_pool: list,
    size: int,
    batch: int,
    reps: int,
    warmup: int,
    seed: int,
    device: str,
) -> GridCellTiming:
    """
    Time ``reps`` homogeneous batches of ``batch`` systems drawn from ``atomic_pool``.

    Collation is done outside the timed region; only ``predict`` is timed (CUDA
    events on GPU). A per-cell RNG keyed by (seed, size, batch) makes the sampled
    batches identical across backend runs regardless of OOM/order. Returns a cell
    with ``oom=True`` on out-of-memory.
    """
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    is_cuda = device == "cuda" and torch.cuda.is_available()
    rng = np.random.default_rng([seed, size, batch])

    def _one() -> tuple[float, int]:
        idx = rng.integers(0, len(atomic_pool), size=batch)
        data = atomicdata_list_to_batch([atomic_pool[i] for i in idx])
        natoms = int(sum(int(atomic_pool[i].natoms.item()) for i in idx))
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            predict_unit.predict(data)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), natoms
        t0 = time.perf_counter()
        predict_unit.predict(data)
        return (time.perf_counter() - t0) * 1000.0, natoms

    def _oom_cell() -> GridCellTiming:
        return GridCellTiming(
            size=size,
            batch=batch,
            reps=0,
            median_ms=float("nan"),
            mean_ms=float("nan"),
            std_ms=float("nan"),
            min_ms=float("nan"),
            max_ms=float("nan"),
            atoms_per_s=0.0,
            mean_atoms_per_batch=0.0,
            oom=True,
        )

    try:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            _one()
        durations: list[float] = []
        atoms: list[int] = []
        for _ in range(reps):
            dt, na = _one()
            durations.append(dt)
            atoms.append(na)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if _is_oom(e):
            if is_cuda:
                torch.cuda.empty_cache()
            return _oom_cell()
        raise

    arr = np.array(durations)
    med = float(np.median(arr))
    mean_atoms = float(np.mean(atoms))
    peak = torch.cuda.max_memory_allocated() / (1024**2) if is_cuda else None
    return GridCellTiming(
        size=size,
        batch=batch,
        reps=reps,
        median_ms=med,
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        atoms_per_s=(mean_atoms * 1000.0 / med) if med > 0 else 0.0,
        mean_atoms_per_batch=mean_atoms,
        peak_gpu_memory_mb=peak,
    )


def format_grid(
    cells: dict[tuple[int, int], GridCellTiming],
    sizes: tuple[int, ...],
    batches: tuple[int, ...],
    key: str,
    fmt: str,
    title: str,
) -> str:
    """
    Render a size x batch grid of one ``GridCellTiming`` field as a table.
    """
    label = "size\\batch"
    lines = [title]
    header = f"{label:>12}" + "".join(f"{b:>12}" for b in batches)
    lines.append(header)
    lines.append("-" * len(header))
    for s in sizes:
        row = f"{s:>12}"
        for b in batches:
            cell = cells.get((s, b))
            if cell is None or cell.oom:
                row += f"{'OOM':>12}"
            else:
                row += f"{getattr(cell, key):{fmt}}".rjust(12)
        lines.append(row)
    return "\n".join(lines)


class GridPerfCheckRunner(Runner):
    """
    Benchmark inference across a homogeneous system-size x batch-size grid.

    For each (size, batch) cell, times forward inference on batches whose systems
    are all drawn from a tight window around the target size, and optionally
    validates accuracy against a cached fp64 baseline. Run once per backend
    (pytorch vs dgl-grouped vs dgl-looped); pass ``reference_report`` on a later
    run to print a speedup grid vs that report.

    Usage:
        fairchem -c configs/uma/benchmark/perf_check/grid_benchmark.yaml \
            '+runner.overrides={backbone:{moe_layer_type:pytorch}}'
        fairchem -c configs/uma/benchmark/perf_check/grid_benchmark.yaml \
            '+runner.overrides={backbone:{moe_layer_type:dgl}}' \
            runner.reference_report=<baseline_run>/grid_benchmark_report.json
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        sizes: tuple[int, ...] = DEFAULT_GRID_SIZES,
        batches: tuple[int, ...] = DEFAULT_GRID_BATCHES,
        variants_per_size: int = 4,
        size_jitter: float = 0.1,
        task: str = "omat",
        reps: int = 20,
        warmup: int = 5,
        seed: int = 42,
        use_grouped_gemm: bool = True,
        check_accuracy: bool = True,
        overrides: dict | None = None,
        reference_report: str | None = None,
        inference_settings: InferenceSettings = inference_settings_default(),  # noqa: B008
    ):
        from fairchem.core.components.benchmark.systems import get_size_bucket_pool

        self.checkpoint = checkpoint
        self.device = device
        self.sizes = tuple(int(s) for s in sizes)
        self.batches = tuple(int(b) for b in batches)
        self.variants_per_size = int(variants_per_size)
        self.size_jitter = float(size_jitter)
        self.task = task
        self.reps = int(reps)
        self.warmup = int(warmup)
        self.seed = int(seed)
        self.use_grouped_gemm = bool(use_grouped_gemm)
        self.check_accuracy = bool(check_accuracy)
        # Candidate model overrides (e.g. {"backbone": {"moe_layer_type": "nvmath"}}).
        # Normalized to plain Python so the run report stays JSON-serializable.
        if OmegaConf.is_config(overrides):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        self.overrides = overrides
        self.reference_report = reference_report
        self.inference_settings = inference_settings
        self.pool = get_size_bucket_pool(
            sizes=self.sizes,
            variants_per_size=self.variants_per_size,
            jitter=self.size_jitter,
            task=self.task,
            seed=self.seed,
        )

    def _flat_systems(self) -> list[BenchmarkSystem]:
        return [s for size in self.sizes for s in self.pool[size]]

    def _baselines(self, cache_dir: str) -> dict[str, InferenceResult]:
        systems = self._flat_systems()
        cache_path = os.path.join(cache_dir, GRID_BASELINE_CACHE_FILE)
        cache_key = _grid_baseline_cache_key(
            self.checkpoint,
            systems,
            self.device,
            self.seed,
            self.sizes,
            self.variants_per_size,
            self.size_jitter,
            self.task,
        )
        cached = _load_baseline_cache(cache_path, cache_key)
        if cached is not None:
            logger.warning(
                "Using cached grid baseline results from %s. "
                "Delete this file to force recomputation.",
                cache_path,
            )
            return cached
        logger.info("Running per-system fp64 baselines for %d systems...", len(systems))
        baselines: dict[str, InferenceResult] = {}
        for system in systems:
            baselines[system.name] = run_inference(
                checkpoint=self.checkpoint,
                system=system,
                inference_settings=BASELINE_SETTINGS,
                device=self.device,
                seed=self.seed,
            )
        _save_baseline_cache(cache_path, cache_key, baselines)
        return baselines

    def _build_predict_unit(self) -> MLIPPredictUnit:
        checkpoint = self.checkpoint
        if not os.path.exists(checkpoint):
            from fairchem.core.calculate.pretrained_mlip import (
                pretrained_checkpoint_path_from_name,
            )

            checkpoint = pretrained_checkpoint_path_from_name(checkpoint)
        # Select grouped vs looped GEMM through the backbone config (not a module
        # global): the MOLEDGL backends read moe_use_grouped_gemm at construction.
        overrides = dict(self.overrides) if self.overrides else {}
        backbone = dict(overrides.get("backbone", {}))
        backbone["moe_use_grouped_gemm"] = self.use_grouped_gemm
        overrides["backbone"] = backbone
        return MLIPPredictUnit(
            checkpoint,
            self.device,
            overrides=overrides,
            inference_settings=self.inference_settings,
        )

    def run(self) -> dict:
        from fairchem.core.datasets.atomic_data import (
            AtomicData,
            atomicdata_list_to_batch,
        )

        output_dir = self.job_config.metadata.results_dir
        os.makedirs(output_dir, exist_ok=True)
        cache_dir = self.job_config.run_dir
        os.makedirs(cache_dir, exist_ok=True)

        baselines = self._baselines(cache_dir) if self.check_accuracy else {}

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # grouped/looped is selected via backbone overrides in _build_predict_unit.
        predict_unit = self._build_predict_unit()
        atomic_pool = {
            size: [
                AtomicData.from_ase(s.atoms, task_name=s.task_name)
                for s in self.pool[size]
            ]
            for size in self.sizes
        }

        cells: dict[tuple[int, int], GridCellTiming] = {}
        per_system_errors: dict[str, dict[str, Any]] = {}
        try:
            for size in self.sizes:
                for batch in self.batches:
                    logger.info("  size=%d batch=%d", size, batch)
                    cells[(size, batch)] = _time_grid_cell(
                        predict_unit,
                        atomic_pool[size],
                        size=size,
                        batch=batch,
                        reps=self.reps,
                        warmup=self.warmup,
                        seed=self.seed,
                        device=self.device,
                    )
            if self.check_accuracy:
                for size in self.sizes:
                    natoms = [int(d.natoms.item()) for d in atomic_pool[size]]
                    try:
                        batch_data = atomicdata_list_to_batch(atomic_pool[size])
                        preds = predict_unit.predict(batch_data)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        split = _split_batched_predictions(preds, natoms)
                        for slot, system in enumerate(self.pool[size]):
                            cand = _to_inference_result(split[slot])
                            per_system_errors[system.name] = compare_results(
                                baselines[system.name], cand
                            )
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if _is_oom(e):
                            for system in self.pool[size]:
                                per_system_errors[system.name] = {"error": "OOM"}
                        else:
                            raise
        finally:
            del predict_unit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "Grid results:\n%s\n\n%s",
            format_grid(
                cells,
                self.sizes,
                self.batches,
                "median_ms",
                ".2f",
                "MEDIAN forward time (ms)",
            ),
            format_grid(
                cells, self.sizes, self.batches, "atoms_per_s", ".0f", "atoms/s"
            ),
        )

        report = {
            "checkpoint": self.checkpoint,
            "overrides": self.overrides,
            "use_grouped_gemm": self.use_grouped_gemm,
            "inference_settings": str(self.inference_settings),
            "sizes": list(self.sizes),
            "batches": list(self.batches),
            "task": self.task,
            "reps": self.reps,
            "cells": [asdict(c) for c in cells.values()],
            "accuracy": per_system_errors,
        }
        report_path = os.path.join(output_dir, GRID_REPORT_FILE)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Grid report saved to %s", report_path)

        if self.reference_report is not None:
            self._print_speedup(cells)

        return report

    def _print_speedup(self, cells: dict[tuple[int, int], GridCellTiming]) -> None:
        try:
            with open(self.reference_report) as f:
                ref = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Could not read reference_report %s: %s", self.reference_report, e
            )
            return
        if ref.get("checkpoint") != self.checkpoint:
            logger.warning(
                "reference_report checkpoint %s != %s; speedup may be meaningless",
                ref.get("checkpoint"),
                self.checkpoint,
            )
        ref_ms = {
            (c["size"], c["batch"]): c["median_ms"]
            for c in ref.get("cells", [])
            if not c.get("oom")
        }
        label = "size\\batch"
        header = f"{label:>12}" + "".join(f"{b:>12}" for b in self.batches)
        lines = [
            "SPEEDUP = reference_median / this_median  (>1.0 => this run FASTER)",
            header,
            "-" * len(header),
        ]
        for s in self.sizes:
            row = f"{s:>12}"
            for b in self.batches:
                cell = cells.get((s, b))
                rm = ref_ms.get((s, b))
                if cell is None or cell.oom or rm is None or cell.median_ms <= 0:
                    row += f"{'-':>12}"
                else:
                    row += f"{rm / cell.median_ms:>12.2f}"
            lines.append(row)
        logger.info("\n%s", "\n".join(lines))

    def save_state(self, _):
        return

    def load_state(self, _):
        return
