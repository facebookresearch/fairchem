"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetrics:
    """
    Aggregated metrics from one distributed training run.
    """

    results_dir: Path
    world_size: int
    steps: int
    seconds_per_step: float
    synchronized_seconds_per_step: float
    total_train_seconds: float
    peak_allocated_gib_mean: float
    peak_allocated_gib_max: float
    peak_reserved_gib_mean: float
    peak_reserved_gib_max: float
    validation_history: tuple[dict[str, float], ...]
    rank_results: tuple[dict[str, Any], ...]

    @property
    def steps_per_second(self) -> float:
        return 1.0 / self.seconds_per_step


def _candidate_results_dirs(root: Path) -> list[Path]:
    if root.name == "results":
        candidates = [root]
    else:
        candidates = list(root.glob("*/results"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def load_latest_complete_run(root: Path, warmup_steps: int) -> RunMetrics:
    """
    Load the newest complete distributed result under a run root.

    Args:
        root: Run root containing timestamped result directories, or a results
            directory itself.
        warmup_steps: Initial timing samples to discard.

    Returns:
        Aggregated metrics for the newest complete run.

    Raises:
        FileNotFoundError: If no complete result set exists.
        ValueError: If the result history is too short.
    """
    for results_dir in _candidate_results_dirs(root):
        paths = sorted(results_dir.glob("training_metrics.rank*.pkl"))
        if not paths:
            continue
        rank_results = tuple(pickle.loads(path.read_bytes()) for path in paths)
        world_size = int(rank_results[0]["world_size"])
        ranks = {int(result["rank"]) for result in rank_results}
        if len(rank_results) != world_size or ranks != set(range(world_size)):
            continue

        step_count = min(len(result["step_times"]) for result in rank_results)
        if step_count <= warmup_steps:
            raise ValueError(
                f"Run {results_dir} has {step_count} steps, which does not exceed "
                f"the {warmup_steps} warm-up steps"
            )

        rank_means = [
            statistics.mean(result["step_times"][warmup_steps:step_count])
            for result in rank_results
        ]
        synchronized_step_means = [
            max(result["step_times"][step] for result in rank_results)
            for step in range(warmup_steps, step_count)
        ]
        allocated = [result["peak_memory_mb"] / 1024 for result in rank_results]
        reserved = [result["peak_reserved_memory_mb"] / 1024 for result in rank_results]
        rank_zero_result = next(
            result for result in rank_results if int(result["rank"]) == 0
        )
        return RunMetrics(
            results_dir=results_dir,
            world_size=world_size,
            steps=step_count,
            seconds_per_step=max(rank_means),
            synchronized_seconds_per_step=statistics.mean(synchronized_step_means),
            total_train_seconds=max(
                float(result["total_train_seconds"]) for result in rank_results
            ),
            peak_allocated_gib_mean=statistics.mean(allocated),
            peak_allocated_gib_max=max(allocated),
            peak_reserved_gib_mean=statistics.mean(reserved),
            peak_reserved_gib_max=max(reserved),
            validation_history=tuple(rank_zero_result.get("validation_history", [])),
            rank_results=rank_results,
        )

    raise FileNotFoundError(f"No complete training result found under {root}")


def _maximum_errors(
    baseline: RunMetrics,
    candidate: RunMetrics,
    key: str,
) -> tuple[float, float]:
    absolute_error = 0.0
    relative_error = 0.0
    value_count = 0
    baseline_by_rank = {int(result["rank"]): result for result in baseline.rank_results}
    candidate_by_rank = {
        int(result["rank"]): result for result in candidate.rank_results
    }
    for rank in sorted(baseline_by_rank):
        baseline_values = baseline_by_rank[rank][key]
        candidate_values = candidate_by_rank[rank][key]
        if len(baseline_values) != len(candidate_values):
            raise ValueError(f"Rank {rank} has different {key} history lengths")
        for baseline_value, candidate_value in zip(baseline_values, candidate_values):
            if baseline_value is None or candidate_value is None:
                if baseline_value != candidate_value:
                    raise ValueError(f"Rank {rank} has mismatched missing {key}")
                continue
            difference = abs(float(candidate_value) - float(baseline_value))
            value_count += 1
            absolute_error = max(absolute_error, difference)
            relative_error = max(
                relative_error,
                difference / max(abs(float(baseline_value)), 1e-12),
            )
    if value_count == 0:
        return math.nan, math.nan
    return absolute_error, relative_error


def compare_runs(
    baseline: RunMetrics,
    zero: RunMetrics,
) -> dict[str, float]:
    """
    Compare fidelity and performance for baseline and ZeRO runs.
    """
    if baseline.world_size != zero.world_size:
        raise ValueError("Baseline and ZeRO world sizes differ")
    if baseline.steps != zero.steps:
        raise ValueError("Baseline and ZeRO step counts differ")

    loss_absolute, loss_relative = _maximum_errors(baseline, zero, "losses")
    grad_absolute, grad_relative = _maximum_errors(baseline, zero, "grad_norms")
    comparison = {
        "speedup_percent": 100
        * (baseline.seconds_per_step / zero.seconds_per_step - 1),
        "total_runtime_speedup_percent": 100
        * (baseline.total_train_seconds / zero.total_train_seconds - 1),
        "allocated_memory_delta_gib_mean": zero.peak_allocated_gib_mean
        - baseline.peak_allocated_gib_mean,
        "reserved_memory_delta_gib_mean": zero.peak_reserved_gib_mean
        - baseline.peak_reserved_gib_mean,
        "loss_max_absolute_error": loss_absolute,
        "loss_max_relative_error": loss_relative,
        "grad_norm_max_absolute_error": grad_absolute,
        "grad_norm_max_relative_error": grad_relative,
    }
    if baseline.validation_history and zero.validation_history:
        baseline_losses = [result["val_loss"] for result in baseline.validation_history]
        zero_losses = [result["val_loss"] for result in zero.validation_history]
        common_target = max(min(baseline_losses), min(zero_losses))

        def time_to_target(run: RunMetrics) -> float:
            return next(
                result["elapsed_seconds"]
                for result in run.validation_history
                if result["val_loss"] <= common_target
            )

        baseline_time = time_to_target(baseline)
        zero_time = time_to_target(zero)
        comparison.update(
            {
                "baseline_final_val_loss": baseline_losses[-1],
                "zero_final_val_loss": zero_losses[-1],
                "baseline_best_val_loss": min(baseline_losses),
                "zero_best_val_loss": min(zero_losses),
                "common_val_loss_target": common_target,
                "baseline_time_to_target_seconds": baseline_time,
                "zero_time_to_target_seconds": zero_time,
                "time_to_target_speedup_percent": 100 * (baseline_time / zero_time - 1),
            }
        )
    return comparison


def _format_run(name: str, run: RunMetrics) -> str:
    return (
        f"| {name} | {run.total_train_seconds / 3600:.3f} | "
        f"{run.seconds_per_step:.6f} | "
        f"{run.steps_per_second:.4f} | {run.peak_allocated_gib_mean:.3f} / "
        f"{run.peak_allocated_gib_max:.3f} | "
        f"{run.peak_reserved_gib_mean:.3f} / "
        f"{run.peak_reserved_gib_max:.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare distributed SOAP training with and without ZeRO"
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--fidelity-rtol", type=float, default=1e-5)
    args = parser.parse_args()

    baseline = load_latest_complete_run(args.baseline_root, args.warmup_steps)
    zero = load_latest_complete_run(args.zero_root, args.warmup_steps)
    comparison = compare_runs(baseline, zero)
    fidelity_captured = not math.isnan(comparison["loss_max_relative_error"])
    fidelity_passed = fidelity_captured and (
        comparison["loss_max_relative_error"] <= args.fidelity_rtol
        and comparison["grad_norm_max_relative_error"] <= args.fidelity_rtol
    )

    print(
        f"Compared {baseline.steps} steps on {baseline.world_size} ranks; "
        f"discarded {args.warmup_steps} warm-up steps."
    )
    print(
        "| Run | total hours | seconds/step | steps/s | "
        "allocated GiB mean/max | reserved GiB mean/max |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    print(_format_run("No ZeRO", baseline))
    print(_format_run("ZeRO", zero))
    print(f"ZeRO speedup: {comparison['speedup_percent']:+.2f}%")
    print(
        "ZeRO end-to-end training runtime delta: "
        f"{comparison['total_runtime_speedup_percent']:+.2f}%"
    )
    print(
        "Mean memory delta: "
        f"{comparison['allocated_memory_delta_gib_mean']:+.3f} GiB allocated, "
        f"{comparison['reserved_memory_delta_gib_mean']:+.3f} GiB reserved"
    )
    if "common_val_loss_target" in comparison:
        print(
            "Validation loss (final / best): "
            f"no ZeRO {comparison['baseline_final_val_loss']:.6g} / "
            f"{comparison['baseline_best_val_loss']:.6g}; "
            f"ZeRO {comparison['zero_final_val_loss']:.6g} / "
            f"{comparison['zero_best_val_loss']:.6g}"
        )
        print(
            f"Time to common val/loss {comparison['common_val_loss_target']:.6g}: "
            f"no ZeRO {comparison['baseline_time_to_target_seconds'] / 3600:.3f} h, "
            f"ZeRO {comparison['zero_time_to_target_seconds'] / 3600:.3f} h "
            f"({comparison['time_to_target_speedup_percent']:+.2f}%)"
        )
    else:
        print("Validation history: not captured")
    if fidelity_captured:
        print(
            "Loss error (max abs / rel): "
            f"{comparison['loss_max_absolute_error']:.6g} / "
            f"{comparison['loss_max_relative_error']:.6g}"
        )
        print(
            "Grad-norm error (max abs / rel): "
            f"{comparison['grad_norm_max_absolute_error']:.6g} / "
            f"{comparison['grad_norm_max_relative_error']:.6g}"
        )
        print(
            f"Step fidelity at rtol={args.fidelity_rtol:g}: "
            f"{'PASS' if fidelity_passed else 'REVIEW'}"
        )
    else:
        print("Step fidelity: not captured; compare validation curves in W&B")


if __name__ == "__main__":
    main()
