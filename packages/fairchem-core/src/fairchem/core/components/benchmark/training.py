"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torchtnt.framework.callback import Callback

from fairchem.core.components.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class TrainingBenchmarkResult:
    """
    Fidelity and throughput metrics from a training benchmark run.
    """

    loss_abs_error: float
    loss_rel_error: float
    grad_norm_rel_error: float
    steps_per_second: float
    baseline_steps_per_second: float
    peak_memory_mb: float
    baseline_peak_memory_mb: float
    baseline_loss: float
    candidate_loss: float
    baseline_final_loss: float
    candidate_final_loss: float
    baseline_grad_norm: float | None
    candidate_grad_norm: float | None


class BenchmarkTrainCallback(Callback):
    """
    TorchTNT callback that captures per-step training metrics for benchmarking.
    """

    def __init__(self, benchmark_results_path: str):
        self.benchmark_results_path = benchmark_results_path
        self.losses: list[float] = []
        self.grad_norms: list[float | None] = []
        self.step_times: list[float] = []
        self.peak_memory_mb: float = 0.0
        self._step_start_time: float | None = None

    def on_train_start(self, state, unit) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_step_start(self, state, unit) -> None:
        self._step_start_time = time.perf_counter()

    def on_train_step_end(self, state, unit) -> None:
        if self._step_start_time is not None:
            self.step_times.append(time.perf_counter() - self._step_start_time)
        if unit.last_loss is not None:
            self.losses.append(unit.last_loss)
        self.grad_norms.append(
            unit.last_grad_norm if hasattr(unit, "last_grad_norm") else None
        )
        if torch.cuda.is_available():
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def on_train_end(self, state, unit) -> None:
        results = {
            "losses": self.losses,
            "grad_norms": self.grad_norms,
            "step_times": self.step_times,
            "peak_memory_mb": self.peak_memory_mb,
        }
        with open(self.benchmark_results_path, "wb") as f:
            pickle.dump(results, f)


def _run_single(
    config_path: str,
    data_root_dir: str,
    device: str,
    bf16: bool,
    max_steps: int,
    results_path: str,
    seed: int,
    extra_overrides: list[str] | None = None,
) -> dict:
    """
    Run a single training session and return the pickled results.
    """
    import numpy as np
    from hydra.core.global_hydra import GlobalHydra
    from torch import distributed as dist

    from fairchem.core._cli import main

    GlobalHydra.instance().clear()

    # Tear down any existing process group so the inner main() can
    # re-initialize it (needed when called from the CLI runner which
    # already set up distributed).
    if dist.is_initialized():
        dist.destroy_process_group()

    torch.manual_seed(seed)
    np.random.seed(seed)

    args = argparse.Namespace(config=config_path)
    override_args = [
        f"datasets.data_root_dir={data_root_dir}",
        f"job.device_type={device.upper()}",
        f"bf16={bf16}",
        f"max_steps={max_steps}",
        f"runner.callbacks.0.benchmark_results_path={results_path}",
    ]
    if extra_overrides:
        override_args.extend(extra_overrides)

    main(args=args, override_args=override_args)

    with open(results_path, "rb") as f:
        return pickle.load(f)


def run_training_benchmark(
    training_config: str,
    data_root_dir: str | None = None,
    device: str = "cpu",
    bf16: bool = True,
    throughput_steps: int = 10,
    seed: int = 42,
    candidate_overrides: list[str] | None = None,
) -> TrainingBenchmarkResult:
    """
    Run a training benchmark comparing fp32 baseline against candidate settings.

    Args:
        training_config: Path to the training YAML config.
        data_root_dir: Path to the dataset root directory.
            If None, generates fake benchmark datasets automatically.
        device: Device to run on ("cpu" or "cuda").
        bf16: Whether to enable bf16 for the candidate run.
        throughput_steps: Number of training steps for each run.
        seed: Random seed for reproducibility.
        candidate_overrides: Extra Hydra overrides applied only to the
            candidate run (e.g. ["moe_layer_type=cpu_blas"]).

    Returns:
        TrainingBenchmarkResult with fidelity and throughput metrics.
    """
    if data_root_dir is None:
        from fairchem.core.components.benchmark.fake_dataset import (
            create_fake_benchmark_dataset,
        )

        cache_dir = os.path.join(tempfile.gettempdir(), "fairchem_benchmark_datasets")
        data_root_dir = cache_dir
        create_fake_benchmark_dataset(data_root_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate.pkl")

        # Run 1: fp32 baseline (cached across runs)
        baseline_cache_dir = os.path.join(
            tempfile.gettempdir(), "fairchem_benchmark_baseline"
        )
        os.makedirs(baseline_cache_dir, exist_ok=True)
        cache_key = hashlib.sha256(
            f"{training_config}:{data_root_dir}:{device}:{throughput_steps}:{seed}".encode()
        ).hexdigest()[:16]
        baseline_cache_path = os.path.join(
            baseline_cache_dir, f"baseline_{cache_key}.pkl"
        )

        if os.path.exists(baseline_cache_path):
            logger.warning(
                "Using cached fp32 baseline from %s. "
                "Delete this file to force a fresh baseline run.",
                baseline_cache_path,
            )
            with open(baseline_cache_path, "rb") as f:
                baseline = pickle.load(f)
        else:
            logger.info("Running fp32 baseline training...")
            baseline_tmp_path = os.path.join(tmpdir, "baseline.pkl")
            try:
                baseline = _run_single(
                    config_path=training_config,
                    data_root_dir=data_root_dir,
                    device=device,
                    bf16=False,
                    max_steps=throughput_steps,
                    results_path=baseline_tmp_path,
                    seed=seed,
                )
            except Exception as e:
                raise RuntimeError(f"Baseline training run failed: {e}") from e
            with open(baseline_cache_path, "wb") as f:
                pickle.dump(baseline, f)

        # Run 2: candidate settings
        logger.info(f"Running candidate training (bf16={bf16})...")
        try:
            candidate = _run_single(
                config_path=training_config,
                data_root_dir=data_root_dir,
                device=device,
                bf16=bf16,
                max_steps=throughput_steps,
                results_path=candidate_path,
                seed=seed,
                extra_overrides=candidate_overrides,
            )
        except Exception as e:
            raise RuntimeError(f"Candidate training run failed: {e}") from e

    # Fidelity: compare step 0 losses and grad norms
    baseline_loss = baseline["losses"][0]
    candidate_loss = candidate["losses"][0]
    baseline_final_loss = baseline["losses"][-1]
    candidate_final_loss = candidate["losses"][-1]
    loss_abs_error = abs(baseline_loss - candidate_loss)
    loss_rel_error = loss_abs_error / abs(baseline_loss) if baseline_loss != 0 else 0.0

    baseline_grad_norm = baseline["grad_norms"][0]
    candidate_grad_norm = candidate["grad_norms"][0]
    if baseline_grad_norm is not None and candidate_grad_norm is not None:
        grad_norm_rel_error = (
            abs(baseline_grad_norm - candidate_grad_norm) / abs(baseline_grad_norm)
            if baseline_grad_norm != 0
            else 0.0
        )
    else:
        grad_norm_rel_error = 0.0

    # Throughput: discard warmup step (step 0)
    baseline_step_times = baseline["step_times"][1:]
    if baseline_step_times:
        baseline_steps_per_second = len(baseline_step_times) / sum(baseline_step_times)
    else:
        baseline_steps_per_second = 0.0

    candidate_step_times = candidate["step_times"][1:]
    if candidate_step_times:
        steps_per_second = len(candidate_step_times) / sum(candidate_step_times)
    else:
        steps_per_second = 0.0

    peak_memory_mb = candidate["peak_memory_mb"]
    baseline_peak_memory_mb = baseline["peak_memory_mb"]

    result = TrainingBenchmarkResult(
        loss_abs_error=loss_abs_error,
        loss_rel_error=loss_rel_error,
        grad_norm_rel_error=grad_norm_rel_error,
        steps_per_second=steps_per_second,
        baseline_steps_per_second=baseline_steps_per_second,
        peak_memory_mb=peak_memory_mb,
        baseline_peak_memory_mb=baseline_peak_memory_mb,
        baseline_loss=baseline_loss,
        candidate_loss=candidate_loss,
        baseline_final_loss=baseline_final_loss,
        candidate_final_loss=candidate_final_loss,
        baseline_grad_norm=baseline_grad_norm,
        candidate_grad_norm=candidate_grad_norm,
    )

    speedup = (
        steps_per_second / baseline_steps_per_second
        if baseline_steps_per_second > 0
        else float("nan")
    )
    baseline_gn = (
        f"{baseline_grad_norm:.4f}" if baseline_grad_norm is not None else "N/A"
    )
    candidate_gn = (
        f"{candidate_grad_norm:.4f}" if candidate_grad_norm is not None else "N/A"
    )

    sep = "-" * 62
    header = f"{'Metric':<28} {'Baseline':>15} {'Candidate':>15}"
    logger.info(
        "Training Benchmark Results:\n"
        f"  {sep}\n"
        f"  {header}\n"
        f"  {sep}\n"
        f"  {'Loss (step 0)':<28} {baseline_loss:>15.6f} {candidate_loss:>15.6f}\n"
        f"  {'Loss (final step)':<28} {baseline_final_loss:>15.6f} {candidate_final_loss:>15.6f}\n"
        f"  {'Grad norm (step 0)':<28} {baseline_gn:>15} {candidate_gn:>15}\n"
        f"  {'Steps/second':<28} {baseline_steps_per_second:>15.2f} {steps_per_second:>15.2f}\n"
        f"  {'Peak memory (MB)':<28} {baseline_peak_memory_mb:>15.1f} {peak_memory_mb:>15.1f}\n"
        f"  {sep}\n"
        f"  {'Loss abs error (step 0)':<28} {loss_abs_error:.6e}\n"
        f"  {'Loss rel error (step 0)':<28} {loss_rel_error:.6e}\n"
        f"  {'Grad norm rel error':<28} {grad_norm_rel_error:.6e}\n"
        f"  {'Speedup (candidate/base)':<28} {speedup:.2f}x"
    )

    return result


class TrainingBenchmarkRunner(Runner):
    """
    Benchmark training fidelity and throughput via the fairchem CLI.

    Runs an fp32 baseline and a candidate (optionally bf16) training session,
    then compares loss, gradient norms, throughput, and memory usage.

    Usage via fairchem CLI:
        fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml
        fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml \
            runner.bf16=True
        fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml \
            runner.device=cpu runner.throughput_steps=5
    """

    def __init__(
        self,
        training_config: str,
        device: str = "cuda",
        bf16: bool = True,
        throughput_steps: int = 10,
        seed: int = 42,
        candidate_overrides: list[str] | None = None,
    ):
        self.training_config = training_config
        self.device = device
        self.bf16 = bf16
        self.throughput_steps = throughput_steps
        self.seed = seed
        self.candidate_overrides = candidate_overrides

    def run(self) -> dict[str, Any]:
        """
        Run the training benchmark and save a JSON report.

        Returns:
            Dict with training benchmark metrics.
        """
        output_dir = self.job_config.metadata.results_dir
        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(output_dir, "training_benchmark.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        try:
            result = run_training_benchmark(
                device=self.device,
                bf16=self.bf16,
                throughput_steps=self.throughput_steps,
                seed=self.seed,
                training_config=self.training_config,
                candidate_overrides=self.candidate_overrides,
            )

            report = asdict(result)
            report_path = os.path.join(output_dir, "training_benchmark_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Training benchmark report saved to %s", report_path)
            logger.info("Training benchmark log saved to %s", log_path)
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        return report

    def save_state(self, _):
        return

    def load_state(self, _):
        return
