"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.components.benchmark.training import (
    BenchmarkTrainCallback,
    TrainingBenchmarkResult,
    run_training_benchmark,
)
from fairchem.core.scripts.compare_zero_training import (
    compare_runs,
    load_latest_complete_run,
)

TRAINING_CONFIG = "configs/uma/benchmark/perf_check/training_inner.yaml"


def test_training_benchmark_smoke():
    result = run_training_benchmark(
        device="cpu",
        bf16=False,
        throughput_steps=2,
        training_config=TRAINING_CONFIG,
    )
    assert isinstance(result, TrainingBenchmarkResult)
    assert result.steps_per_second > 0
    assert result.loss_abs_error >= 0


def test_benchmark_callback_writes_rank_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fairchem.core.components.benchmark.training.distutils.get_rank",
        lambda: 3,
    )
    monkeypatch.setattr(
        "fairchem.core.components.benchmark.training.distutils.get_world_size",
        lambda: 8,
    )
    callback = BenchmarkTrainCallback(str(tmp_path / "metrics.pkl"), rank_suffix=True)
    unit = type(
        "Unit",
        (),
        {
            "last_loss": torch.tensor(1.25),
            "last_grad_norm": torch.tensor(2.5),
        },
    )()
    callback.on_train_start(None, unit)
    callback.on_train_step_start(None, unit)
    callback.on_train_step_end(None, unit)
    callback.on_train_end(None, unit)

    with open(tmp_path / "metrics.rank3.pkl", "rb") as handle:
        result = pickle.load(handle)
    assert result["rank"] == 3
    assert result["world_size"] == 8
    assert result["losses"] == [1.25]
    assert result["grad_norms"] == [2.5]
    assert result["total_train_seconds"] >= 0
    assert result["validation_history"] == []


def test_benchmark_callback_records_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fairchem.core.components.benchmark.training.distutils.is_master",
        lambda: True,
    )
    callback = BenchmarkTrainCallback(str(tmp_path / "metrics.pkl"))
    unit = SimpleNamespace(
        eval_unit=SimpleNamespace(
            total_loss_metrics=SimpleNamespace(metric=torch.tensor(0.125))
        ),
        train_progress=SimpleNamespace(num_steps_completed=2500),
    )
    callback.on_train_start(None, unit)
    callback.on_eval_epoch_end(None, unit)

    assert callback.validation_history[0]["step"] == 2500
    assert callback.validation_history[0]["val_loss"] == 0.125
    assert callback.validation_history[0]["elapsed_seconds"] >= 0


def test_compare_zero_training_results(tmp_path):
    def write_run(name, seconds, loss_offset=0.0):
        results_dir = tmp_path / name / "timestamp" / "results"
        results_dir.mkdir(parents=True)
        for rank in range(2):
            result = {
                "rank": rank,
                "world_size": 2,
                "losses": [1.0 + loss_offset, 0.5 + loss_offset],
                "grad_norms": [2.0 + loss_offset, 1.0 + loss_offset],
                "step_times": [seconds, seconds],
                "peak_memory_mb": 1024 + rank,
                "peak_reserved_memory_mb": 2048 + rank,
                "total_train_seconds": seconds * 10,
                "validation_history": (
                    [
                        {
                            "step": 10,
                            "elapsed_seconds": seconds * 5,
                            "val_loss": 0.8 + loss_offset,
                        },
                        {
                            "step": 20,
                            "elapsed_seconds": seconds * 10,
                            "val_loss": 0.4 + loss_offset,
                        },
                    ]
                    if rank == 0
                    else []
                ),
            }
            with open(results_dir / f"training_metrics.rank{rank}.pkl", "wb") as handle:
                pickle.dump(result, handle)
        return load_latest_complete_run(tmp_path / name, warmup_steps=1)

    baseline = write_run("baseline", 2.0)
    zero = write_run("zero", 1.0, loss_offset=1e-6)
    comparison = compare_runs(baseline, zero)

    assert comparison["speedup_percent"] == pytest.approx(100.0)
    assert comparison["loss_max_absolute_error"] == pytest.approx(1e-6)
    assert comparison["time_to_target_speedup_percent"] == pytest.approx(100.0)
