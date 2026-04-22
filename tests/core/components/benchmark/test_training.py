"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.components.benchmark.training import (
    TrainingBenchmarkResult,
    run_training_benchmark,
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
