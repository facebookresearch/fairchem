"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from fairchem.core.components.benchmark.perf_check import (
    BASELINE_SETTINGS,
    InferenceResult,
    PerfCheckRunner,
    PerfMetrics,
    compare_results,
    format_report_table,
    run_inference,
)
from fairchem.core.components.benchmark.systems import make_benchmark_system
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def test_compare_results_and_format_table():
    """
    Verify aggregate error computation and table formatting together.
    """
    baseline = InferenceResult(
        energy=-10.0,
        forces=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        stress=np.eye(3),
    )
    candidate = InferenceResult(
        energy=-10.01,
        forces=np.array([[1.01, 2.0, 2.99], [4.0, 5.02, 6.0]]),
        stress=np.eye(3) + 0.01,
    )
    perf = PerfMetrics(
        qps=100.0,
        wall_time_seconds=0.2,
        peak_gpu_memory_mb=1024.0,
        warmup_time_seconds=1.0,
    )
    metrics = compare_results([baseline], [candidate], perf)
    assert metrics["n_systems"] == 1
    assert metrics["energy_abs_error_mean"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["energy_abs_error_max"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["force_max_error"] == pytest.approx(0.02, abs=1e-6)
    assert metrics["stress_mae_mean"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["qps"] == 100.0

    # Format table with normal + error entries
    table = format_report_table({"sys1": metrics, "oom_sys": {"error": "OOM"}})
    assert "sys1" in table
    assert "OOM" in table


def test_compare_results_aggregates_across_pairs():
    """
    Mean/max keys aggregate across multiple baseline/candidate pairs.
    """
    baselines = [
        InferenceResult(energy=0.0, forces=np.zeros((1, 3))),
        InferenceResult(energy=0.0, forces=np.zeros((1, 3))),
    ]
    candidates = [
        InferenceResult(energy=1.0, forces=np.array([[1.0, 0.0, 0.0]])),
        InferenceResult(energy=3.0, forces=np.array([[5.0, 0.0, 0.0]])),
    ]
    metrics = compare_results(baselines, candidates)
    assert metrics["n_systems"] == 2
    assert metrics["energy_abs_error_mean"] == pytest.approx(2.0)
    assert metrics["energy_abs_error_max"] == pytest.approx(3.0)
    # force_mae averages |Δ|/3 per system: (1/3, 5/3) -> mean = 1.0
    assert metrics["force_mae_mean"] == pytest.approx(1.0)
    assert metrics["force_max_error"] == pytest.approx(5.0)


def test_runner_catches_oom(tmp_path):
    """
    Verify OOM is caught and non-OOM errors are re-raised.
    """
    runner = PerfCheckRunner(
        checkpoint="x", device="cpu", warmup_iters=2, timed_iters=3
    )

    from omegaconf import OmegaConf

    job_config = OmegaConf.create(
        {
            "metadata": {"results_dir": str(tmp_path)},
            "run_dir": str(tmp_path),
        }
    )
    runner.job_config = job_config

    def mock_baseline(checkpoint, systems, inference_settings, **kw):
        # Baselines are single-system, untimed; candidates are batched + timed.
        if kw.get("warmup_iters", 0) == 0 and kw.get("timed_iters", 1) == 1:
            atoms = systems[0].atoms
            return (
                [InferenceResult(energy=-10.0, forces=np.zeros((len(atoms), 3)))],
                PerfMetrics(),
            )
        raise RuntimeError("CUDA out of memory")

    with patch(
        "fairchem.core.components.benchmark.perf_check.run_inference",
        side_effect=mock_baseline,
    ):
        result = runner.run()

    # One row per archetype; all OOM in the candidate phase
    assert set(result["results"].keys()) == {
        "small_molecule",
        "medium_bulk",
        "large_bulk",
    }
    for metrics in result["results"].values():
        assert metrics == {"error": "OOM"}

    # Non-OOM errors should propagate. Reuse the cached baselines so this run
    # only exercises the candidate code path.
    def mock_unrelated(checkpoint, systems, inference_settings, **kw):
        if kw.get("warmup_iters", 0) == 0 and kw.get("timed_iters", 1) == 1:
            atoms = systems[0].atoms
            return (
                [InferenceResult(energy=-10.0, forces=np.zeros((len(atoms), 3)))],
                PerfMetrics(),
            )
        raise RuntimeError("unrelated error")

    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=mock_unrelated,
        ),
        pytest.raises(RuntimeError, match="unrelated error"),
    ):
        runner.run()


@pytest.mark.skip(reason="Requires full UMA model download and large GPU memory")
@pytest.mark.gpu()
def test_run_inference_predictions_and_perf():
    """
    Verify inference returns correct predictions and perf metrics on GPU.
    """
    sys_a = make_benchmark_system(name="tiny_a", natoms=8, task_name="omat", seed=0)
    sys_b = make_benchmark_system(name="tiny_b", natoms=8, task_name="omat", seed=1)

    # Baseline shape: single-system, no perf
    baseline_results, baseline_perf = run_inference(
        "uma-s-1p2", [sys_a], BASELINE_SETTINGS, device="cuda"
    )
    assert len(baseline_results) == 1
    assert baseline_results[0].forces.shape == (8, 3)
    assert baseline_results[0].forces.dtype == np.float64
    assert baseline_perf.qps is None

    # Perf mode: rotates through both systems, returns timed-slice predictions
    timed_systems = [sys_a, sys_b, sys_a]  # 1 warmup + 2 timed
    timed_results, perf = run_inference(
        "uma-s-1p2",
        [sys_a, *timed_systems],  # 2 warmup + 2 timed = 4 total
        InferenceSettings(tf32=False, compile=False),
        device="cuda",
        warmup_iters=2,
        timed_iters=2,
    )
    assert len(timed_results) == 2
    assert perf.qps > 0
    assert perf.peak_gpu_memory_mb > 0


@pytest.mark.skip(reason="Requires full UMA model download and large GPU memory")
@pytest.mark.gpu()
def test_benchmark_runner_end_to_end(tmp_path):
    """
    Full pipeline: per-variant baselines -> evaluate -> aggregate -> JSON.
    """
    from omegaconf import OmegaConf

    runner = PerfCheckRunner(
        checkpoint="uma-s-1p2",
        device="cuda",
        warmup_iters=2,
        timed_iters=3,
        inference_settings=InferenceSettings(tf32=True, compile=False),
    )
    runner.job_config = OmegaConf.create(
        {
            "metadata": {"results_dir": str(tmp_path)},
            "run_dir": str(tmp_path),
        }
    )

    result = runner.run()

    # One row per archetype; aggregate keys present
    for arch in ("small_molecule", "medium_bulk", "large_bulk"):
        assert arch in result["baseline"]
        assert arch in result["results"]
        assert result["results"][arch]["force_mae_mean"] < 0.01
        assert result["results"][arch]["n_systems"] == 3

    # JSON report written
    saved = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert saved == result
