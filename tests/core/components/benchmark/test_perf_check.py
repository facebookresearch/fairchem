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
    compare_results,
    format_report_table,
    run_inference,
)
from fairchem.core.components.benchmark.systems import make_benchmark_system
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def test_compare_results_and_format_table():
    """
    Verify error computation and table formatting together.
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
        qps=100.0,
        wall_time_seconds=0.2,
        peak_gpu_memory_mb=1024.0,
        warmup_time_seconds=1.0,
    )
    metrics = compare_results(baseline, candidate)
    assert metrics["energy_abs_error"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["force_max_error"] == pytest.approx(0.02, abs=1e-6)
    assert metrics["stress_mae"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["qps"] == 100.0

    # Format table with normal + error entries
    table = format_report_table({"sys1": metrics, "oom_sys": {"error": "OOM"}})
    assert "sys1" in table
    assert "OOM" in table


def test_runner_catches_oom(tmp_path):
    """
    Verify OOM is caught and non-OOM errors are re-raised.
    """
    runner = PerfCheckRunner(checkpoint="x", device="cpu")
    num_systems = len(runner.systems)

    from omegaconf import OmegaConf

    job_config = OmegaConf.create(
        {
            "metadata": {"results_dir": str(tmp_path)},
            "run_dir": str(tmp_path),
        }
    )
    runner.job_config = job_config

    call_count = {"n": 0}

    def mock_run(checkpoint, system, inference_settings, **kw):
        call_count["n"] += 1
        if call_count["n"] <= num_systems:  # baseline calls
            return InferenceResult(
                energy=-10.0, forces=np.zeros((len(system.atoms), 3))
            )
        raise RuntimeError("CUDA out of memory")

    with patch(
        "fairchem.core.components.benchmark.perf_check.run_inference",
        side_effect=mock_run,
    ):
        result = runner.run()

    for metrics in result["results"].values():
        assert metrics == {"error": "OOM"}

    # Non-OOM errors should propagate
    call_count["n"] = 0
    # Remove cached baselines so the next run recomputes them
    cache_file = os.path.join(str(tmp_path), "baseline_cache.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    def mock_run_bad(checkpoint, system, inference_settings, **kw):
        call_count["n"] += 1
        if call_count["n"] <= num_systems:
            return InferenceResult(
                energy=-10.0, forces=np.zeros((len(system.atoms), 3))
            )
        raise RuntimeError("unrelated error")

    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=mock_run_bad,
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
    system = make_benchmark_system(name="tiny", natoms=8, task_name="omat")

    # Baseline mode: predictions only, no perf
    baseline = run_inference("uma-s-1p2", system, BASELINE_SETTINGS, device="cuda")
    assert baseline.forces.shape == (8, 3)
    assert baseline.forces.dtype == np.float64
    assert baseline.qps is None

    # Perf mode: predictions + metrics
    result = run_inference(
        "uma-s-1p2",
        system,
        InferenceSettings(tf32=False, compile=False),
        device="cuda",
        warmup_iters=2,
        timed_iters=3,
    )
    assert result.qps > 0
    assert result.peak_gpu_memory_mb > 0
    assert abs(baseline.energy - result.energy) < 0.01


@pytest.mark.skip(reason="Requires full UMA model download and large GPU memory")
@pytest.mark.gpu()
def test_benchmark_runner_end_to_end(tmp_path):
    """
    Full pipeline: baseline -> evaluate -> compare -> JSON report.
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

    # All 4 default systems benchmarked
    for name in ("small_molecule", "medium_bulk", "large_bulk", "slab_adsorbate"):
        assert name in result["baseline"]
        assert name in result["results"]
        assert result["results"][name]["force_mae"] < 0.01

    # JSON report written
    saved = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert saved == result
