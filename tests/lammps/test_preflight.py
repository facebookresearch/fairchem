"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from fairchem.lammps.preflight import (
    PROFILE_BY_NAME,
    BenchmarkCase,
    RecordingPredictor,
    _namespace_from_config,
    add_break_even_steps,
    add_numerical_comparison,
    build_case_matrix,
    generate_fcc_system,
    hydra_overrides_for_case,
    numerical_tolerances,
    projected_runtime_seconds,
    recommend_case,
)


def _result(
    *,
    profile="fp32_eager",
    graph_version=2,
    workers=1,
    initialization=1.0,
    cold_start=2.0,
    warmup=4.0,
    step_ms=10.0,
    cv=0.01,
):
    return {
        "profile": profile,
        "graph_version": graph_version,
        "workers": workers,
        "status": "passed",
        "numerical_passed": True,
        "resolved_execution_mode": "umas_fast_gpu",
        "initialization_seconds": initialization,
        "cold_start_seconds": cold_start,
        "warmup_steps": 10,
        "warmup_seconds": warmup,
        "median_ms_per_step": step_ms,
        "timing_cv": cv,
    }


def test_profiles_retain_automatic_execution_mode():
    for profile in PROFILE_BY_NAME.values():
        settings = profile.settings(graph_version=2)
        assert settings.execution_mode is None
        assert settings.merge_mole is True


def test_hydra_config_composes_benchmark_overrides():
    config_dir = Path(__file__).resolve().parents[2] / "src" / "fairchem" / "lammps"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="preflight_config",
            overrides=[
                "mode=benchmark",
                "generated_atoms=64",
                "worker_counts=[1,2,4]",
                "output=results.json",
            ],
        )
    args = _namespace_from_config(cfg)
    assert args.mode == "benchmark"
    assert args.generated_atoms == 64
    assert args.worker_counts == [1, 2, 4]
    assert args.output == Path("results.json")


def test_resolved_execution_mode_uses_loaded_parallel_rank():
    outer_settings = PROFILE_BY_NAME["memory_saving"].settings(2)
    inner_settings = PROFILE_BY_NAME["turbo"].settings(2)
    inner_settings.execution_mode = "umas_fast_gpu"
    predictor = SimpleNamespace(
        inference_settings=outer_settings,
        local_rank0=SimpleNamespace(
            predict_unit=SimpleNamespace(inference_settings=inner_settings)
        ),
    )
    assert RecordingPredictor(predictor).resolved_execution_mode == "umas_fast_gpu"


def test_recording_predictor_retains_callback_failure():
    class BrokenPredictor:
        def predict(self, data):
            raise ValueError("original callback failure")

    recording = RecordingPredictor(BrokenPredictor())
    data = SimpleNamespace(clone=lambda: "copy")
    with pytest.raises(ValueError, match="original callback failure"):
        recording.predict(data)
    assert isinstance(recording.last_error, ValueError)


def test_generated_fcc_system_is_deterministic_and_orthogonal():
    first = generate_fcc_system(17)
    second = generate_fcc_system(17)
    assert len(first) == 17
    assert np.array_equal(first.positions, second.positions)
    assert np.allclose(first.cell.array, np.diag(np.diag(first.cell.array)))
    assert first.pbc.all()


def test_build_case_matrix_limits_graph_v3_to_one_gpu():
    cases = build_case_matrix(gpu_count=2)
    assert len(cases) == 15
    assert BenchmarkCase("turbo", graph_version=3, workers=1) in cases
    assert BenchmarkCase("turbo", graph_version=3, workers=2) not in cases
    assert BenchmarkCase("turbo", graph_version=2, workers=2) in cases


@pytest.mark.parametrize("counts", [[0], [3], []])
def test_build_case_matrix_rejects_invalid_worker_counts(counts):
    with pytest.raises(ValueError, match="Worker counts"):
        build_case_matrix(gpu_count=2, worker_counts=counts)


def test_projected_runtime_includes_observed_warmup():
    result = _result()
    assert projected_runtime_seconds(result, expected_steps=5) == pytest.approx(5.0)
    assert projected_runtime_seconds(result, expected_steps=110) == pytest.approx(8.0)


def test_break_even_steps_include_extra_startup_cost():
    baseline = _result(step_ms=10.0)
    compiled = _result(
        profile="fp32_compiled",
        cold_start=12.0,
        step_ms=5.0,
    )
    add_break_even_steps([baseline, compiled])
    assert baseline["break_even_steps_vs_fp32_eager"] is None
    assert compiled["break_even_steps_vs_fp32_eager"] == 2010


def test_tf32_uses_relaxed_established_tolerances():
    fp32 = numerical_tolerances(PROFILE_BY_NAME["fp32_eager"])
    tf32 = numerical_tolerances(PROFILE_BY_NAME["turbo"])
    assert tf32["force_rtol"] > fp32["force_rtol"]
    assert tf32["energy_rtol"] > fp32["energy_rtol"]


def test_numerical_comparison_records_errors_and_removes_private_arrays():
    result = {
        "profile": "fp32_eager",
        "_energy": np.array([2.0]),
        "_forces": np.ones((2, 3)),
    }
    reference = {"energy": np.array([2.0]), "forces": np.ones((2, 3))}
    add_numerical_comparison(result, reference)
    assert result["numerical_passed"] is True
    assert result["force_max_error_eV_per_A"] == 0
    assert "_energy" not in result
    assert "_forces" not in result


def test_recommendation_requires_material_speedup():
    baseline = _result(step_ms=10.0)
    marginal = _result(profile="turbo", step_ms=9.5)
    recommendation = recommend_case([baseline, marginal], expected_steps=1000)
    assert recommendation["profile"] == "fp32_eager"


def test_recommendation_selects_faster_parallel_case_and_emits_overrides():
    baseline = _result(step_ms=10.0)
    parallel = _result(
        profile="turbo",
        workers=2,
        initialization=1.0,
        cold_start=2.0,
        warmup=2.0,
        step_ms=4.0,
    )
    recommendation = recommend_case([baseline, parallel], expected_steps=10000)
    assert recommendation["profile"] == "turbo"
    assert recommendation["workers"] == 2
    assert "predict_unit=${parallel_predict_unit}" in recommendation["hydra_overrides"]


def test_recommendation_rejects_marginal_parallel_gain_over_best_single_gpu():
    baseline = _result(step_ms=10.0)
    single_turbo = _result(profile="turbo", step_ms=4.0)
    parallel_turbo = _result(profile="turbo", workers=2, step_ms=3.8)
    recommendation = recommend_case(
        [baseline, single_turbo, parallel_turbo], expected_steps=10000
    )
    assert recommendation["profile"] == "turbo"
    assert recommendation["workers"] == 1


def test_recommendation_rejects_marginal_compile_gain_over_best_eager():
    baseline = _result(step_ms=10.0)
    eager = _result(profile="tf32_eager", step_ms=5.0)
    compiled = _result(profile="turbo", step_ms=4.8)
    recommendation = recommend_case([baseline, eager, compiled], expected_steps=10000)
    assert recommendation["profile"] == "tf32_eager"


def test_no_recommendation_for_noisy_or_failed_cases():
    noisy = _result(cv=0.2)
    assert recommend_case([noisy], expected_steps=1000) is None


def test_memory_fallback_can_be_recommended_when_baseline_does_not_fit():
    memory = _result(profile="memory_saving", step_ms=20.0)
    recommendation = recommend_case([memory], expected_steps=1000)
    assert recommendation["profile"] == "memory_saving"


def test_local_overrides_keep_automatic_backend():
    result = _result(profile="tf32_eager", graph_version=3)
    result["model"] = "uma-s-1p2"
    overrides = hydra_overrides_for_case(result)
    assert "predict_unit=${local_predict_unit}" in overrides
    assert "if_settings.internal_graph_gen_version=3" in overrides
    assert "local_predict_unit.path.model_name=uma-s-1p2" in overrides
    assert not any("execution_mode" in override for override in overrides)
