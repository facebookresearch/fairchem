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
    MIXED_BASELINE_CACHE_FILE,
    MIXED_REPORT_FILE,
    BatchTiming,
    InferenceResult,
    MixedInferenceResult,
    MixedPerfCheckRunner,
    _mixed_baseline_cache_key,
    build_batch_schedule,
    format_mixed_report_table,
)
from fairchem.core.components.benchmark.systems import (
    SystemPool,
    get_diverse_benchmark_pool,
    make_benchmark_system,
)

# ---------- SystemPool / diverse pool ----------


def test_systempool_signature_is_sorted_and_stable():
    a = make_benchmark_system("a", structure_type="fcc", natoms=8, task_name="omat")
    b = make_benchmark_system("b", structure_type="fcc", natoms=10, task_name="omat")
    pool1 = SystemPool(systems=[a, b])
    pool2 = SystemPool(systems=[b, a])
    assert pool1.signature() == pool2.signature()
    assert [e["name"] for e in pool1.signature()] == ["a", "b"]


def test_systempool_rejects_duplicate_names():
    a = make_benchmark_system("dup", structure_type="fcc", natoms=8, task_name="omat")
    b = make_benchmark_system("dup", structure_type="fcc", natoms=10, task_name="omat")
    with pytest.raises(ValueError, match="unique"):
        SystemPool(systems=[a, b])


def test_diverse_pool_covers_tasks_and_buckets():
    buckets = (20, 80)
    n = 2
    tasks = ("oc20", "omat", "omol", "odac", "omc")
    pool = get_diverse_benchmark_pool(
        seed=7, size_buckets=buckets, n_per_bucket=n, tasks=tasks
    )
    # Every requested task should appear (the fake_dataset specs always
    # generate structures in the right size range for these buckets).
    seen_tasks = {s.task_name for s in pool.systems}
    assert seen_tasks == set(tasks)
    # Each entry's atom count should fall within the bucket_tolerance window
    # of some requested bucket center (default tolerance is 0.5).
    for s in pool.systems:
        n_atoms = len(s.atoms)
        assert any(
            int(b * 0.5) <= n_atoms <= int(b * 1.5) for b in buckets
        ), f"{s.name} has {n_atoms} atoms, outside any bucket window"


def test_diverse_pool_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        get_diverse_benchmark_pool(tasks=("not-a-task",))


# ---------- build_batch_schedule ----------


def test_schedule_is_deterministic_and_correct_shape():
    s1 = build_batch_schedule(pool_size=10, batch_sizes=[4, 8], num_steps=6, seed=1)
    s2 = build_batch_schedule(pool_size=10, batch_sizes=[4, 8], num_steps=6, seed=1)
    assert s1 == s2
    assert len(s1) == 6
    # round-robin through batch_sizes
    assert [bsz for bsz, _ in s1] == [4, 8, 4, 8, 4, 8]
    # indices are within pool
    for bsz, idxs in s1:
        assert len(idxs) == bsz
        assert all(0 <= i < 10 for i in idxs)


def test_schedule_no_adjacent_repeats_under_normal_conditions():
    # Pool large enough that reroll always succeeds.
    schedule = build_batch_schedule(
        pool_size=64, batch_sizes=[4, 8, 16], num_steps=30, seed=42
    )
    prev_key = None
    for bsz, idxs in schedule:
        key = (bsz, tuple(sorted(idxs)))
        assert key != prev_key, "adjacent batches must not match"
        prev_key = key


def test_schedule_empty_inputs():
    assert build_batch_schedule(pool_size=4, batch_sizes=[2], num_steps=0) == []
    with pytest.raises(ValueError):
        build_batch_schedule(pool_size=0, batch_sizes=[2], num_steps=1)
    with pytest.raises(ValueError):
        build_batch_schedule(pool_size=4, batch_sizes=[], num_steps=1)


# ---------- cache key sensitivity ----------


def test_mixed_cache_key_invalidates_on_pool_change():
    pool_a = SystemPool(
        systems=[
            make_benchmark_system(
                "s1", structure_type="fcc", natoms=8, task_name="omat"
            )
        ]
    )
    pool_b = SystemPool(
        systems=[
            make_benchmark_system(
                "s2", structure_type="fcc", natoms=8, task_name="omat"
            )
        ]
    )
    k_a = _mixed_baseline_cache_key("ckpt", pool_a, "cpu", 42, (4,))
    k_b = _mixed_baseline_cache_key("ckpt", pool_b, "cpu", 42, (4,))
    assert k_a != k_b


def test_mixed_cache_key_invalidates_on_batch_sizes():
    pool = SystemPool(
        systems=[
            make_benchmark_system(
                "s1", structure_type="fcc", natoms=8, task_name="omat"
            )
        ]
    )
    k1 = _mixed_baseline_cache_key("ckpt", pool, "cpu", 42, (4, 8))
    k2 = _mixed_baseline_cache_key("ckpt", pool, "cpu", 42, (4, 8, 16))
    assert k1 != k2


# ---------- format_mixed_report_table ----------


def test_format_mixed_report_table_renders():
    per_batch = {
        4: BatchTiming(4, 2, 0.5, 16.0, 64.0),
        8: BatchTiming(8, 1, 0.25, 32.0, 128.0),
    }
    per_sys = {
        "s1": {
            "energy_abs_error": 0.001,
            "force_mae": 0.0001,
            "force_max_error": 0.001,
        },
        "s2": {"error": "missing"},
    }
    text = format_mixed_report_table(per_batch, per_sys)
    assert "Throughput by batch size" in text
    assert "Per-system error vs fp64 baseline" in text
    assert "s1" in text
    assert "missing" in text


# ---------- MixedPerfCheckRunner: end-to-end with mocks ----------


def _tiny_runner(tmp_path) -> MixedPerfCheckRunner:
    """
    Minimal runner: 1 task, 2 buckets, 1 variant per bucket,
    2 batch sizes, 2 warmup + 4 timed steps.
    """
    runner = MixedPerfCheckRunner(
        checkpoint="dummy-ckpt",
        device="cpu",
        batch_sizes=(2, 4),
        warmup_steps=2,
        timed_steps=4,
        seed=42,
        pool_size_buckets=(8, 32),
        pool_n_per_bucket=1,
        pool_tasks=("omat",),
        oom_policy="skip",
    )
    from omegaconf import OmegaConf

    runner.job_config = OmegaConf.create(
        {"metadata": {"results_dir": str(tmp_path)}, "run_dir": str(tmp_path)}
    )
    return runner


def _mock_run_inference(checkpoint, system, inference_settings, **_kw):
    return InferenceResult(
        energy=-1.0 * len(system.atoms),
        forces=np.zeros((len(system.atoms), 3), dtype=np.float64),
    )


def _mock_run_mixed_inference(predict_unit, pool, schedule, warmup_steps, **_kw):
    per_sys = {
        s.name: InferenceResult(
            energy=-1.0 * len(s.atoms),
            forces=np.zeros((len(s.atoms), 3), dtype=np.float64),
        )
        for s in pool.systems
    }
    per_batch = {}
    for bsz, _ in schedule[warmup_steps:]:
        per_batch.setdefault(bsz, BatchTiming(bsz, 0, 0.0, 0.0, 0.0))
        t = per_batch[bsz]
        per_batch[bsz] = BatchTiming(
            bsz,
            t.n_steps + 1,
            t.total_seconds + 0.01,
            (t.n_steps + 1) * bsz / max(t.total_seconds + 0.01, 1e-9),
            (t.n_steps + 1) * bsz * 8 / max(t.total_seconds + 0.01, 1e-9),
        )
    return MixedInferenceResult(
        per_system=per_sys,
        per_batch_size=per_batch,
        warmup_seconds=0.001,
        total_timed_seconds=0.01,
    )


def test_mixed_runner_end_to_end_writes_report(tmp_path):
    runner = _tiny_runner(tmp_path)
    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=_mock_run_inference,
        ),
        patch(
            "fairchem.core.components.benchmark.perf_check.run_mixed_inference",
            side_effect=_mock_run_mixed_inference,
        ),
        patch.object(MixedPerfCheckRunner, "_build_predict_unit", return_value=None),
    ):
        report = runner.run()

    # Report written to disk and matches return value (allowing for JSON
    # turning int dict keys into strings)
    saved = json.loads((tmp_path / MIXED_REPORT_FILE).read_text())
    assert {int(k): v for k, v in saved["per_batch_size"].items()} == report[
        "per_batch_size"
    ]
    assert saved["per_system"] == report["per_system"]
    # Per-batch-size populated for both sizes
    assert set(report["per_batch_size"]) == {2, 4}
    # Every pool entry got an error row
    assert set(report["per_system"]) == {s.name for s in runner.pool.systems}
    # Baseline cache file was created
    assert (tmp_path / MIXED_BASELINE_CACHE_FILE).exists()


def test_mixed_runner_reuses_cached_baseline(tmp_path):
    runner = _tiny_runner(tmp_path)
    call_count = {"n": 0}

    def counting_run(*args, **kwargs):
        call_count["n"] += 1
        return _mock_run_inference(*args, **kwargs)

    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=counting_run,
        ),
        patch(
            "fairchem.core.components.benchmark.perf_check.run_mixed_inference",
            side_effect=_mock_run_mixed_inference,
        ),
        patch.object(MixedPerfCheckRunner, "_build_predict_unit", return_value=None),
    ):
        runner.run()
        baseline_calls_first = call_count["n"]
        runner.run()  # second run should hit the cache
        baseline_calls_second = call_count["n"] - baseline_calls_first

    assert baseline_calls_first == len(runner.pool)
    assert baseline_calls_second == 0


def test_mixed_runner_invalidates_cache_when_pool_changes(tmp_path):
    runner = _tiny_runner(tmp_path)
    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=_mock_run_inference,
        ),
        patch(
            "fairchem.core.components.benchmark.perf_check.run_mixed_inference",
            side_effect=_mock_run_mixed_inference,
        ),
        patch.object(MixedPerfCheckRunner, "_build_predict_unit", return_value=None),
    ):
        runner.run()

    cache_path = os.path.join(str(tmp_path), MIXED_BASELINE_CACHE_FILE)
    with open(cache_path) as f:
        data = json.load(f)
    data["cache_key"] = "stale-key"
    with open(cache_path, "w") as f:
        json.dump(data, f)

    call_count = {"n": 0}

    def counting(*a, **kw):
        call_count["n"] += 1
        return _mock_run_inference(*a, **kw)

    with (
        patch(
            "fairchem.core.components.benchmark.perf_check.run_inference",
            side_effect=counting,
        ),
        patch(
            "fairchem.core.components.benchmark.perf_check.run_mixed_inference",
            side_effect=_mock_run_mixed_inference,
        ),
        patch.object(MixedPerfCheckRunner, "_build_predict_unit", return_value=None),
    ):
        runner.run()

    assert call_count["n"] == len(runner.pool)  # full recompute
