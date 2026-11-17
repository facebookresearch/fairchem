"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.components.benchmark import perf_check
from fairchem.core.components.benchmark.systems import BenchmarkSystem, SystemPool
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def _predict(data):
    return {
        "energy": data.natoms.float(),
        "forces": data.pos.clone(),
        "stress": torch.eye(3).repeat(data.natoms.numel(), 1, 1),
    }


def _pool():
    return SystemPool(
        systems=[
            BenchmarkSystem("a", Atoms("H2"), "omol"),
            BenchmarkSystem("b", Atoms("H3"), "omol"),
        ]
    )


def test_single_inference_counts_timed_iterations_and_converts_outputs(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.pt"
    path.touch()
    unit = SimpleNamespace(predict=Mock(side_effect=_predict))
    monkeypatch.setattr(perf_check, "MLIPPredictUnit", Mock(return_value=unit))
    ticks = itertools.count()
    monkeypatch.setattr(perf_check.time, "perf_counter", lambda: next(ticks) * 0.01)
    system = _pool().systems[0]
    baseline = perf_check.run_inference(
        str(path), system, InferenceSettings(), device="cpu"
    )
    assert baseline.energy == 2
    assert baseline.forces.dtype == np.float64
    assert baseline.stress.shape == (1, 3, 3)
    assert baseline.qps is None
    assert unit.predict.call_count == 1
    measured = perf_check.run_inference(
        str(path),
        system,
        InferenceSettings(),
        device="cpu",
        warmup_iters=2,
        timed_iters=3,
    )
    assert measured.qps == pytest.approx(300)
    assert measured.warmup_time_seconds == pytest.approx(0.01)
    assert measured.wall_time_seconds == pytest.approx(0.01)
    assert measured.peak_gpu_memory_mb is None
    assert unit.predict.call_count == 6


def test_mixed_inference_counts_samples_and_atoms_per_batch_size(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(perf_check.time, "perf_counter", lambda: next(ticks) * 0.01)
    schedule = [(1, (0,)), (2, (0, 1)), (1, (1,)), (2, (1, 0)), (1, (0,))]
    result = perf_check.run_mixed_inference(
        SimpleNamespace(predict=_predict),
        _pool(),
        schedule,
        warmup_steps=1,
        device="cpu",
    )
    assert result.per_batch_size[2].n_steps == 2
    assert result.per_batch_size[2].samples_per_sec == pytest.approx(200)
    assert result.per_batch_size[2].atoms_per_sec == pytest.approx(500)
    assert result.per_batch_size[1].atoms_per_sec == pytest.approx(250)
    assert result.per_system["a"].energy == 2
    assert result.per_system["b"].energy == 3
    assert result.per_system["b"].forces.shape == (3, 3)
    assert result.per_system["a"].stress.dtype == np.float64
    assert result.oom_batch_sizes == []


def test_mixed_inference_skips_oom_sizes_and_propagates_other_errors():
    calls = []

    def predict(data):
        calls.append(data.natoms.numel())
        if data.natoms.numel() > 1:
            raise torch.OutOfMemoryError("capacity")
        return _predict(data)

    unit = SimpleNamespace(predict=predict)
    schedule = [(2, (0, 1)), (2, (0, 1)), (1, (0,)), (2, (1, 0)), (1, (1,))]
    result = perf_check.run_mixed_inference(
        unit, _pool(), schedule, warmup_steps=1, device="cpu", oom_policy="skip"
    )
    assert result.oom_batch_sizes == [2]
    assert set(result.per_batch_size) == {1}
    assert calls.count(2) == 2
    with pytest.raises(torch.OutOfMemoryError):
        perf_check.run_mixed_inference(
            unit, _pool(), schedule, warmup_steps=1, device="cpu"
        )
    unit.predict = Mock(side_effect=RuntimeError("invalid input"))
    with pytest.raises(RuntimeError, match="invalid input"):
        perf_check.run_mixed_inference(
            unit, _pool(), schedule, warmup_steps=1, device="cpu", oom_policy="skip"
        )
    with pytest.raises(ValueError, match="oom_policy"):
        perf_check.run_mixed_inference(
            unit, _pool(), [], 0, device="cpu", oom_policy="invalid"
        )
