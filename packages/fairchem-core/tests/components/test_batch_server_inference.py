"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from ase import Atoms

from fairchem.core.components import batch_server
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


def _data(natoms):
    return AtomicData.from_ase(
        Atoms("H" * natoms, positions=[[i, 0, 0] for i in range(natoms)])
    )


def test_oom_splitting_preserves_request_order_and_atom_predictions():
    server = batch_server.BatchPredictServerMixin()
    server.split_oom_batch = True
    items = [_data(n) for n in (2, 3, 4)]
    seen_flags = []

    def predict(batch, undo_element_references):
        seen_flags.append(undo_element_references)
        if len(batch) > 1:
            raise torch.OutOfMemoryError("capacity")
        return {"energy": batch.natoms.float(), "forces": batch.pos.clone()}

    result = server._run_batched_inference(
        items, SimpleNamespace(predict=predict), False
    )
    assert len(result) == len(items)
    for item, prediction in zip(items, result):
        assert prediction["energy"].item() == item.natoms.item()
        assert torch.equal(prediction["forces"], item.pos)
        assert prediction["forces"].device.type == "cpu"
        assert not prediction["forces"].requires_grad
    assert seen_flags
    assert not any(seen_flags)


def test_oom_splitting_reports_disabled_and_single_system_failure():
    server = batch_server.BatchPredictServerMixin()
    unit = SimpleNamespace(predict=Mock(side_effect=torch.OutOfMemoryError("capacity")))
    server.split_oom_batch = False
    with pytest.raises(torch.OutOfMemoryError, match="split_oom_batch=True"):
        server._run_batched_inference([_data(2), _data(3)], unit, True)
    server.split_oom_batch = True
    with pytest.raises(torch.OutOfMemoryError, match="single system"):
        server._run_batched_inference([_data(2)], unit, True)
    batch = atomicdata_list_to_batch([_data(2), _data(3)])
    with pytest.raises(ValueError, match="unexpected shape"):
        server._split_predictions({"energy": torch.zeros(7)}, batch)


@pytest.mark.parametrize("failure", [torch.OutOfMemoryError, RuntimeError])
def test_autobatch_respects_capacity_backoff_and_single_request_latency(
    monkeypatch, failure
):
    monkeypatch.setattr(batch_server.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(batch_server.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        batch_server.time,
        "perf_counter",
        Mock(side_effect=[i * 0.01 for i in range(100)]),
    )
    calls = []

    def predict(batch, undo_element_references):
        calls.append(batch.natoms.sum().item())
        assert not undo_element_references
        if len(calls) == 1:
            raise RuntimeError("cold-start failure")
        if batch.natoms.sum() > 8:
            raise failure("capacity")
        return {}

    config = batch_server.AutobatchConfig(
        min_batch_size=2,
        max_batch_size_cap=16,
        warmup_steps=1,
        probe_steps=2,
        backoff_factor=0.75,
    )
    result = batch_server.probe_optimal_batch_size(
        SimpleNamespace(device="cuda", predict=predict), [_data(2)], config
    )
    assert result.max_batch_size == 6
    assert result.median_latency_s == pytest.approx(0.01)
    assert result.batch_wait_timeout_s == pytest.approx(0.02)
    assert max(calls) > 8


def test_autobatch_defaults_empty_data_and_failed_latency(monkeypatch):
    data = [_data(2)]
    unit = SimpleNamespace(device="cpu")
    defaults = batch_server.probe_optimal_batch_size(unit, data)
    assert defaults.max_batch_size == batch_server.AutobatchConfig().min_batch_size
    with pytest.raises(ValueError, match="probe_data cannot be empty"):
        batch_server.probe_optimal_batch_size(unit, [])
    with pytest.raises(ValueError, match="data_list cannot be empty"):
        batch_server._expand_probe_data([], 2)
    monkeypatch.setattr(batch_server.torch.cuda, "synchronize", lambda: None)
    unit.device = "cuda"
    unit.predict = Mock(side_effect=RuntimeError("unavailable"))
    config = batch_server.AutobatchConfig(
        min_batch_size=2, max_batch_size_cap=4, probe_steps=1, warmup_steps=0
    )
    result = batch_server.probe_optimal_batch_size(unit, data, config)
    assert result.max_batch_size == 2
    assert result.median_latency_s == 0.1
    assert result.batch_wait_timeout_s == config.timeout_ceil_s
