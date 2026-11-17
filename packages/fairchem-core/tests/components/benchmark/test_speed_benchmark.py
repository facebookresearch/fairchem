"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
import torch
from ase import Atoms
from ase.io import write
from omegaconf import OmegaConf

from fairchem.core.components.benchmark._single import uma_speed_benchmark as speed
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def test_qps_converts_iterations_to_simulation_nanoseconds(monkeypatch):
    predictor = Mock()
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    def repeat(function, number, repeat):
        for _ in range(number * repeat):
            function()
        return [2.0, 4.0]

    monkeypatch.setattr(speed.timeit, "repeat", repeat)
    qps, ns_per_day = speed.get_qps(None, predictor, warmups=2, timeiters=3, repeats=2)
    assert qps == 1
    assert ns_per_day == pytest.approx(0.0864)
    assert predictor.predict.call_count == barrier.call_count == 8


@pytest.mark.parametrize("file_input", [False, True])
def test_runner_saves_throughput_and_stops_at_oom(tmp_path, monkeypatch, file_input):
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    path = tmp_path / "system.xyz"
    write(path, atoms)
    predictor = Mock()
    monkeypatch.setattr(speed, "MLIPPredictUnit", Mock(return_value=predictor))
    monkeypatch.setattr(speed, "seed_everywhere", Mock())
    monkeypatch.setattr(
        speed, "get_fcc_crystal_by_num_atoms", lambda n: atoms.repeat((n, 1, 1))
    )
    monkeypatch.setattr(speed.distutils, "is_master", lambda: True)
    monkeypatch.setattr(speed.distutils, "get_world_size", lambda: 1)
    timing = Mock(side_effect=[(2.0, 0.1728), torch.cuda.OutOfMemoryError("capacity")])
    monkeypatch.setattr(speed, "get_qps", timing)
    inputs = (
        {"input_system": {"a": str(path), "b": str(path)}, "expand_supercells": 2}
        if file_input
        else {"natoms_list": [1, 2, 3]}
    )
    runner = speed.InferenceBenchRunner(
        {"tiny": "unused.pt"},
        device="cpu",
        inference_settings=InferenceSettings(),
        **inputs,
    )
    runner.job_config = OmegaConf.create({"metadata": {"results_dir": str(tmp_path)}})
    runner.run()
    result = json.loads((tmp_path / "benchmark_results.json").read_text())
    first, second = (8, 8) if file_input else (1, 2)
    assert result["model_to_qps_data"] == {"tiny": [[first, 0.1728], [second, "OOM"]]}
    assert result["config"]["device"] == "cpu"
    assert result["config"]["world_size"] == 1
    assert timing.call_count == 2
    assert timing.call_args.kwargs == {"timeiters": 10, "repeats": 5}


def test_runner_rethrows_errors_unrelated_to_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(speed, "MLIPPredictUnit", Mock())
    monkeypatch.setattr(speed, "seed_everywhere", Mock())
    monkeypatch.setattr(speed, "get_qps", Mock(side_effect=RuntimeError("bad model")))
    runner = speed.InferenceBenchRunner(
        {"tiny": "unused.pt"}, natoms_list=[1], device="cpu"
    )
    runner.job_config = OmegaConf.create({"metadata": {"results_dir": str(tmp_path)}})
    with pytest.raises(RuntimeError, match="bad model"):
        runner.run()
    assert not (tmp_path / "benchmark_results.json").exists()
    with pytest.raises(AssertionError, match="input must be"):
        speed.InferenceBenchRunner({}, natoms_list=[1], input_system={})
