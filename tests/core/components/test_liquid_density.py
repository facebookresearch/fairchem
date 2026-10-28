"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io.jsonio import encode
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from fairchem.core._cli import get_hydra_config_from_yaml
from fairchem.core.components.benchmark._single.liquid_density_reducer import (
    LiquidDensityReducer,
    load_liquid_density_references,
)
from fairchem.core.components.calculate._single import liquid_density_runner
from fairchem.core.components.calculate._single.liquid_density_runner import (
    LiquidDensityRunner,
    _system_seed,
    load_liquid_density_inputs,
)


def _entry(system_id: str, density: float = 1.0) -> dict:
    atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    atoms.info.update(
        {
            "charge": 0,
            "spin": 1,
            "smiles": "O",
            "number_of_molecules": 1,
            "experimental_temperature(K)": 298.0,
            "experimental_density(g/cc)": density,
        }
    )
    return {
        "system_id": system_id,
        "atoms": encode(atoms),
        "smiles": "O",
        "number_of_molecules": 1,
        "number_of_atoms": len(atoms),
        "experimental_temperature_K": 298.0,
        "experimental_density_g_cm3": density,
    }


def _write_inputs(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "systems": entries}))
    return path


def _write_references(path: Path, entries: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "references": {
            entry["system_id"]: {
                "experimental_density_g_cm3": entry["experimental_density_g_cm3"]
            }
            for entry in entries
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_load_inputs_validates_and_sorts(tmp_path: Path) -> None:
    path = _write_inputs(tmp_path / "inputs.json", [_entry("water"), _entry("a")])
    systems = load_liquid_density_inputs(path)
    assert list(systems) == ["a", "water"]
    assert systems["water"]["number_of_atoms"] == 3


def test_load_inputs_rejects_atom_count_mismatch(tmp_path: Path) -> None:
    entry = _entry("water")
    entry["number_of_atoms"] = 4
    path = _write_inputs(tmp_path / "inputs.json", [entry])
    with pytest.raises(ValueError, match="number_of_atoms"):
        load_liquid_density_inputs(path)


def test_load_references_rejects_invalid_density(tmp_path: Path) -> None:
    path = tmp_path / "references.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 1,
                "references": {"water": {"experimental_density_g_cm3": -1.0}},
            },
            handle,
        )
    with pytest.raises(ValueError, match="positive"):
        load_liquid_density_references(path)


def test_system_seed_is_stable_and_distinct() -> None:
    assert _system_seed(42, "water") == _system_seed(42, "water")
    assert _system_seed(42, "water") != _system_seed(42, "benzene")
    assert _system_seed(42, "water") != _system_seed(43, "water")


def test_runner_protocol_and_array_partition(monkeypatch, tmp_path: Path) -> None:
    entries = [_entry("water"), _entry("benzene")]
    input_path = _write_inputs(tmp_path / "inputs.json", entries)
    calls: dict = {}

    class FakeDynamics:
        def __init__(self, atoms, timestep, **kwargs):
            self.atoms = atoms
            self.timestep = timestep
            self.steps = 0
            self.callbacks = []
            calls["kwargs"] = kwargs
            calls["timestep"] = timestep

        def attach(self, callback, interval):
            calls["sample_interval"] = interval
            self.callbacks.append(callback)

        def run(self, steps):
            calls.setdefault("run_steps", []).append(steps)
            self.steps += steps
            for callback in self.callbacks:
                callback()

        def get_time(self):
            return self.steps * self.timestep

    monkeypatch.setattr(liquid_density_runner, "IsotropicMTKNPT", FakeDynamics)
    monkeypatch.setattr(
        liquid_density_runner, "thermalize_momenta", lambda *a, **k: None
    )
    monkeypatch.setattr(liquid_density_runner, "Stationary", lambda *a, **k: None)
    monkeypatch.setattr(liquid_density_runner, "ZeroRotation", lambda *a, **k: None)

    runner = LiquidDensityRunner(
        Calculator(),
        str(input_path),
        timestep_fs=1.0,
        total_time_ps=1.0,
        equilibration_fraction=0.3,
        sample_interval=10,
        seed=7,
    )
    results = runner.calculate(job_num=1, num_jobs=2)

    assert len(results) == 1
    assert results[0]["system_id"] == "water"
    assert results[0]["status"] == "ok"
    assert results[0]["completed_time_ps"] == pytest.approx(1.0)
    assert results[0]["num_samples"] == 2
    assert calls["run_steps"] == [300, 700]
    assert calls["sample_interval"] == 10
    assert calls["timestep"] == units.fs
    assert calls["kwargs"]["pressure_au"] == pytest.approx(1.01325 * units.bar)
    assert calls["kwargs"]["tdamp"] == pytest.approx(100 * units.fs)
    assert calls["kwargs"]["pdamp"] == pytest.approx(1000 * units.fs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestep_fs": 0}, "timestep_fs"),
        ({"total_time_ps": 0}, "total_time_ps"),
        ({"equilibration_fraction": 1}, "equilibration_fraction"),
        ({"timestep_fs": 3, "total_time_ps": 0.001}, "integral"),
    ],
)
def test_runner_rejects_invalid_protocol(tmp_path: Path, kwargs, message) -> None:
    input_path = _write_inputs(tmp_path / "inputs.json", [_entry("water")])
    with pytest.raises(ValueError, match=message):
        LiquidDensityRunner(Calculator(), str(input_path), **kwargs)


def test_runner_writes_failure_record(monkeypatch, tmp_path: Path) -> None:
    input_path = _write_inputs(tmp_path / "inputs.json", [_entry("water")])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(liquid_density_runner, "thermalize_momenta", fail)
    runner = LiquidDensityRunner(Calculator(), str(input_path), total_time_ps=0.001)
    results = runner.calculate()
    assert results[0]["status"] == "failed"
    assert results[0]["error"] == "synthetic failure"
    assert "RuntimeError" in results[0]["traceback"]

    runner.write_results(results, str(tmp_path), job_num=0, num_jobs=1)
    with gzip.open(tmp_path / "liquid_density_1-0.json.gz", "rt") as handle:
        assert json.load(handle)[0]["system_id"] == "water"


def test_runner_rejects_invalid_cell_volume(monkeypatch, tmp_path: Path) -> None:
    entry = _entry("water")
    atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[0, 10, 10],
        pbc=True,
    )
    atoms.info.update(
        {
            "charge": 0,
            "spin": 1,
            "smiles": "O",
            "number_of_molecules": 1,
            "experimental_temperature(K)": 298.0,
            "experimental_density(g/cc)": 1.0,
        }
    )
    entry["atoms"] = encode(atoms)
    input_path = _write_inputs(tmp_path / "inputs.json", [entry])

    class FakeDynamics:
        def __init__(self, atoms, timestep, **kwargs):
            self.timestep = timestep
            self.steps = 0

        def run(self, steps):
            self.steps += steps

        def attach(self, callback, interval):
            pass

        def get_time(self):
            return self.steps * self.timestep

    monkeypatch.setattr(liquid_density_runner, "IsotropicMTKNPT", FakeDynamics)
    monkeypatch.setattr(
        liquid_density_runner, "thermalize_momenta", lambda *a, **k: None
    )
    monkeypatch.setattr(liquid_density_runner, "Stationary", lambda *a, **k: None)
    monkeypatch.setattr(liquid_density_runner, "ZeroRotation", lambda *a, **k: None)

    runner = LiquidDensityRunner(Calculator(), str(input_path), total_time_ps=0.001)
    result = runner.calculate()[0]
    assert result["status"] == "failed"
    assert result["error"] == "water: invalid cell volume"


def test_reducer_merges_references_and_computes_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    entries = [_entry("a", 1.0), _entry("b", 2.0)]
    input_path = _write_inputs(tmp_path / "inputs.json", entries)
    reference_path = _write_references(tmp_path / "references.pkl", entries)
    results = [
        {
            "system_id": "a",
            "status": "ok",
            "predicted_density_g_cm3": 1.1,
        },
        {
            "system_id": "b",
            "status": "ok",
            "predicted_density_g_cm3": 1.8,
        },
    ]
    with gzip.open(tmp_path / "liquid_density_2-0.json.gz", "wt") as handle:
        json.dump(results, handle)

    reducer = LiquidDensityReducer(
        "omol-liquid-density", str(input_path), str(reference_path)
    )
    joined = reducer.join_results(str(tmp_path), reducer.glob_pattern)
    metrics = reducer.compute_metrics(joined, "test").iloc[0]

    assert joined["absolute_error_g_cm3"].tolist() == pytest.approx([0.1, 0.2])
    assert metrics["density_mae_g_cm3"] == pytest.approx(0.15)
    assert metrics["density_rmse_g_cm3"] == pytest.approx(np.sqrt(0.025))
    assert metrics["density_mean_absolute_relative_error_pct"] == pytest.approx(10)
    assert metrics["density_bias_g_cm3"] == pytest.approx(-0.05)
    assert metrics["num_failed"] == 0

    parity_plot_path = Path(reducer.save_parity_plot(joined, str(tmp_path)))
    assert parity_plot_path.name == "omol-liquid-density_parity.png"
    assert parity_plot_path.read_bytes().startswith(b"\x89PNG")

    logger = Mock()
    monkeypatch.setattr(LiquidDensityReducer, "logger", property(lambda self: logger))
    reducer.job_config = OmegaConf.create(
        {"metadata": {"results_dir": str(tmp_path)}, "run_name": "test"}
    )
    reducer.reduce()
    logger.log_plots.assert_called_once_with(
        [str(parity_plot_path)],
        caption="Predicted vs. experimental liquid density",
    )
    logger.log_artifact.assert_called_once_with(
        name="omol-liquid-density-parity",
        type="plot",
        file_location=str(parity_plot_path),
    )


def test_reducer_rejects_json_pickle_density_mismatch(tmp_path: Path) -> None:
    entry = _entry("water", 1.0)
    input_path = _write_inputs(tmp_path / "inputs.json", [entry])
    reference_entry = copy.deepcopy(entry)
    reference_entry["experimental_density_g_cm3"] = 0.9
    reference_path = _write_references(tmp_path / "references.pkl", [reference_entry])
    with pytest.raises(ValueError, match="disagrees"):
        LiquidDensityReducer("density", str(input_path), str(reference_path))


def test_hydra_config_exposes_runtime_controls() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "uma"
        / "benchmark"
        / "omol-liquid-density.yaml"
    )
    try:
        cfg = get_hydra_config_from_yaml(
            str(config_path),
            [
                "cluster=h100_local",
                "job.scheduler.num_array_jobs=1",
                "runner.timestep_fs=0.5",
                "runner.total_time_ps=500",
            ],
        )
        assert cfg.runner.timestep_fs == 0.5
        assert cfg.runner.total_time_ps == 500
        assert cfg.job.scheduler.num_array_jobs == 1
        assert cfg.reducer.reference_data_path.endswith("liquid_density_references.pkl")
    finally:
        GlobalHydra.instance().clear()
