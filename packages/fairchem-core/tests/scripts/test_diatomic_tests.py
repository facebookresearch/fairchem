"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fairchem.core.scripts import diatomic_tests as diatomic


@pytest.mark.parametrize("task", ["omol", "omat"])
def test_diatomic_curve_preserves_metadata_and_subtracts_atomic_reference(task):
    metadata = {"spin": 3, "charge": 1} if task == "omol" else None

    def predict(batch):
        if metadata:
            assert (batch.charge == 1).all()
            assert (batch.spin == 3).all()
        distances = batch.pos.reshape(-1, 2, 3)[:, 1, 2]
        return {"energy": distances.square() + 5.0}

    distances, energies, first, second = diatomic.compute_diatomic_curve(
        SimpleNamespace(predict=predict),
        task,
        "H",
        "H",
        {"atoms_info": metadata, "ref_energy": 5.0},
    )
    assert (first, second) == ("H", "H")
    assert len(distances) == len(energies) == 100
    assert np.all(np.diff(distances) > 0)
    assert distances[-1] == 6.0
    assert energies == pytest.approx(distances**2, abs=1e-5)
    shifted = diatomic.get_relative_energies(energies)
    assert shifted[-1] == 0
    assert shifted == pytest.approx(np.asarray(energies) - energies[-1])


@pytest.mark.parametrize("task", ["omol", "omat"])
def test_diatomic_cli_selects_loader_and_writes_results(tmp_path, monkeypatch, task):
    path = tmp_path / "results"
    if task == "omat":
        path.mkdir()
    arguments = ["diatomic", "-m", "model", "-t", task, "-s", str(path)]
    if task == "omol":
        arguments.append("--released")
    monkeypatch.setattr(sys, "argv", arguments)
    loader = Mock(return_value="predictor")
    monkeypatch.setattr(diatomic.pretrained_mlip, "get_predict_unit", loader)
    monkeypatch.setattr(diatomic.pretrained_mlip, "load_predict_unit", loader)
    entries = {("H", "H"): {"atoms_info": {}, "ref_energy": 0}}
    monkeypatch.setattr(diatomic, "mol_dict", entries)
    monkeypatch.setattr(diatomic, "mat_dict", entries)
    monkeypatch.setattr(
        diatomic,
        "compute_diatomic_curve",
        Mock(return_value=(np.array([1.0, 2.0]), [3.0, 4.0], "H", "H")),
    )
    monkeypatch.setattr(diatomic, "generate_plot", Mock())
    monkeypatch.setattr(diatomic, "combined_plot", Mock())
    diatomic.main()
    loader.assert_called_once_with(
        "model", inference_settings="high-precision", device="cuda"
    )
    with (path / "results.pkl").open("rb") as file:
        results = pickle.load(file)
    assert set(results) == {("H", "H")}
    assert results[("H", "H")]["energies"] == [3.0, 4.0]
    assert np.array_equal(results[("H", "H")]["distances"], [1.0, 2.0])
    assert diatomic.generate_plot.call_args.kwargs["save_path"].endswith(
        "H_H_potential_energy_curve.png"
    )
    diatomic.combined_plot.assert_called_once()
