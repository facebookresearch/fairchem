"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GPU tests for UMA Nitro running through MLIPPredictUnit and FAIRChemCalculator
on the pinned uma-s-1p2p1 checkpoint. Fallbacks are in
test_uma_nitro_fallbacks.py.
"""

from __future__ import annotations

import logging

import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.predict import _REDUCED_PRECISION_WARNING
from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    assert_close,
    batch,
    checkpoint_path,
    general_predictor,
    nitro_settings,
    requires_nitro_gpu,
    task_names,
)

# These check Nitro's own results, so they need a GPU Nitro runs on; on other
# GPUs (CI's T4s) Nitro falls back, which test_uma_nitro_fallbacks.py covers.
pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.pretrained("uma-s-1p2p1"),
    requires_nitro_gpu,
]


@pytest.fixture(scope="module")
def general(pretrained_model_name):
    return general_predictor(checkpoint_path(pretrained_model_name))


def _assert_runs_nitro(predictor) -> None:
    assert predictor.inference_settings.nitro is True
    assert type(predictor.model.module).__name__ == "UMANitroModel"
    # The preset's compile=True is for the multigpu fallback; Nitro is never
    # wrapped by torch.compile.
    assert type(predictor.model).__name__ == "_ModelWrapper"


@pytest.mark.parametrize(
    "system",
    [
        (bulk("Cu", "fcc", a=3.6) * (2, 2, 2), "omat"),
        (bulk("NaCl", "rocksalt", a=5.6) * (2, 2, 2), "omc"),
        (molecule("H2O"), "omol", 0, 1),
        (molecule("H2O"), "omol", 1, 2),
        (Atoms("H", positions=[[0.0, 0.0, 0.0]]), "omol", 0, 2),
        (Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]), "omol", 0, 3),
    ],
    ids=["Cu-omat", "NaCl-omc", "H2O-omol", "H2O+-omol", "H-atom", "no-edges"],
)
def test_matches_general(pretrained_model_name, general, system):
    """
    Nitro runs without falling back and agrees with the general backend,
    including isolated atoms and systems without edges.
    """
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    assert task_names(predictor) == task_names(general)
    task = system[1]
    if task not in general.dataset_to_tasks:
        pytest.skip(f"checkpoint does not support {task}")
    atoms = system[0]
    for step in range(3):
        moved = atoms.copy()
        moved.positions += 0.01 * step
        moved_system = (moved, *system[1:])
        output = predictor.predict(batch(moved_system))
        _assert_runs_nitro(predictor)
        assert_close(output, general.predict(batch(moved_system)))


def test_matches_general_without_element_references(pretrained_model_name, general):
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    data = batch((bulk("Cu", "fcc", a=3.6) * (2, 2, 2), "omat"))
    output = predictor.predict(data.clone(), undo_element_references=False)
    _assert_runs_nitro(predictor)
    expected = general.predict(data.clone(), undo_element_references=False)
    assert_close(output, expected)


@pytest.mark.parametrize(
    "change",
    [
        {"compile": False},
        {"hessian_vmap": True},
        {"edge_chunk_size": 1000},
        {"use_quaternion_wigner": False},
        {"torch_num_threads": 2},
        {"auto_add_default_untrained_tasks": False},
        {
            "auto_add_default_untrained_tasks": False,
            "predict_untrained_stress": {"omol"},
        },
        {"predict_untrained_forces": {"omol"}},
    ],
    ids=lambda change: ",".join(change),
)
def test_runs_with_settings_it_ignores_or_supports(pretrained_model_name, change):
    """
    HydraModel-only tuning and untrained tasks leave Nitro running, with the
    same tasks the HydraModel would expose for those settings.
    """
    path = checkpoint_path(pretrained_model_name)
    predictor = MLIPPredictUnit(
        path, "cuda", inference_settings=nitro_settings(**change)
    )
    task_changes = {
        key: value
        for key, value in change.items()
        if key.startswith(("auto_add", "predict_untrained"))
    }
    assert task_names(predictor) == task_names(general_predictor(path, **task_changes))
    output = predictor.predict(batch((molecule("H2O"), "omol", 0, 1)))
    _assert_runs_nitro(predictor)
    for name, value in output.items():
        assert torch.isfinite(value).all(), name
    assert ("stress" in output) == ("omol_stress" in task_names(predictor)["omol"])


def test_default_settings_run_nitro(pretrained_model_name):
    """
    The default inference settings run Nitro for uma-s-1p2p1, through both the
    predict unit and the ASE calculator.
    """
    from fairchem.core.units.mlip_unit import load_predict_unit

    path = checkpoint_path(pretrained_model_name)
    predictor = load_predict_unit(path, device="cuda")
    predictor.predict(batch((molecule("H2O"), "omol", 0, 1)))
    _assert_runs_nitro(predictor)

    calc = FAIRChemCalculator.from_model_checkpoint(path, "omat", device="cuda")
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    atoms.calc = calc
    atoms.get_potential_energy()
    _assert_runs_nitro(calc.predictor)


def test_calculator(pretrained_model_name, caplog):
    """
    The ASE calculator drop-in returns finite energy, forces and stress.
    """
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            checkpoint_path(pretrained_model_name),
            "cuda",
            inference_settings=nitro_settings(),
        )
    # One reduced-precision warning, the same one TF32 modes print.
    assert caplog.text.count(_REDUCED_PRECISION_WARNING) == 1
    calc = FAIRChemCalculator(predictor, "omat")
    atoms = bulk("C", "diamond", a=3.57) * (2, 2, 2)
    atoms.calc = calc
    assert torch.isfinite(torch.tensor(atoms.get_potential_energy()))
    assert torch.isfinite(torch.tensor(atoms.get_forces())).all()
    assert torch.isfinite(torch.tensor(atoms.get_stress())).all()
    _assert_runs_nitro(calc.predictor)


def test_recovers_from_stale_edge_capacity(pretrained_model_name, monkeypatch):
    """
    Discard an undersized graph and replay with a remeasured capacity.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    data = batch((bulk("Cu", "fcc", a=3.6) * (2, 2, 2), "omat"))

    monkeypatch.setattr(inference, "_LIVE_EDGES", {})
    inference._flush_graphs()
    try:
        predictor.predict(data.clone())
        cache_key = inference._capacity_cache_key(
            data.pos.shape[0], data.dataset[0], inference._geometry_key(data)
        )
        _, cell_capacity, rep_max = inference._LIVE_EDGES[cache_key]
        inference._LIVE_EDGES[cache_key] = (1, cell_capacity, rep_max)
        # The high-water mark would otherwise keep the capacity that fit.
        predictor.model.module.backbone._uma_nitro_capacity_high_water.clear()

        output = predictor.predict(data.clone())

        assert inference._LIVE_EDGES[cache_key][0] > 64
        _assert_runs_nitro(predictor)
        for field in ("energy", "forces", "stress"):
            assert torch.isfinite(output[field]).all(), field
    finally:
        inference._flush_graphs()
