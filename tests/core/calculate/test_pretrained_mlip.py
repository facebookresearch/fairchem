"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy.testing as npt
import pytest
from ase.build import bulk

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from tests.conftest import get_predict_unit_for_test

pytestmark = [pytest.mark.uses_uma, pytest.mark.uma_models("uma-s-1p1")]


def test_get_predict_unit_by_name(uma_checkpoint):
    """Registered name → predictor with populated external refs."""
    predictor = pretrained_mlip.get_predict_unit(uma_checkpoint, device="cpu")
    assert isinstance(predictor, MLIPPredictUnit)
    assert predictor.atom_refs, "atom_refs should be populated by the name branch"
    assert (
        predictor.form_elem_refs
    ), "form_elem_refs should be populated by the name branch"


def test_get_predict_unit_for_test_by_path(uma_s_1p1_checkpoint):
    """
    Filesystem path via get_predict_unit_for_test → predictor loads
    successfully but external refs are empty (path goes through
    load_predict_unit which does not auto-fetch YAMLs).
    """
    predictor = get_predict_unit_for_test(uma_s_1p1_checkpoint, device="cpu")
    assert isinstance(predictor, MLIPPredictUnit)
    assert predictor.atom_refs == {}, "path-load should leave atom_refs empty"
    assert predictor.form_elem_refs == {}, "path-load should leave form_elem_refs empty"


def test_name_and_path_forward_match(uma_checkpoint, uma_s_1p1_checkpoint):
    """
    Forward pass on a multi-atom system must agree between name-load and
    path-load — the in-checkpoint normalizer and element_references buffers
    travel with the .pt, so external refs don't affect ordinary energies.
    """
    pu_by_name = pretrained_mlip.get_predict_unit(uma_checkpoint, device="cpu")
    pu_by_path = get_predict_unit_for_test(uma_s_1p1_checkpoint, device="cpu")

    atoms = bulk("Cu", "fcc", a=3.6).repeat((2, 1, 1))  # 2-atom system
    calc_name = FAIRChemCalculator(pu_by_name, task_name="omat")
    calc_path = FAIRChemCalculator(pu_by_path, task_name="omat")

    a_name = atoms.copy()
    a_name.calc = calc_name
    a_path = atoms.copy()
    a_path.calc = calc_path

    npt.assert_allclose(
        a_name.get_potential_energy(), a_path.get_potential_energy(), atol=1e-6
    )
    npt.assert_allclose(a_name.get_forces(), a_path.get_forces(), atol=1e-6)


def test_get_predict_unit_unknown_name_raises_keyerror():
    """Bare string that's not a file and not a registered name → KeyError."""
    with pytest.raises(KeyError, match="not found"):
        pretrained_mlip.get_predict_unit("definitely-not-a-real-model", device="cpu")


def test_get_predict_unit_for_test_dispatches_on_path(uma_s_1p1_checkpoint):
    """
    get_predict_unit_for_test dispatches path-vs-name: a real filesystem
    path goes through load_predict_unit (empty refs), a registered name
    goes through pretrained_mlip.get_predict_unit (populated refs).
    """
    # Path branch
    pu_path = get_predict_unit_for_test(uma_s_1p1_checkpoint, device="cpu")
    assert pu_path.atom_refs == {}

    # Name branch
    pu_name = get_predict_unit_for_test("uma-s-1p1", device="cpu")
    assert pu_name.atom_refs != {}


def test_get_predict_unit_for_test_symlink(tmp_path, uma_s_1p1_checkpoint):
    """
    A symlink to a checkpoint is still recognized as a file path,
    so get_predict_unit_for_test takes the path branch (empty refs).
    """
    link = tmp_path / "uma-s-1p1"
    os.symlink(uma_s_1p1_checkpoint, link)

    predictor = get_predict_unit_for_test(str(link), device="cpu")
    assert isinstance(predictor, MLIPPredictUnit)
    assert predictor.atom_refs == {}
    assert predictor.form_elem_refs == {}


def test_from_model_checkpoint_path_behavior_unchanged(uma_s_1p1_checkpoint):
    """
    Regression guard: FAIRChemCalculator.from_model_checkpoint with a path
    drops external refs. get_predict_unit_for_test matches this behavior.
    """
    calc = FAIRChemCalculator.from_model_checkpoint(
        uma_s_1p1_checkpoint, task_name="omat", device="cpu"
    )
    assert calc.predictor.atom_refs == {}
    assert calc.predictor.form_elem_refs == {}
