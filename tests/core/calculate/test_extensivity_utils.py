"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import make_supercell

from tests.core.calculate.extensivity_utils import (
    assert_cell_major_forces_close,
    expected_extensivity_failure,
)


@pytest.mark.parametrize(
    "repeats", [(2, 2, 2), (2, 1, 1), (4, 4, 4), (4, 1, 1), (3, 4, 1)]
)
def test_cell_major_force_correspondence(repeats):
    unit = Atoms(
        "HHO",
        scaled_positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        cell=np.diag([3.0, 4.0, 5.0]),
        pbc=True,
    )
    unit.new_array("unit_cell_index", np.arange(len(unit)))
    forces_unit = np.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    supercell = make_supercell(unit, np.diag(repeats), order="cell-major")
    forces_super = forces_unit[supercell.arrays["unit_cell_index"]]
    assert_cell_major_forces_close(
        forces_super, forces_unit, int(np.prod(repeats)), atol=1e-10
    )


def test_force_noise_cannot_change_atom_correspondence():
    expected = np.array([[1.0, 0.0, -1e-8], [-1.0, 0.0, 1e-8]])
    actual = np.array([[1.0, 0.0, 1e-8], [-1.0, 0.0, -1e-8]])
    # Independent lexsorts reverse these rows and produce a false 2.0 error.
    assert_cell_major_forces_close(actual, expected, 1, atol=1e-6)


def test_force_comparison_rejects_deviation():
    expected = np.array([[0.0, 0.0, 0.0]])
    actual = np.array([[2e-6, 0.0, 0.0]])
    with pytest.raises(AssertionError, match="Supercell forces differ"):
        assert_cell_major_forces_close(actual, expected, 1, atol=1e-6)


def test_force_comparison_preserves_relative_tolerance():
    expected = np.array([[10.0, 0.0, 0.0]])
    actual = np.array([[10.0 + 5e-7, 0.0, 0.0]])
    assert_cell_major_forces_close(actual, expected, 1, atol=1e-10)
    with pytest.raises(AssertionError, match="Supercell forces differ"):
        assert_cell_major_forces_close(
            actual + [2e-6, 0.0, 0.0], expected, 1, atol=1e-10
        )


def test_force_comparison_preserves_replica_count():
    expected = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with pytest.raises(AssertionError):
        assert_cell_major_forces_close(expected, expected, 2, atol=1e-6)


def test_force_comparison_rejects_exchanging_distinct_atomic_sites():
    forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with pytest.raises(AssertionError, match="Supercell forces differ"):
        assert_cell_major_forces_close(forces[::-1], forces, 1, atol=1e-6)


def test_known_extensivity_numeric_failure_is_expected():
    with (
        pytest.raises(pytest.xfail.Exception, match="known extensivity regression"),
        expected_extensivity_failure("uma-s-1p2", "PBC energy"),
    ):
        np.testing.assert_allclose(1.0, 2.0)


@pytest.mark.parametrize("checkpoint", ["uma-s-1p1", "uma-s-1p2p1"])
def test_other_checkpoints_keep_numeric_failure(checkpoint):
    with (
        pytest.raises(AssertionError),
        expected_extensivity_failure(checkpoint, "PBC energy"),
    ):
        np.testing.assert_allclose(1.0, 2.0)


@pytest.mark.parametrize("error", [RuntimeError, ImportError, ValueError])
def test_known_extensivity_marker_keeps_unexpected_errors(error):
    with (
        pytest.raises(error, match="unrelated failure"),
        expected_extensivity_failure("uma-s-1p2", "PBC energy"),
    ):
        raise error("unrelated failure")


def test_known_extensivity_model_can_pass_unmasked():
    with expected_extensivity_failure("uma-s-1p2", "PBC energy"):
        np.testing.assert_allclose(2.0, 2.0)
