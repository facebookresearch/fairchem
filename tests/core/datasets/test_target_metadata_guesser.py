"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from fairchem.core.datasets.target_metadata_guesser import (
    guess_property_metadata,
    guess_target_metadata,
    target_extensive,
    target_per_atom,
)


@pytest.mark.parametrize(
    "lengths, samples, shape, kind, extensive",
    [
        ([2, 4], [2.0, 4.0], (), "per-image", True),
        ([2, 8], [1.0, 1.0], (), "per-image", False),
        ([4, 4], [1.0, 2.0], (), "per-image", True),
        ([4, 4], [np.ones((4, 3))] * 2, ("N", 3), "per-atom", True),
        ([2, 4], [np.ones((2, 3)), np.ones((4, 3))], (3,), "per-atom", True),
        ([2, 4], [np.ones(2), np.ones(4)], (), "per-atom", True),
        ([2, 4], [np.ones(3), np.ones(5)], None, "unknown", None),
        ([2, 4], [np.array(object()), np.array(object())], None, "unknown", None),
        ([2, 4], [np.array([2.0, 4.0]), np.array([4.0, 8.0])], (2,), "per-image", True),
    ],
)
def test_guess_target_metadata(lengths, samples, shape, kind, extensive):
    result = guess_target_metadata(lengths, [np.asarray(sample) for sample in samples])
    assert result["shape"] == shape
    assert result["type"] == kind
    assert result["extensive"] == extensive


def test_per_atom_requires_consistent_remaining_dimensions():
    assert not target_per_atom([2, 4], [np.ones((2, 3)), np.ones((4, 2))])
    assert not target_per_atom([2, 4], [np.array(1), np.array(2)])


def test_extensive_rejects_variable_shape_or_nonnumeric_values():
    with pytest.raises(AssertionError, match="not constant"):
        target_extensive([2, 4], [np.ones(2), np.ones(4)])
    assert not target_extensive([2, 4], [np.array("a"), np.array("b")])


@pytest.mark.parametrize("voigt", [False, True])
def test_guess_ase_properties_and_stress(voigt):
    structures = []
    for n in (2, 4):
        atoms = Atoms("H" * n)
        atoms.info["custom_energy"] = float(n)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(n),
            forces=np.ones((n, 3)),
            stress=np.ones(6) if voigt else np.ones((3, 3)),
        )
        structures.append(atoms)
    metadata = guess_property_metadata(structures)
    assert metadata["info.custom_energy"]["extensive"]
    assert metadata["energy"]["shape"] == ()
    assert metadata["forces"]["type"] == "per-atom"
    assert metadata["forces"]["shape"] == (3,)
    assert metadata["stress"]["shape"] == (3, 3)
    assert not metadata["stress"]["extensive"]
    assert guess_property_metadata([Atoms("H")]) == {}
