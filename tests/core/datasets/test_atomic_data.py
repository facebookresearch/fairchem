"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from ase.build import molecule

from fairchem.core.datasets.atomic_data import AtomicData


@pytest.fixture
def ase_atoms():
    return molecule("H2O")


def test_to_ase_single(ase_atoms):
    atoms = AtomicData.from_ase(ase_atoms).to_ase_single()
    assert atoms.get_chemical_formula() == "H2O"
