"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from ase import Atoms

from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.atomic_data import AtomicData


@pytest.fixture()
def pyg_data():
    return AtomicData.from_ase(
        Atoms(
            "HCCC",
            positions=[(0, 0, 0), (-1, 0, 0), (1, 0, 0), (2, 0, 0)],
            info={"energy": 123.4},
        )
    )


def test_rename_data_object_keys(pyg_data):
    assert "energy" in pyg_data
    key_mapping = {"energy": "test_energy"}
    pyg_data = rename_data_object_keys(pyg_data, key_mapping)
    assert "energy" not in pyg_data
    assert "test_energy" in pyg_data
    key_mapping = {"test_energy": "test_energy"}
    pyg_data = rename_data_object_keys(pyg_data, key_mapping)
    assert "test_energy" in pyg_data
