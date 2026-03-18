"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase.io import read

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import get_pbc_distances


@pytest.fixture(scope="class")
def load_data(request) -> None:
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    request.cls.data = AtomicData.from_ase(
        atoms,
        max_neigh=12,
        radius=6,
        r_edges=True,
    )


@pytest.mark.usefixtures("load_data")
class TestPBC:
    def test_pbc_distances(self) -> None:
        data = self.data
        batch = data_list_collater([data] * 5)
        out = get_pbc_distances(
            batch.pos,
            batch.edge_index,
            batch.cell,
            batch.cell_offsets,
            batch.neighbors,
        )
        edge_index, _ = out["edge_index"], out["distances"]

        np.testing.assert_array_equal(
            batch.edge_index,
            edge_index,
        )
