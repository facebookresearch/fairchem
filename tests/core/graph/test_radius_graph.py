"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase import Atoms, build

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph


@pytest.mark.parametrize("radius_pbc_version", [1, 2])
def test_radius_graph_1d(radius_pbc_version):
    cutoff = 6.0
    atoms = build.bulk("Cu", "fcc", a=3.58)  # minimum distance is 2.53
    atoms.pbc = [True, False, False]
    data_dict = AtomicData.from_ase(atoms)

    # case with number of neighbors within max_neighbors
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=10, 
        enforce_max_neighbors_strictly=False, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 4
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=10, 
        enforce_max_neighbors_strictly=True, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"]
    )
    assert graph_dict["neighbors"] == 4

    # case with number of neighbors exceeding max_neighbors
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=1, 
        enforce_max_neighbors_strictly=False, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 2
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=1, 
        enforce_max_neighbors_strictly=True, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"]
    )
    assert graph_dict["neighbors"] == 1

    # case without max_neighbors
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=-1, 
        enforce_max_neighbors_strictly=False, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 4
    graph_dict = generate_graph(
        data_dict, 
        cutoff=cutoff, 
        max_neighbors=-1, 
        enforce_max_neighbors_strictly=True, 
        radius_pbc_version=radius_pbc_version, 
        pbc=data_dict["pbc"]
    )
    assert graph_dict["neighbors"] == 4