"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import ase
import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
    warn_if_upcasting,
)
from fairchem.core.graph.compute import get_pbc_distances


@pytest.fixture()
def ase_atoms():
    return molecule("H2O")


def test_to_ase_single(ase_atoms):
    atoms = AtomicData.from_ase(ase_atoms).to_ase_single()
    assert atoms.get_chemical_formula() == "H2O"


@pytest.mark.gpu()
def test_to_ase_single_cuda(ase_atoms):
    atomic_data = AtomicData.from_ase(ase_atoms).cuda()
    atoms = atomic_data.to_ase_single()
    assert atoms.get_chemical_formula() == "H2O"


@pytest.fixture()
def batch_edgeless():
    # Create AtomicData batch of two ase.Atoms molecules without edges
    ase_atoms = ase.Atoms(positions=[[0.5, 0, 0], [1, 0, 0]], cell=(2, 2, 2), pbc=True)
    atomicdata_list_edgeless = [AtomicData.from_ase(ase_atoms) for _ in range(2)]
    batch_edgeless = atomicdata_list_to_batch(atomicdata_list_edgeless)
    return batch_edgeless


def test_to_ase_batch(batch_edgeless):
    # Define edge targets
    edge_index = torch.tensor([[1, 0, 3, 2], [0, 1, 2, 3]])
    cell_offsets = torch.zeros((4, 3))
    neighbors = torch.tensor([2, 2])
    # or equivalently:
    # edge_index, cell_offsets, neighbors = radius_graph_pbc_v2(
    #     batch_edgeless,
    #     radius=1,
    #     max_num_neighbors_threshold=100,
    #     pbc=batch_edgeless["pbc"][0],  # use the PBC from molecule 0
    # )

    # Add edge information to batch and check it is correct
    batch = batch_edgeless.clone()
    batch.update_batch_edges(edge_index, cell_offsets, neighbors)
    # or equivalently:
    # batch = batch_edgeless.update_batch_edges(edge_index, cell_offsets, neighbors)
    assert (batch.edge_index == edge_index).all()

    # Note: if we simply do `batch.edge_index = edge_index`, there will be no edges
    # after unbatching because `batch.__slices__` would contain only zeros.

    # Unbatch and check that edges have been added correctly
    atomicdata_list = batch.batch_to_atomicdata_list()
    assert (atomicdata_list[0].edge_index == edge_index[:, :2]).all()
    assert (atomicdata_list[1].edge_index == edge_index[:, :2]).all()


def test_warn_if_upcasting(caplog):
    """
    Test that warn_if_upcasting logs when upcasting and is silent otherwise.
    """
    # Upcasting float32 -> float64 should warn
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        result = warn_if_upcasting(torch.float32, torch.float64)
        assert result is True
        assert "Upcasting atomic coordinates" in caplog.text

    # Same dtype should not warn
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        result = warn_if_upcasting(torch.float64, torch.float64)
        assert result is False
        assert caplog.text == ""

    # Downcasting float64 -> float32 should not warn
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        result = warn_if_upcasting(torch.float64, torch.float32)
        assert result is False
        assert caplog.text == ""


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_get_pbc_distances_preserves_dtype(dtype):
    """
    Test that get_pbc_distances returns distances and vectors
    in the same dtype as the inputs, verifying the change from
    hardcoded .float() to .to(dtype=cell.dtype).
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 5.0
    edge_index = torch.tensor([[0, 1], [1, 0]])
    # cell_offsets: second edge wraps through the periodic boundary
    cell_offsets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    neighbors = torch.tensor([2])

    out = get_pbc_distances(
        pos,
        edge_index,
        cell,
        cell_offsets,
        neighbors,
        return_distance_vec=True,
        return_offsets=True,
    )

    assert out["distances"].dtype == dtype
    assert out["distance_vec"].dtype == dtype
    assert out["offsets"].dtype == dtype
