"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase import build

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import (
    filter_edges_by_node_partition,
    generate_graph,
    get_pbc_distances,
)


@pytest.mark.parametrize("radius_pbc_version", [1, 2, 3])
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
        pbc=data_dict["pbc"],
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
        pbc=data_dict["pbc"],
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
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 4


@pytest.mark.parametrize("radius_pbc_version", [1, 2, 3])
def test_radius_graph_2d(radius_pbc_version):
    cutoff = 6.0
    atoms = build.bulk("Cu", "fcc", a=3.58)  # minimum distance is 2.53
    atoms.pbc = [True, True, False]
    data_dict = AtomicData.from_ase(atoms)

    # case with number of neighbors within max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=20,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 18
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=20,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 18

    # case with number of neighbors exceeding max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=2,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 6
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=2,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 2

    # case without max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=-1,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 18
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=-1,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 18


@pytest.mark.parametrize("radius_pbc_version", [1, 2, 3])
def test_radius_graph_3d(radius_pbc_version):
    cutoff = 6.0
    atoms = build.bulk("Cu", "fcc", a=3.58)  # minimum distance is 2.53
    atoms.pbc = [True, True, True]
    data_dict = AtomicData.from_ase(atoms)

    # case with number of neighbors within max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=100,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 78
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=100,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 78

    # case with number of neighbors exceeding max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=10,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 12
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=10,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 10

    # case without max_neighbors
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=-1,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 78
    graph_dict = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=-1,
        enforce_max_neighbors_strictly=True,
        radius_pbc_version=radius_pbc_version,
        pbc=data_dict["pbc"],
    )
    assert graph_dict["neighbors"] == 78


def _validate_edges_match(data1, data2):
    """Helper function to validate that two graph datasets have matching edges."""
    if data1.nedges.item() != data2.nedges.item():
        return False

    # Convert edge indices to sets of tuples for comparison
    edges1 = set()
    edges2 = set()

    for i in range(data1.nedges.item()):
        edge1 = (
            data1.edge_index[1, i].item(),
            data1.edge_index[0, i].item(),
            data1.cell_offsets[i, 0].item(),
            data1.cell_offsets[i, 1].item(),
            data1.cell_offsets[i, 2].item(),
        )
        edges1.add(edge1)

        edge2 = (
            data2.edge_index[1, i].item(),
            data2.edge_index[0, i].item(),
            data2.cell_offsets[i, 0].item(),
            data2.cell_offsets[i, 1].item(),
            data2.cell_offsets[i, 2].item(),
        )
        edges2.add(edge2)

    return edges1 == edges2


@pytest.mark.parametrize("external_graph_method", ["nvidia"])
def test_nvidia_graph_1d(external_graph_method):
    """Test nvidia graph generation methods with 1D PBC."""
    cutoff = 6.0
    max_neigh = 10
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms.pbc = [True, False, False]

    # Generate graph using nvidia method
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method=external_graph_method,
    )

    # Verify basic properties
    assert data.nedges is not None
    assert data.edge_index is not None
    assert data.edge_index.shape[0] == 2  # [2, num_edges]
    assert data.cell_offsets is not None

    # Generate reference graph using pymatgen
    data_ref = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="pymatgen",
    )

    # Verify edge counts match
    assert (
        data.nedges.item() == data_ref.nedges.item()
    ), f"{external_graph_method} produced {data.nedges.item()} edges, expected {data_ref.nedges.item()}"

    # Verify edges match
    assert _validate_edges_match(
        data, data_ref
    ), f"{external_graph_method} produced different edges than pymatgen"


@pytest.mark.parametrize("external_graph_method", ["nvidia"])
def test_nvidia_graph_2d(external_graph_method):
    """Test nvidia graph generation methods with 2D PBC."""
    cutoff = 6.0
    max_neigh = 20
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms.pbc = [True, True, False]

    # Generate graph using nvidia method
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method=external_graph_method,
    )

    # Verify basic properties
    assert data.nedges is not None
    assert data.edge_index is not None
    assert data.edge_index.shape[0] == 2
    assert data.cell_offsets is not None

    # Generate reference graph using pymatgen
    data_ref = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="pymatgen",
    )

    # Verify edge counts match
    assert (
        data.nedges.item() == data_ref.nedges.item()
    ), f"{external_graph_method} produced {data.nedges.item()} edges, expected {data_ref.nedges.item()}"

    # Verify edges match
    assert _validate_edges_match(
        data, data_ref
    ), f"{external_graph_method} produced different edges than pymatgen"


@pytest.mark.parametrize("external_graph_method", ["nvidia"])
def test_nvidia_graph_3d(external_graph_method):
    """Test nvidia graph generation methods with 3D PBC."""
    cutoff = 6.0
    max_neigh = 100
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms.pbc = [True, True, True]

    # Generate graph using nvidia method
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method=external_graph_method,
    )

    # Verify basic properties
    assert data.nedges is not None
    assert data.edge_index is not None
    assert data.edge_index.shape[0] == 2
    assert data.cell_offsets is not None

    # Generate reference graph using pymatgen
    data_ref = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="pymatgen",
    )

    # Verify edge counts match
    assert (
        data.nedges.item() == data_ref.nedges.item()
    ), f"{external_graph_method} produced {data.nedges.item()} edges, expected {data_ref.nedges.item()}"

    # Verify edges match
    assert _validate_edges_match(
        data, data_ref
    ), f"{external_graph_method} produced different edges than pymatgen"


@pytest.mark.parametrize("external_graph_method", ["nvidia"])
def test_nvidia_graph_max_neighbors(external_graph_method):
    """Test nvidia graph methods with varying max_neighbors settings."""
    cutoff = 6.0
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms.pbc = [True, True, True]

    # Test with sufficient max_neigh
    data_large = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=100,
        external_graph_method=external_graph_method,
    )

    # Test with limited max_neigh
    data_small = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=10,
        external_graph_method=external_graph_method,
    )

    # The limited version should have fewer or equal edges
    assert (
        data_small.nedges.item() <= data_large.nedges.item()
    ), "Limited max_neigh produced more edges than unlimited"

    # Verify that limited version actually limits the edges
    assert (
        data_small.nedges.item() < data_large.nedges.item()
    ), "max_neigh=10 should produce fewer edges than max_neigh=100"


@pytest.mark.parametrize("external_graph_method", ["nvidia"])
def test_nvidia_graph_larger_system(external_graph_method):
    """Test nvidia graph methods with a larger system."""
    cutoff = 6.0
    max_neigh = 300

    # Create a larger system (2x2x2 supercell)
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms = atoms * (2, 2, 2)  # 32 atoms
    atoms.pbc = [True, True, True]

    # Generate graph using nvidia method
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method=external_graph_method,
    )

    # Generate reference graph using pymatgen
    data_ref = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="pymatgen",
    )

    # Verify edge counts match
    assert (
        data.nedges.item() == data_ref.nedges.item()
    ), f"{external_graph_method} produced {data.nedges.item()} edges, expected {data_ref.nedges.item()}"

    # Verify edges match
    assert _validate_edges_match(
        data, data_ref
    ), f"{external_graph_method} produced different edges than pymatgen for larger system"


class TestFilterEdgesByNodePartition:
    """Test filter_edges_by_node_partition function."""

    def test_filter_keeps_correct_edges(self):
        """Test that only edges with target atoms in node_partition are kept."""
        # 4 atoms, edges: 0->1, 0->2, 1->0, 1->3, 2->3, 3->2
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 3, 2]])
        cell_offsets = torch.zeros(6, 3, dtype=torch.long)
        neighbors = torch.tensor([6])  # single system with 6 edges
        node_partition = torch.tensor([0, 1])  # only atoms 0 and 1

        new_edge_index, new_cell_offsets, new_neighbors = (
            filter_edges_by_node_partition(
                node_partition, edge_index, cell_offsets, neighbors, num_atoms=4
            )
        )

        # Should keep edges where target (row 1) is in {0, 1}: edges 0->1, 1->0
        assert new_edge_index.shape[1] == 2
        edge_pairs = {
            (new_edge_index[0, i].item(), new_edge_index[1, i].item())
            for i in range(new_edge_index.shape[1])
        }
        assert edge_pairs == {(0, 1), (1, 0)}
        assert new_neighbors.sum().item() == 2

    def test_filter_with_multiple_systems(self):
        """Test filtering with batched systems."""
        # 2 systems: system 0 has 3 edges, system 1 has 2 edges
        edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 0, 3, 2]])
        cell_offsets = torch.zeros(5, 3, dtype=torch.long)
        neighbors = torch.tensor([3, 2])  # 3 edges in sys0, 2 in sys1
        node_partition = torch.tensor([0, 2])  # atoms 0 and 2

        new_edge_index, new_cell_offsets, new_neighbors = (
            filter_edges_by_node_partition(
                node_partition, edge_index, cell_offsets, neighbors, num_atoms=4
            )
        )

        # Edges with target in {0, 2}: 0->2, 1->0, 3->2
        assert new_edge_index.shape[1] == 3
        # System 0 had edges 0->1, 0->2, 1->0; targets {0,2} keeps 0->2, 1->0 (2 edges)
        # System 1 had edges 2->3, 3->2; targets {0,2} keeps 3->2 (1 edge)
        assert new_neighbors.tolist() == [2, 1]

    def test_filter_empty_partition(self):
        """Test with empty node partition returns no edges."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        cell_offsets = torch.zeros(2, 3, dtype=torch.long)
        neighbors = torch.tensor([2])
        node_partition = torch.tensor([], dtype=torch.long)

        new_edge_index, new_cell_offsets, new_neighbors = (
            filter_edges_by_node_partition(
                node_partition, edge_index, cell_offsets, neighbors, num_atoms=2
            )
        )

        assert new_edge_index.shape[1] == 0
        assert new_neighbors.numel() == 0

    def test_filter_all_atoms_in_partition(self):
        """Test with all atoms in partition keeps all edges."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        cell_offsets = torch.zeros(3, 3, dtype=torch.long)
        neighbors = torch.tensor([3])
        node_partition = torch.tensor([0, 1, 2])

        new_edge_index, new_cell_offsets, new_neighbors = (
            filter_edges_by_node_partition(
                node_partition, edge_index, cell_offsets, neighbors, num_atoms=3
            )
        )

        assert new_edge_index.shape[1] == 3
        assert new_neighbors.tolist() == [3]

    def test_filter_preserves_cell_offsets(self):
        """Test that cell_offsets are correctly filtered."""
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]])
        cell_offsets = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        neighbors = torch.tensor([3])
        node_partition = torch.tensor([1])  # only atom 1

        new_edge_index, new_cell_offsets, new_neighbors = (
            filter_edges_by_node_partition(
                node_partition, edge_index, cell_offsets, neighbors, num_atoms=3
            )
        )

        # Edges with target=1: 0->1, 2->1
        assert new_edge_index.shape[1] == 2
        assert new_cell_offsets.shape[0] == 2
        # Check offsets match the kept edges
        offsets_set = {tuple(new_cell_offsets[i].tolist()) for i in range(2)}
        assert offsets_set == {(1, 0, 0), (0, 0, 1)}


class TestGetPbcDistances:
    """Test get_pbc_distances function."""

    def test_basic_distances_no_pbc(self):
        """Test distance calculation without periodic boundary conditions."""
        # 3 atoms in a line: 0 at origin, 1 at (1,0,0), 2 at (3,0,0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        edge_index = torch.tensor(
            [[0, 1, 0, 2], [1, 0, 2, 0]]
        )  # 0->1, 1->0, 0->2, 2->0
        cell = torch.eye(3).unsqueeze(0) * 10.0  # single system, 10x10x10 cell
        cell_offsets = torch.zeros(4, 3, dtype=torch.long)  # no PBC offsets
        neighbors = torch.tensor([4])

        out = get_pbc_distances(
            pos,
            edge_index,
            cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        # Distances: 0->1 = 1.0, 1->0 = 1.0, 0->2 = 3.0, 2->0 = 3.0
        expected_distances = torch.tensor([1.0, 1.0, 3.0, 3.0])
        assert torch.allclose(out["distances"], expected_distances)

    def test_distances_with_pbc_offsets(self):
        """Test distance calculation with periodic boundary condition offsets."""
        # 2 atoms at opposite ends of a 10 Å cell
        pos = torch.tensor([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])  # 0->1
        cell = torch.eye(3).unsqueeze(0) * 10.0  # 10x10x10 cell
        # Offset of [-1, 0, 0] means atom 1 is in the previous cell image
        cell_offsets = torch.tensor([[-1, 0, 0]])
        neighbors = torch.tensor([1])

        get_pbc_distances(
            pos,
            edge_index,
            cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        # Without offset: distance = 9.5 - 0.5 = 9.0
        # With offset [-1,0,0]: distance = 0.5 - 9.5 + (-1)*10 = 0.5 - 9.5 - 10 = -19 -> |1.0|
        # Actually: distance_vec = pos[0] - pos[1] + offset = (0.5-9.5, 0, 0) + (-10, 0, 0) = (-19, 0, 0)
        # Hmm, let me recalculate: row=0, col=1, distance_vec = pos[row] - pos[col] = (0.5, 0, 0) - (9.5, 0, 0) = (-9, 0, 0)
        # Then offset = cell_offsets @ cell = [-1, 0, 0] @ [[10,0,0],[0,10,0],[0,0,10]] = (-10, 0, 0)
        # distance_vec += offset => (-9, 0, 0) + (-10, 0, 0) = (-19, 0, 0) -> distance = 19.0
        # That's the distance going through periodic image in the negative direction
        # For minimum image, we'd use offset [+1, 0, 0]: (-9, 0, 0) + (10, 0, 0) = (1, 0, 0) -> distance = 1.0

        # Let's test with the positive offset for minimum image
        cell_offsets_min = torch.tensor([[1, 0, 0]])
        out_min = get_pbc_distances(
            pos,
            edge_index,
            cell,
            cell_offsets_min,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )
        assert torch.allclose(out_min["distances"], torch.tensor([1.0]))

    def test_returns_distance_vec(self):
        """Test that distance vectors are correctly returned."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])
        cell = torch.eye(3).unsqueeze(0) * 10.0
        cell_offsets = torch.zeros(1, 3, dtype=torch.long)
        neighbors = torch.tensor([1])

        out = get_pbc_distances(
            pos, edge_index, cell, cell_offsets, neighbors, return_distance_vec=True
        )

        expected_vec = torch.tensor([[-3.0, -4.0, 0.0]])
        assert torch.allclose(out["distance_vec"], expected_vec)
        assert torch.allclose(out["distances"], torch.tensor([5.0]))

    def test_removes_zero_distances(self):
        """Test that zero-distance edges are removed."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # Edge 0->2 has zero distance (same position)
        edge_index = torch.tensor([[0, 0], [1, 2]])
        cell = torch.eye(3).unsqueeze(0) * 10.0
        cell_offsets = torch.zeros(2, 3, dtype=torch.long)
        neighbors = torch.tensor([2])

        out = get_pbc_distances(pos, edge_index, cell, cell_offsets, neighbors)

        # Zero-distance edge should be removed
        assert out["edge_index"].shape[1] == 1
        assert out["distances"].shape[0] == 1
        assert torch.allclose(out["distances"], torch.tensor([1.0]))

    def test_multiple_systems_in_batch(self):
        """Test with multiple systems batched together."""
        # System 0: 2 atoms, System 1: 2 atoms
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # System 0
                [0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],  # System 1
            ]
        )
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3],  # sources
                [1, 0, 3, 2],  # targets
            ]
        )
        cell = torch.stack([torch.eye(3) * 10.0, torch.eye(3) * 10.0])  # 2 cells
        cell_offsets = torch.zeros(4, 3, dtype=torch.long)
        neighbors = torch.tensor([2, 2])  # 2 edges per system

        out = get_pbc_distances(
            pos, edge_index, cell, cell_offsets, neighbors, return_distance_vec=True
        )

        # System 0: distances = 2.0, 2.0
        # System 1: distances = 3.0, 3.0
        expected_distances = torch.tensor([2.0, 2.0, 3.0, 3.0])
        assert torch.allclose(out["distances"], expected_distances)

    def test_returns_offsets(self):
        """Test that offset distances are correctly returned."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])
        cell = torch.eye(3).unsqueeze(0) * 10.0
        cell_offsets = torch.tensor([[1, 0, 0]])  # offset in x direction
        neighbors = torch.tensor([1])

        out = get_pbc_distances(
            pos, edge_index, cell, cell_offsets, neighbors, return_offsets=True
        )

        # Offset should be cell_offsets @ cell = [1,0,0] @ [[10,0,0],...] = [10,0,0]
        expected_offsets = torch.tensor([[10.0, 0.0, 0.0]])
        assert torch.allclose(out["offsets"], expected_offsets)
