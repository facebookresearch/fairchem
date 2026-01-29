"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from ase import build

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


@pytest.mark.parametrize("radius_pbc_version", [1, 2])
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


@pytest.mark.parametrize("radius_pbc_version", [1, 2])
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


@pytest.mark.parametrize("external_graph_method", ["nvidia-cell", "nvidia-naive"])
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


@pytest.mark.parametrize("external_graph_method", ["nvidia-cell", "nvidia-naive"])
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


@pytest.mark.parametrize("external_graph_method", ["nvidia-cell", "nvidia-naive"])
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


def test_nvidia_methods_consistency():
    """Test that nvidia-cell and nvidia-naive produce identical results."""
    cutoff = 6.0
    max_neigh = 100
    atoms = build.bulk("Cu", "fcc", a=3.58)
    atoms.pbc = [True, True, True]

    # Generate graphs using both nvidia methods
    data_cell = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="nvidia-cell",
    )

    data_naive = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=cutoff,
        max_neigh=max_neigh,
        external_graph_method="nvidia-naive",
    )

    # Verify both methods produce identical edge counts
    assert (
        data_cell.nedges.item() == data_naive.nedges.item()
    ), f"nvidia-cell produced {data_cell.nedges.item()} edges, nvidia-naive produced {data_naive.nedges.item()}"

    # Verify edges match exactly
    assert _validate_edges_match(
        data_cell, data_naive
    ), "nvidia-cell and nvidia-naive produced different edge sets"


@pytest.mark.parametrize("external_graph_method", ["nvidia-cell", "nvidia-naive"])
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


@pytest.mark.parametrize("external_graph_method", ["nvidia-cell", "nvidia-naive"])
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
