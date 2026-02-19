"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import pytest
import torch
from ase import Atoms, build
from ase.build import molecule
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import (
    get_fcc_crystal_by_num_atoms,
    get_fcc_crystal_by_num_cells,
    get_water_box,
)
from fairchem.core.graph.compute import generate_graph
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc, radius_graph_pbc_v2


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


def _graph_dict_to_edge_set(graph_dict):
    """Convert a graph dict from generate_graph to a set of edge tuples for comparison."""
    edges = set()
    edge_index = graph_dict["edge_index"]
    cell_offsets = graph_dict["cell_offsets"]
    num_edges = edge_index.shape[1]

    for i in range(num_edges):
        edge = (
            edge_index[0, i].item(),
            edge_index[1, i].item(),
            cell_offsets[i, 0].item(),
            cell_offsets[i, 1].item(),
            cell_offsets[i, 2].item(),
        )
        edges.add(edge)

    return edges


@pytest.fixture()
def copper_bulk_atoms():
    """Create a copper bulk structure."""
    return get_fcc_crystal_by_num_cells(
        n_cells=5, atom_type="Cu", lattice_constant=3.58
    )


@pytest.fixture()
def water_box_atoms():
    """Create a random water box structure."""
    return get_water_box(num_molecules=10, box_size=10.0, seed=42)


@pytest.mark.parametrize(
    "cutoff, max_neighbors",
    [(6.0, 30), (6.0, 300), (20.0, 30), (12.0, 10000), (1.0, 1)],
)
@pytest.mark.parametrize("atoms_fixture", ["copper_bulk_atoms", "water_box_atoms"])
def test_radius_pbc_version_2_and_3_produce_identical_edges(
    cutoff, max_neighbors, atoms_fixture, request
):
    """Test that radius_pbc_version 2 and 3 produce identical edges after sorting."""
    atoms = request.getfixturevalue(atoms_fixture)
    data_dict = AtomicData.from_ase(atoms)

    # Generate graph using version 2
    graph_v2 = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
        pbc=data_dict["pbc"],
    )

    # Generate graph using version 3
    graph_v3 = generate_graph(
        data_dict,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=3,
        pbc=data_dict["pbc"],
    )

    # Verify neighbor counts match
    assert (
        graph_v2["neighbors"] == graph_v3["neighbors"]
    ), f"Neighbor counts differ: v2={graph_v2['neighbors']}, v3={graph_v3['neighbors']}"

    # Convert to sets of edges for order-independent comparison
    edges_v2 = _graph_dict_to_edge_set(graph_v2)
    edges_v3 = _graph_dict_to_edge_set(graph_v3)

    # Verify edge sets match
    assert edges_v2 == edges_v3, (
        f"Edge sets differ between radius_pbc_version 2 and 3.\n"
        f"Edges only in v2: {edges_v2 - edges_v3}\n"
        f"Edges only in v3: {edges_v3 - edges_v2}"
    )


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


# ==============================================================================
# Tests merged from test_radius_graph_pbc.py
# ==============================================================================


@pytest.fixture(scope="class")
def load_data(request) -> None:
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    request.cls.data = AtomicData.from_ase(
        atoms, max_neigh=200, radius=6, r_edges=True, r_data_keys=["spin", "charge"]
    )


def check_features_match(
    edge_index_1, cell_offsets_1, edge_index_2, cell_offsets_2
) -> bool:
    # Combine both edge indices and offsets to one tensor
    features_1 = torch.cat((edge_index_1, cell_offsets_1.T), dim=0).T
    features_2 = torch.cat((edge_index_2, cell_offsets_2.T), dim=0).T.long()

    # Convert rows of tensors to sets. The order of edges is not guaranteed
    features_1_set = {tuple(x.tolist()) for x in features_1}
    features_2_set = {tuple(x.tolist()) for x in features_2}

    # Ensure sets are not empty
    assert len(features_1_set) > 0
    assert len(features_2_set) > 0

    # Ensure sets are the same
    assert features_1_set == features_2_set

    return True


@pytest.mark.usefixtures("load_data")
class TestRadiusGraphPBC:
    def test_radius_graph_pbc(self) -> None:
        data = self.data
        batch = data_list_collater([data] * 5)
        generated_graphs = generate_graph(
            data=batch,
            cutoff=6,
            max_neighbors=2000,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=1,
            pbc=torch.BoolTensor([[True, True, False]] * 5),
        )
        assert check_features_match(
            batch.edge_index,
            batch.cell_offsets,
            generated_graphs["edge_index"],
            generated_graphs["cell_offsets"],
        )

    def test_bulk(self) -> None:
        radius = 10

        # Must be sufficiently large to ensure all edges are retained
        max_neigh = 2000

        a2g = partial(
            AtomicData.from_ase,
            max_neigh=max_neigh,
            radius=radius,
            r_edges=True,
            r_data_keys=["spin", "charge"],
        )

        structure = FaceCenteredCubic("Pt", size=[1, 2, 3])
        data = a2g(structure)
        batch = data_list_collater([data])

        # Ensure adequate distance between repeated cells
        structure.cell[0] *= radius
        structure.cell[1] *= radius
        structure.cell[2] *= radius

        # [False, False, False]
        data = a2g(structure)
        non_pbc = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [True, False, False]
        structure.cell[0] /= radius
        data = a2g(structure)
        pbc_x = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([True, False, False]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [True, True, False]
        structure.cell[1] /= radius
        data = a2g(structure)
        pbc_xy = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([True, True, False]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [False, True, False]
        structure.cell[0] *= radius
        data = a2g(structure)
        pbc_y = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([False, True, False]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [False, True, True]
        structure.cell[2] /= radius
        data = a2g(structure)
        pbc_yz = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([False, True, True]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [False, False, True]
        structure.cell[1] *= radius
        data = a2g(structure)
        pbc_z = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([False, False, True]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [True, False, True]
        structure.cell[0] /= radius
        data = a2g(structure)
        pbc_xz = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([True, False, True]),
        )
        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # [True, True, True]
        structure.cell[1] /= radius
        data = a2g(structure)
        pbc_all = data.edge_index.shape[1]

        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([True, True, True]),
        )

        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])

        # Ensure edges are actually found
        assert non_pbc > 0
        assert pbc_x > non_pbc
        assert pbc_y > non_pbc
        assert pbc_z > non_pbc
        assert pbc_xy > max(pbc_x, pbc_y)
        assert pbc_yz > max(pbc_y, pbc_z)
        assert pbc_xz > max(pbc_x, pbc_z)
        assert pbc_all > max(pbc_xy, pbc_yz, pbc_xz)

    def test_molecule(self) -> None:
        radius = 6
        max_neigh = 1000
        structure = molecule("CH3COOH")
        structure.cell = [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
        data = AtomicData.from_ase(
            structure, radius=radius, max_neigh=max_neigh, r_edges=True
        )
        batch = data_list_collater([data])
        out = radius_graph_pbc(
            batch,
            radius=radius,
            max_num_neighbors_threshold=max_neigh,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert check_features_match(data.edge_index, data.cell_offsets, out[0], out[1])


@pytest.mark.parametrize(
    ("atoms,expected_edge_index,max_neighbors,enforce_max_neighbors_strictly"),
    [
        (
            Atoms("HCCC", positions=[(0, 0, 0), (-1, 0, 0), (1, 0, 0), (2, 0, 0)]),
            # we currently have an off by one in our code, the answer should be this
            # but for now lets just stay consistent
            # tensor([[1, 2, 0, 0, 3, 2], [0, 0, 1, 2, 2, 3]]) # [ with fix ]
            torch.tensor([[1, 2, 0, 2, 0, 3, 0, 2], [0, 0, 1, 1, 2, 2, 3, 3]]),
            1,
            False,
        ),
        (
            Atoms("HCCC", positions=[(0, 0, 0), (-1, 0, 0), (1, 0, 0), (2, 0, 0)]),
            # this could change since tie breaking order is undefined
            torch.tensor([[1, 0, 0, 2], [0, 1, 2, 3]]),
            1,
            True,
        ),
        (
            Atoms(
                "HCCCC",
                positions=[
                    (0, 0, 0),
                    (0, -1.0 / 20, 0),
                    (0, 1.0 / 5, 0),
                    (-1, 0, 0),
                    (1, 0, 0),
                ],
            ),
            # we currently have an off by one in our code, the answer should be this
            # but for now lets just stay consistent
            # tensor([[1, 0, 0, 0, 1, 0, 1],[0, 1, 2, 3, 3, 4, 4]])
            torch.tensor(
                [[1, 2, 0, 2, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]
            ),
            1,
            False,
        ),
        (
            Atoms(
                "HCCCC",
                positions=[
                    (0, 0, 0),
                    (0, -1.0 / 20, 0),
                    (0, 1.0 / 5, 0),
                    (-1, 0, 0),
                    (1, 0, 0),
                ],
            ),
            torch.tensor([[1, 0, 0, 0, 0], [0, 1, 2, 3, 4]]),
            1,
            True,
        ),
    ],
)
def test_simple_systems_nopbc(
    atoms,
    expected_edge_index,
    max_neighbors,
    enforce_max_neighbors_strictly,
    torch_deterministic,
):
    data = AtomicData.from_ase(atoms)

    batch = data_list_collater([data])

    for radius_graph_pbc_fn in (radius_graph_pbc_v2, radius_graph_pbc):
        edge_index, _, _ = radius_graph_pbc_fn(
            batch,
            radius=6,
            max_num_neighbors_threshold=max_neighbors,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert (
            len(
                {tuple(x) for x in edge_index.T.tolist()}
                - {tuple(x) for x in expected_edge_index.T.tolist()}
            )
            == 0
        )


@pytest.mark.parametrize(
    "atoms",
    [FaceCenteredCubic("Cu", size=(2, 2, 2), latticeconstant=5.0), molecule("H2O")],
)
def test_pymatgen_vs_internal_graph(atoms):
    radius = 10
    max_neigh = 200

    for radius_pbc_version in (1, 2):
        for pbc in [True, False]:
            atoms_copy = Atoms(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions().copy(),
                cell=(
                    atoms.get_cell().copy()
                    if atoms.cell is not None
                    and np.linalg.det(atoms.get_cell().copy()) != 0
                    else np.eye(3) * 5.0
                ),
                pbc=atoms.get_pbc().copy(),
            )
            if pbc:
                atoms_copy.set_pbc([True, True, True])
            else:
                atoms_copy.set_pbc([False, False, False])
            for unwrap in [True, False]:
                if unwrap and pbc:
                    atoms_copy.positions = -atoms_copy.positions

                # Use pymatgen graph generation
                data = AtomicData.from_ase(
                    atoms_copy,
                    max_neigh=max_neigh,
                    radius=radius,
                    r_edges=True,
                    r_data_keys=["spin", "charge"],
                )

                # Use FairChem internal graph generation (from core/graph/compute.py)
                batch = data_list_collater([data])
                graph_dict = generate_graph(
                    batch,
                    cutoff=radius,
                    max_neighbors=max_neigh,
                    enforce_max_neighbors_strictly=False,
                    radius_pbc_version=radius_pbc_version,
                    pbc=data.pbc,
                )

                assert check_features_match(
                    batch.edge_index,
                    batch.cell_offsets,
                    graph_dict["edge_index"],
                    graph_dict["cell_offsets"],
                )


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "num_atoms, num_partitions, pbc, device",
    [
        (5, 2, False, "cpu"),
        (20, 4, True, "cpu"),
        (30, 3, True, "cpu"),
        (101, 8, True, "cpu"),
        (101, 2, False, "cuda"),
        (105, 2, True, "cuda"),
    ],
)
def test_partitioned_radius_graph_pbc(
    num_atoms: int, num_partitions: int, pbc: bool, device: str
):
    radius = 6
    max_neighbors = 300
    pbc_tensor = torch.BoolTensor([pbc, pbc, pbc]).to(device)
    rgbv2 = partial(
        radius_graph_pbc_v2,
        radius=radius,
        max_num_neighbors_threshold=max_neighbors,
        pbc=pbc_tensor,
    )
    atoms = get_fcc_crystal_by_num_atoms(num_atoms)
    data = AtomicData.from_ase(atoms).to(device)
    batch = data_list_collater([data])
    edge_index_no_partition, _, _ = rgbv2(batch)
    edge_index_list = []
    for i in range(num_partitions):
        batch["node_partition"] = torch.tensor_split(
            torch.arange(num_atoms, device=device), num_partitions
        )[i]
        edge_index_part, _, _ = rgbv2(batch)
        edge_index_list.append(edge_index_part)

    # Verify that combined partitioned edges match non-partitioned edges
    combined_edges = torch.cat(edge_index_list, dim=1)

    # Convert edge pairs to sets for comparison (order doesn't matter)
    no_partition_pairs = {tuple(edge.tolist()) for edge in edge_index_no_partition.T}
    combined_pairs = {tuple(edge.tolist()) for edge in combined_edges.T}
    assert (
        no_partition_pairs == combined_pairs
    ), "Partitioned edges don't match non-partitioned edges"


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "num_systems, num_partitions, pbc, device",
    [
        (2, 2, True, "cpu"),
    ],
)
def test_generate_graph_h2o_partition(
    num_systems: int, num_partitions: int, pbc: bool, device: str
):
    radius = 6
    max_neighbors = 300
    # Create H2O molecule
    # Create H2O and O molecules batch
    h2o = molecule("H2O")
    h2o.info.update({"charge": 0, "spin": 1})
    h2o.pbc = True

    o_atom = molecule("O")
    o_atom.info.update({"charge": 0, "spin": 2})  # triplet oxygen
    o_atom.pbc = True

    h2o_data = AtomicData.from_ase(
        h2o,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    o_data = AtomicData.from_ase(
        o_atom,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch1 = atomicdata_list_to_batch([h2o_data, o_data]).to(device)

    generate_graph_partial = partial(
        generate_graph,
        cutoff=radius,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
        pbc=batch1.pbc,
    )
    no_partition_graph_data = generate_graph_partial(batch1)
    edge_index_no_partition = no_partition_graph_data["edge_index"]

    edge_index_list = []
    for i in range(num_partitions):
        batch1["node_partition"] = torch.tensor_split(
            torch.arange(len(batch1.atomic_numbers), device=device), num_partitions
        )[i]
        graph_dict = generate_graph_partial(batch1)
        edge_index_list.append(graph_dict["edge_index"])
    combined_edges = torch.cat(edge_index_list, dim=1)
    assert combined_edges.shape[1] == edge_index_no_partition.shape[1]

    # Convert edge pairs to sets for comparison (order doesn't matter)
    no_partition_pairs = {tuple(edge.tolist()) for edge in edge_index_no_partition.T}
    combined_pairs = {tuple(edge.tolist()) for edge in combined_edges.T}
    assert (
        no_partition_pairs == combined_pairs
    ), "Partitioned edges don't match non-partitioned edges"


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "num_atoms, num_systems, num_partitions, radius, max_neighbors, device",
    [
        (10, 2, 1, 6, 300, "cpu"),
        (10, 2, 2, 6, 300, "cpu"),
        (10, 4, 4, 6, 300, "cpu"),
        (10, 4, 4, 6, 30, "cpu"),
        (10, 4, 4, 5, 20, "cpu"),
        (34, 2, 2, 6, 1, "cpu"),
        (100, 7, 3, 6, 300, "cpu"),
        (100, 7, 1, 6, 300, "cuda"),
        (100, 7, 2, 6, 300, "cuda"),
    ],
)
def test_generate_graph_batch_partition(
    num_atoms: int,
    num_systems: int,
    num_partitions: int,
    radius: float,
    max_neighbors: int,
    device: str,
):
    # Convert to AtomicData
    data_list = []
    for i in range(num_systems):
        # pick a random lattice constant, this ensures that we have mixed cells in the batch too
        lattice_constant = np.random.uniform(3.7, 3.9)
        # add i to num_atoms to ensure different sizes
        atoms = get_fcc_crystal_by_num_atoms(
            num_atoms + i, lattice_constant=lattice_constant
        )
        data_list.append(
            AtomicData.from_ase(
                atoms,
                task_name="omol",
                r_data_keys=["spin", "charge"],
            ).to(device)
        )
    batch = atomicdata_list_to_batch(data_list)

    generate_graph_partial = partial(
        generate_graph,
        cutoff=radius,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
        pbc=batch.pbc,
    )
    no_partition_graph_data = generate_graph_partial(batch)
    edge_index_no_partition = no_partition_graph_data["edge_index"]
    edge_distance_no_partition = no_partition_graph_data["edge_distance"]
    edge_distance_vec_no_partition = no_partition_graph_data["edge_distance_vec"]

    edge_index_list = []
    edge_distance_list = []
    edge_distance_vecs = []
    for i in range(num_partitions):
        batch["node_partition"] = torch.tensor_split(
            torch.arange(len(batch.atomic_numbers), device=device), num_partitions
        )[i]
        graph_dict = generate_graph_partial(batch)
        edge_index_list.append(graph_dict["edge_index"])
        edge_distance_list.append(graph_dict["edge_distance"])
        edge_distance_vecs.append(graph_dict["edge_distance_vec"])

    combined_edges = torch.cat(edge_index_list, dim=1)
    combined_distances = torch.cat(edge_distance_list, dim=0)
    combined_distance_vecs = torch.cat(edge_distance_vecs, dim=0)
    assert combined_edges.shape[1] == edge_index_no_partition.shape[1]
    assert combined_distances.shape[0] == edge_distance_no_partition.shape[0]
    assert combined_distance_vecs.shape[0] == edge_distance_vec_no_partition.shape[0]

    # Convert edge pairs to sets for comparison (order doesn't matter)
    no_partition_pairs = {tuple(edge.tolist()) for edge in edge_index_no_partition.T}
    combined_pairs = {tuple(edge.tolist()) for edge in combined_edges.T}
    assert (
        no_partition_pairs == combined_pairs
    ), "Partitioned edges don't match non-partitioned edges"

    assert torch.allclose(combined_distances, edge_distance_no_partition)
    assert torch.allclose(combined_distance_vecs, edge_distance_vec_no_partition)
