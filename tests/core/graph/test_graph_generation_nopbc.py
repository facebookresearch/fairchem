"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for graph generation methods on non-PBC (non-periodic boundary condition) systems.
Validates both pymatgen-based and internal graph generation produce correct graphs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph


def check_features_match(
    edge_index_1: torch.Tensor,
    cell_offsets_1: torch.Tensor,
    edge_index_2: torch.Tensor,
    cell_offsets_2: torch.Tensor,
) -> bool:
    """Compare two graphs by converting edges to sets (order-independent).

    Combines edge indices and cell offsets into unified feature tensors,
    then compares as sets to handle edge ordering differences.
    """
    features_1 = torch.cat((edge_index_1, cell_offsets_1.T), dim=0).T
    features_2 = torch.cat((edge_index_2, cell_offsets_2.T), dim=0).T.long()

    features_1_set = {tuple(x.tolist()) for x in features_1}
    features_2_set = {tuple(x.tolist()) for x in features_2}

    # Handle empty edge case
    if len(features_1_set) == 0 and len(features_2_set) == 0:
        return True

    assert features_1_set == features_2_set, (
        f"Edge sets do not match.\n"
        f"In graph 1 but not graph 2: {features_1_set - features_2_set}\n"
        f"In graph 2 but not graph 1: {features_2_set - features_1_set}"
    )
    return True


def get_edge_pairs(edge_index: torch.Tensor) -> set:
    """Convert edge_index to set of (source, target) tuples."""
    return {tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])}


def setup_molecule(atoms: Atoms) -> Atoms:
    """Set a valid cell for pymatgen compatibility (required for neighbor search)."""
    atoms.cell = np.eye(3) * 20.0
    return atoms


class TestNonPBCMolecules:
    """Test graph generation for non-periodic molecules with known geometry."""

    def test_water_connectivity(self):
        """Test water: O-H ~0.97 Å, H-H ~1.53 Å. All 3 atoms connected at cutoff=2.0."""
        water = setup_molecule(molecule("H2O"))
        data = AtomicData.from_ase(water, r_edges=True, radius=2.0, max_neigh=10)

        assert data.edge_index.shape[1] == 6
        edge_pairs = get_edge_pairs(data.edge_index)

        # All pairs connected (bidirectional)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert (i, j) in edge_pairs
            assert (j, i) in edge_pairs

        # No self-loops, zero cell offsets for non-PBC
        for i in range(3):
            assert (i, i) not in edge_pairs
        assert torch.all(data.cell_offsets == 0)

    def test_benzene_ring_structure(self):
        """Test benzene ring: 6 C + 6 H atoms. At cutoff=1.6, only C-C and C-H bonds."""
        benzene = setup_molecule(molecule("C6H6"))
        data = AtomicData.from_ase(benzene, r_edges=True, radius=1.6, max_neigh=20)
        edge_pairs = get_edge_pairs(data.edge_index)

        # 6 C-C ring edges + 6 C-H edges = 12 bond pairs = 24 directed edges
        assert data.edge_index.shape[1] == 24

        # Verify ring connectivity: C0-C1-C2-C3-C4-C5-C0
        for i in range(6):
            j = (i + 1) % 6
            assert (i, j) in edge_pairs

        # Each H (indices 6-11) has exactly 1 neighbor
        for h_idx in range(6, 12):
            assert sum(1 for src, dst in edge_pairs if src == h_idx) == 1

    def test_nitrogen_diatomic(self):
        """Test simplest case: N2 with N-N bond ~1.1 Å."""
        nitrogen = setup_molecule(molecule("N2"))
        data = AtomicData.from_ase(nitrogen, r_edges=True, radius=2.0, max_neigh=10)

        assert data.edge_index.shape[1] == 2
        edge_pairs = get_edge_pairs(data.edge_index)
        assert edge_pairs == {(0, 1), (1, 0)}

    def test_ammonia_connectivity(self):
        """Test ammonia: N at center with 3 H atoms. All connected at cutoff=2.0."""
        ammonia = setup_molecule(molecule("NH3"))
        data = AtomicData.from_ase(ammonia, r_edges=True, radius=2.0, max_neigh=10)

        # All 4 atoms within 2.0 of each other = 4*3 = 12 edges
        assert data.edge_index.shape[1] == 12
        edge_pairs = get_edge_pairs(data.edge_index)

        # N connects to all 3 H atoms, H atoms connect to each other
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            assert (i, j) in edge_pairs

    def test_no_edges_small_cutoff(self):
        """Test that small cutoff produces no edges."""
        water = setup_molecule(molecule("H2O"))
        data = AtomicData.from_ase(water, r_edges=True, radius=0.5, max_neigh=10)
        assert data.edge_index.shape[1] == 0

    def test_single_atom_no_edges(self):
        """Single atom should have no edges."""
        atoms = Atoms("O", positions=[(0, 0, 0)])
        atoms = setup_molecule(atoms)

        data = AtomicData.from_ase(atoms, r_edges=True, radius=6.0, max_neigh=10)
        assert data.edge_index.shape[1] == 0, "Single atom should have no edges"


class TestGraphMethodsConsistency:
    """Test that pymatgen and internal graph generation produce consistent results."""

    @pytest.fixture()
    def test_molecules(self):
        molecules = {
            "water": molecule("H2O"),
            "benzene": molecule("C6H6"),
            "ethane": molecule("C2H6"),
            "ammonia": molecule("NH3"),
        }
        for mol in molecules.values():
            setup_molecule(mol)
        return molecules

    @pytest.mark.parametrize("mol_name", ["water", "benzene", "ethane", "ammonia"])
    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_pymatgen_vs_internal(self, test_molecules, mol_name, radius_pbc_version):
        """Verify pymatgen and internal graph generation produce identical graphs."""
        atoms = test_molecules[mol_name]
        cutoff = 6.0
        max_neigh = 100

        # Method 1: Pymatgen (via AtomicData.from_ase with r_edges=True)
        data_pymatgen = AtomicData.from_ase(
            atoms, r_edges=True, radius=cutoff, max_neigh=max_neigh
        )

        # Method 2: Internal (via generate_graph)
        data_no_edges = AtomicData.from_ase(atoms, r_edges=False)
        batch = data_list_collater([data_no_edges])

        graph_dict = generate_graph(
            batch,
            cutoff=cutoff,
            max_neighbors=max_neigh,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert check_features_match(
            data_pymatgen.edge_index,
            data_pymatgen.cell_offsets,
            graph_dict["edge_index"],
            graph_dict["cell_offsets"],
        )


class TestMaxNeighborsEnforcement:
    """Test max_neighbors parameter behavior."""

    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_max_neighbors_limits_per_atom(
        self, radius_pbc_version, torch_deterministic
    ):
        """Test that max_neighbors limits incoming edges per atom."""
        benzene = setup_molecule(molecule("C6H6"))
        data = AtomicData.from_ase(benzene, r_edges=False)
        batch = data_list_collater([data])

        # Each C in benzene normally has 3 neighbors at cutoff=1.6
        graph_dict = generate_graph(
            batch,
            cutoff=1.6,
            max_neighbors=2,
            enforce_max_neighbors_strictly=True,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        edge_index = graph_dict["edge_index"]
        for atom_idx in range(12):
            num_neighbors = (edge_index[1] == atom_idx).sum().item()
            assert num_neighbors <= 2, f"Atom {atom_idx} has {num_neighbors} neighbors"

    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_max_neighbors_selects_closest(
        self, radius_pbc_version, torch_deterministic
    ):
        """Test that max_neighbors keeps the closest neighbors."""
        # Linear chain with varied spacing: A-B close, C-D far
        atoms = Atoms(
            "CCCC", positions=[(0, 0, 0), (1.0, 0, 0), (2.5, 0, 0), (5.0, 0, 0)]
        )
        atoms = setup_molecule(atoms)

        data = AtomicData.from_ase(atoms, r_edges=False)
        batch = data_list_collater([data])

        graph_dict = generate_graph(
            batch,
            cutoff=6.0,
            max_neighbors=1,
            enforce_max_neighbors_strictly=True,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        edge_index = graph_dict["edge_index"]
        edge_pairs = get_edge_pairs(edge_index)

        # Each atom should have at most 1 neighbor
        for atom_idx in range(4):
            assert (edge_index[1] == atom_idx).sum().item() <= 1

        # Verify closest neighbors are selected
        # A's closest is B, D's closest is C
        if (0, 1) in edge_pairs or (1, 0) in edge_pairs:
            a_neighbors = [dst for src, dst in edge_pairs if src == 0]
            if a_neighbors:
                assert 1 in a_neighbors
        d_neighbors = [dst for src, dst in edge_pairs if src == 3]
        if d_neighbors:
            assert 2 in d_neighbors

    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_enforce_max_neighbors_strictly_with_ties(
        self, radius_pbc_version, torch_deterministic
    ):
        """Test enforce_max_neighbors_strictly with equidistant neighbors.

        Equilateral triangle: all pairwise distances are equal.
        With max_neighbors=1:
        - strictly=True: each atom gets exactly 1 neighbor (3 edges total)
        - strictly=False: tied neighbors are kept (6 edges total, fully connected)
        """
        # Equilateral triangle with side length 1.0
        atoms = Atoms("CCC", positions=[(0, 0, 0), (1.0, 0, 0), (0.5, 0.866, 0)])
        atoms = setup_molecule(atoms)
        data = AtomicData.from_ase(atoms, r_edges=False)
        batch = data_list_collater([data])

        graph_strict = generate_graph(
            batch,
            cutoff=2.0,
            max_neighbors=1,
            enforce_max_neighbors_strictly=True,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )
        graph_loose = generate_graph(
            batch,
            cutoff=2.0,
            max_neighbors=1,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        # Strict: exactly 1 neighbor per atom
        assert graph_strict["edge_index"].shape[1] == 3
        for i in range(3):
            assert (graph_strict["edge_index"][1] == i).sum().item() == 1

        # Loose: keeps tied neighbors, fully connected
        assert graph_loose["edge_index"].shape[1] == 6
        for i in range(3):
            assert (graph_loose["edge_index"][1] == i).sum().item() == 2


@pytest.mark.parametrize(
    "atoms,expected_num_edges,cutoff",
    [
        (molecule("H2O"), 6, 2.0),
        (molecule("C6H6"), 24, 1.6),
        (molecule("N2"), 2, 2.0),
        (molecule("NH3"), 12, 2.0),
    ],
    ids=["water", "benzene", "nitrogen", "ammonia"],
)
def test_edge_count_parametrized(atoms, expected_num_edges, cutoff):
    """Parametrized test for expected edge counts with various molecules."""
    atoms = setup_molecule(atoms)

    data = AtomicData.from_ase(atoms, r_edges=True, radius=cutoff, max_neigh=100)
    assert data.edge_index.shape[1] == expected_num_edges


class TestSeparatedSystems:
    """Test graph generation for multiple separated molecular systems."""

    @pytest.fixture()
    def two_waters_separated(self):
        """Two water molecules with diagonal separation for varied cross-distances."""
        water1 = molecule("H2O")
        water2 = molecule("H2O")
        water2.translate([3.0, 1.5, 0.0])  # Diagonal offset
        return setup_molecule(water1 + water2)

    def test_two_waters_cross_edge_verification(self, two_waters_separated):
        """Verify cross-molecule edges appear/disappear based on cutoff."""
        edge_pairs_small = get_edge_pairs(
            AtomicData.from_ase(
                two_waters_separated, r_edges=True, radius=2.0, max_neigh=100
            ).edge_index
        )
        edge_pairs_large = get_edge_pairs(
            AtomicData.from_ase(
                two_waters_separated, r_edges=True, radius=10.0, max_neigh=100
            ).edge_index
        )

        # At small cutoff: no cross-molecule edges
        water1_atoms, water2_atoms = {0, 1, 2}, {3, 4, 5}
        for src, dst in edge_pairs_small:
            in_same = (src in water1_atoms and dst in water1_atoms) or (
                src in water2_atoms and dst in water2_atoms
            )
            assert in_same, f"Unexpected cross-edge ({src}, {dst}) at cutoff=2.0"

        # At large cutoff: all cross-molecule edges exist
        for i in range(3):
            for j in range(3, 6):
                assert (i, j) in edge_pairs_large

    @pytest.mark.parametrize(
        "cutoff,expected_edges",
        [
            (2.0, 12),  # Only intra-molecule
            (3.5, 24),  # 12 intra + 12 cross
            (4.0, 28),  # 12 intra + 16 cross
            (4.5, 30),  # Fully connected
        ],
        ids=["intra-only", "some-cross", "more-cross", "fully-connected"],
    )
    def test_cutoff_progression(self, two_waters_separated, cutoff, expected_edges):
        """Test that edge count matches expected value at each cutoff."""
        data = AtomicData.from_ase(
            two_waters_separated, r_edges=True, radius=cutoff, max_neigh=100
        )
        assert data.edge_index.shape[1] == expected_edges

    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_two_waters_pymatgen_vs_internal(
        self, two_waters_separated, radius_pbc_version
    ):
        """Verify pymatgen and internal methods match for multi-molecule systems."""
        data_pymatgen = AtomicData.from_ase(
            two_waters_separated, r_edges=True, radius=3.5, max_neigh=100
        )
        data_no_edges = AtomicData.from_ase(two_waters_separated, r_edges=False)
        batch = data_list_collater([data_no_edges])

        graph_dict = generate_graph(
            batch,
            cutoff=3.5,
            max_neighbors=100,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert check_features_match(
            data_pymatgen.edge_index,
            data_pymatgen.cell_offsets,
            graph_dict["edge_index"],
            graph_dict["cell_offsets"],
        )


class TestComplexGeometry:
    """Test graph generation for molecules with complex geometry."""

    @pytest.fixture()
    def ethyl_methyl_ether(self):
        """Ethyl methyl ether (CH3CH2OCH3) - 12 atoms with C-C, C-H, C-O bonds."""
        return setup_molecule(molecule("CH3CH2OCH3"))

    def test_ethyl_methyl_ether_structure(self, ethyl_methyl_ether):
        """Test ethyl methyl ether: verify basic graph properties."""
        data = AtomicData.from_ase(
            ethyl_methyl_ether, r_edges=True, radius=1.8, max_neigh=50
        )
        edge_pairs = get_edge_pairs(data.edge_index)

        # 12 atoms with C-C, C-O, C-H bonds at cutoff=1.8: 18 bond pairs = 36 edges
        assert data.edge_index.shape[1] == 36

        # No self-loops, all edges bidirectional
        for i in range(len(ethyl_methyl_ether)):
            assert (i, i) not in edge_pairs
        for src, dst in edge_pairs:
            assert (dst, src) in edge_pairs

    @pytest.mark.parametrize("radius_pbc_version", [1, 2])
    def test_ethyl_methyl_ether_pymatgen_vs_internal(
        self, ethyl_methyl_ether, radius_pbc_version
    ):
        """Verify pymatgen and internal methods match for complex molecules."""
        data_pymatgen = AtomicData.from_ase(
            ethyl_methyl_ether, r_edges=True, radius=3.0, max_neigh=100
        )
        data_no_edges = AtomicData.from_ase(ethyl_methyl_ether, r_edges=False)
        batch = data_list_collater([data_no_edges])

        graph_dict = generate_graph(
            batch,
            cutoff=3.0,
            max_neighbors=100,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=radius_pbc_version,
            pbc=torch.BoolTensor([False, False, False]),
        )

        assert check_features_match(
            data_pymatgen.edge_index,
            data_pymatgen.cell_offsets,
            graph_dict["edge_index"],
            graph_dict["cell_offsets"],
        )
