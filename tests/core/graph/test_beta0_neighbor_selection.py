"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  connectivity-preserving neighbour selection — per-atom nearest-k
        truncation can split a physically connected structure into
        disconnected components, and `preserve_connectivity` must undo that
        without changing anything else. Covers the component labeller, the
        no-op fast path, the component-count guarantee, atom-order
        invariance of the reserved edges, and bidirectionality of a
        restored bridge.
Models: none. Pure graph statistics, no checkpoint, no GPU.
CI:     test (core shard).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms, build

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph
from fairchem.core.graph.radius_graph_pbc import (
    connected_component_labels,
    get_max_neighbors_mask,
    reconnect_mask,
)

CUTOFF = 6.0


def two_grain_contact(gap: float, repeat: int = 2) -> Atoms:
    """
    Two dense fcc blocks facing each other across a vacuum gap along x.

    Every atom on either face has far more than `max_neighbors` intra-grain
    neighbours closer than the cross-gap contact, so both endpoints of the
    contact rank it outside their own budget.
    """
    grain = build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((repeat,) * 3)
    width = grain.cell[0, 0]
    right = grain.copy()
    right.positions[:, 0] += width + gap
    both = grain + right
    both.set_cell([2 * width + gap + 14.0, grain.cell[1, 1], grain.cell[2, 2]])
    both.pbc = [False, True, True]
    return both


def component_count(edge_index: torch.Tensor, num_atoms: int) -> int:
    """
    Number of connected components, by union-find, independent of the code
    under test.
    """
    parent = list(range(num_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(num_atoms)})


def graph_for(atoms: Atoms, max_neighbors: int, preserve: bool, version: int = 1):
    return generate_graph(
        AtomicData.from_ase(atoms),
        cutoff=CUTOFF,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=version,
        pbc=None,
        preserve_connectivity=preserve,
    )


def full_edge_list(atoms: Atoms):
    """
    Untruncated radius graph plus the squared distances, as
    `get_max_neighbors_mask` receives them.
    """
    data = AtomicData.from_ase(atoms)
    graph = generate_graph(
        data,
        cutoff=CUTOFF,
        max_neighbors=0,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=1,
        pbc=None,
    )
    source, target = graph["edge_index"][0], graph["edge_index"][1]
    distance_sq = graph["edge_distance"] ** 2
    return data, source, target, distance_sq


class TestComponentLabels:
    def test_labels_match_union_find(self):
        edges = torch.tensor([[0, 1, 4, 6], [1, 2, 5, 7]])
        labels = connected_component_labels(edges[0], edges[1], 9)
        groups = {}
        for atom, label in enumerate(labels.tolist()):
            groups.setdefault(label, []).append(atom)
        assert sorted(sorted(g) for g in groups.values()) == [
            [0, 1, 2],
            [3],
            [4, 5],
            [6, 7],
            [8],
        ]

    def test_label_is_smallest_atom_index_in_component(self):
        edges = torch.tensor([[7, 5], [5, 2]])
        labels = connected_component_labels(edges[0], edges[1], 8)
        assert labels[2] == labels[5] == labels[7] == 2

    def test_empty_edge_list_gives_one_component_per_atom(self):
        empty = torch.zeros(0, dtype=torch.long)
        labels = connected_component_labels(empty, empty, 4)
        assert labels.unique().numel() == 4


class TestTruncationCanDisconnect:
    """
    The defect this change corrects. These assert the current behaviour so a
    future refactor cannot silently reintroduce it.
    """

    @pytest.mark.parametrize("max_neighbors", [8, 12, 20, 30])
    def test_bridged_structure_fractures_without_the_flag(self, max_neighbors):
        atoms = two_grain_contact(gap=3.2)
        num_atoms = len(atoms)
        full = graph_for(atoms, max_neighbors=0, preserve=False)
        truncated = graph_for(atoms, max_neighbors, preserve=False)
        assert component_count(full["edge_index"], num_atoms) == 1
        assert component_count(truncated["edge_index"], num_atoms) > 1

    @pytest.mark.parametrize("max_neighbors", [8, 12, 20, 30])
    def test_flag_restores_the_component_count(self, max_neighbors):
        atoms = two_grain_contact(gap=3.2)
        num_atoms = len(atoms)
        full = graph_for(atoms, max_neighbors=0, preserve=False)
        fixed = graph_for(atoms, max_neighbors, preserve=True)
        assert component_count(fixed["edge_index"], num_atoms) == component_count(
            full["edge_index"], num_atoms
        )

    @pytest.mark.parametrize("version", [1, 2])
    def test_every_radius_graph_version_is_fixed(self, version):
        atoms = two_grain_contact(gap=3.2)
        num_atoms = len(atoms)
        broken = graph_for(atoms, 30, preserve=False, version=version)
        fixed = graph_for(atoms, 30, preserve=True, version=version)
        assert component_count(broken["edge_index"], num_atoms) > 1
        assert component_count(fixed["edge_index"], num_atoms) == 1

    def test_already_disconnected_structure_is_not_glued_together(self):
        """Preserving connectivity must not invent bonds across a real vacuum."""
        atoms = two_grain_contact(gap=8.0)
        num_atoms = len(atoms)
        full = graph_for(atoms, max_neighbors=0, preserve=False)
        fixed = graph_for(atoms, 30, preserve=True)
        assert component_count(full["edge_index"], num_atoms) == 2
        assert component_count(fixed["edge_index"], num_atoms) == 2


class TestNoOpWhenNothingFractured:
    def test_dense_bulk_is_untouched(self):
        atoms = build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
        plain = graph_for(atoms, 30, preserve=False)
        preserved = graph_for(atoms, 30, preserve=True)
        assert torch.equal(plain["edge_index"], preserved["edge_index"])
        assert torch.equal(plain["neighbors"], preserved["neighbors"])

    def test_reconnect_mask_is_empty_when_budget_never_binds(self):
        atoms = build.bulk("Cu", "fcc", a=3.61, cubic=True)
        data, source, target, distance_sq = full_edge_list(atoms)
        kept = torch.ones_like(source, dtype=torch.bool)
        extra = reconnect_mask(
            target, source, distance_sq, kept, int(data.natoms.sum())
        )
        assert not bool(extra.any())

    def test_reserved_edges_are_disjoint_from_kept_edges(self):
        atoms = two_grain_contact(gap=3.2)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        kept, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
        )
        extra = reconnect_mask(target, source, distance_sq, kept, num_atoms)
        assert bool(extra.any())
        assert not bool((extra & kept).any())


class TestReservedEdgeProperties:
    def test_reserved_set_at_least_closes_the_component_gap(self):
        """A forest joining c components needs at least c - 1 edges."""
        atoms = two_grain_contact(gap=3.2)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        for max_neighbors in (8, 12, 20, 30):
            kept, _ = get_max_neighbors_mask(
                natoms=data.natoms,
                index=target,
                atom_distance=distance_sq,
                max_num_neighbors_threshold=max_neighbors,
            )
            extra = reconnect_mask(target, source, distance_sq, kept, num_atoms)
            gap = component_count(
                torch.stack([source[kept], target[kept]]), num_atoms
            ) - component_count(torch.stack([source, target]), num_atoms)
            undirected = {
                (min(a, b), max(a, b))
                for a, b in zip(source[extra].tolist(), target[extra].tolist())
            }
            assert len(undirected) >= gap

    def test_a_restored_bridge_carries_messages_both_ways(self):
        atoms = two_grain_contact(gap=3.2)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        kept, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
        )
        extra = reconnect_mask(target, source, distance_sq, kept, num_atoms)
        pairs = set(zip(source[extra].tolist(), target[extra].tolist()))
        assert pairs
        assert all((b, a) in pairs for a, b in pairs)

    def test_reserved_set_does_not_depend_on_atom_order(self):
        """
        Selection reads only distances, so relabelling the atoms must not change
        which physical edges are reserved. This is what stops the correction
        from making the arbitrary choice between degenerate edges that the
        `get_max_neighbors_mask` docstring warns about.
        """
        atoms = two_grain_contact(gap=3.2)
        rng = np.random.default_rng(0)
        signatures = set()
        for trial in range(3):
            order = np.arange(len(atoms)) if trial == 0 else rng.permutation(len(atoms))
            permuted = atoms[order]
            data, source, target, distance_sq = full_edge_list(permuted)
            num_atoms = int(data.natoms.sum())
            kept, _ = get_max_neighbors_mask(
                natoms=data.natoms,
                index=target,
                atom_distance=distance_sq,
                max_num_neighbors_threshold=30,
            )
            extra = reconnect_mask(target, source, distance_sq, kept, num_atoms)
            # map back to the original labelling: new index i holds old atom order[i]
            a = order[source[extra].numpy()]
            b = order[target[extra].numpy()]
            signatures.add(
                tuple(
                    sorted(
                        zip(
                            np.minimum(a, b).tolist(),
                            np.maximum(a, b).tolist(),
                            np.round(distance_sq[extra].numpy(), 6).tolist(),
                        )
                    )
                )
            )
        assert len(signatures) == 1


class TestNeighborCountsStayConsistent:
    def test_reported_neighbor_count_matches_the_edges_returned(self):
        atoms = two_grain_contact(gap=3.2)
        graph = graph_for(atoms, 30, preserve=True)
        assert int(graph["neighbors"].sum()) == graph["edge_index"].shape[1]

    def test_added_edges_are_a_small_fraction_of_the_budget(self):
        atoms = two_grain_contact(gap=3.2)
        plain = graph_for(atoms, 30, preserve=False)
        preserved = graph_for(atoms, 30, preserve=True)
        added = preserved["edge_index"].shape[1] - plain["edge_index"].shape[1]
        assert 0 < added <= 0.05 * plain["edge_index"].shape[1]
