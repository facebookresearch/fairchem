"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  connectivity-preserving neighbour selection — per-atom nearest-k
        truncation can split a physically connected structure into
        disconnected components, and `preserve_connectivity` must undo that
        without changing anything else. Covers the component labeller, the
        no-op fast path, the component-count guarantee, the merge-round
        bounds, bidirectionality of a restored bridge, and the invariants of
        the reserved edge set: rigid motion, uniform scaling, atom
        relabelling, and run-to-run reproducibility.
Models: none. Pure graph statistics, no checkpoint. One `gpu`-marked
        reproducibility test; everything else runs on CPU.
CI:     test (core shard).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from ase import Atoms, build

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
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


def grain_chain(gaps: float | Sequence[float], blocks: int, repeat: int = 2) -> Atoms:
    """
    `blocks` dense fcc blocks in a row along x, each bridged to the next across
    `gaps[k]`. Truncation splits this into `blocks` components inside a single
    system.

    Equal gaps merge in one Boruvka round, because on a path with tied weights
    every edge is minimal at some endpoint. Unequal gaps do not: an edge that is
    minimal at neither endpoint waits for a later round, which is the only way
    the round bound is exercised at all.
    """
    if not isinstance(gaps, Sequence):
        gaps = [float(gaps)] * (blocks - 1)
    assert len(gaps) == blocks - 1
    grain = build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((repeat,) * 3)
    width = grain.cell[0, 0]
    out = grain.copy()
    offset = 0.0
    for gap in gaps:
        offset += width + gap
        nxt = grain.copy()
        nxt.positions[:, 0] += offset
        out = out + nxt
    out.set_cell([offset + width + 14.0, grain.cell[1, 1], grain.cell[2, 2]])
    out.pbc = [False, True, True]
    return out


# Two gap orderings, because they stress different bounds. Staggered (weights
# 1,4,2,5,3) leaves three components after the first round, so it needs more
# rounds than one. Monotone makes every round-one hook point at the block to its
# left, giving the deepest possible parent chain, so it needs the pointer jumps.
STAGGERED_GAPS = [3.0, 3.6, 3.2, 3.8, 3.4]
MONOTONE_GAPS = [3.0, 3.2, 3.4, 3.6, 3.8]


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


def full_edge_list(atoms: Atoms, cutoff: float = CUTOFF):
    """
    Untruncated radius graph plus the squared distances, as
    `get_max_neighbors_mask` receives them.
    """
    data = AtomicData.from_ase(atoms)
    graph = generate_graph(
        data,
        cutoff=cutoff,
        max_neighbors=0,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=1,
        pbc=None,
    )
    source, target = graph["edge_index"][0], graph["edge_index"][1]
    distance_sq = graph["edge_distance"] ** 2
    return data, source, target, distance_sq


def reserved_pairs(atoms: Atoms, max_neighbors: int = 30, cutoff: float = CUTOFF):
    """The undirected atom pairs `reconnect_mask` reserves, as a set."""
    data, source, target, distance_sq = full_edge_list(atoms, cutoff)
    kept, _ = get_max_neighbors_mask(
        natoms=data.natoms,
        index=target,
        atom_distance=distance_sq,
        max_num_neighbors_threshold=max_neighbors,
    )
    extra = reconnect_mask(
        target, source, distance_sq, kept, int(data.natoms.sum()), natoms=data.natoms
    )
    return {
        (min(a, b), max(a, b))
        for a, b in zip(source[extra].tolist(), target[extra].tolist())
    }


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

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reserved_set_is_invariant_under_rigid_motion(self, seed):
        """
        Rotating and translating the structure preserves every interatomic
        distance, so it must reserve exactly the same pairs. A selection rule
        that read a coordinate rather than a distance would fail this.
        """
        atoms = two_grain_contact(gap=3.2)
        rng = np.random.default_rng(seed)
        moved = atoms.copy()
        moved.rotate(float(rng.uniform(0, 360)), rng.normal(size=3), rotate_cell=True)
        moved.positions += rng.uniform(-5, 5, size=3)
        assert reserved_pairs(moved) == reserved_pairs(atoms)

    @pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
    def test_reserved_set_is_equivariant_under_uniform_scaling(self, scale):
        """
        Scaling the positions and the cutoff together rescales every distance by
        the same factor and reorders nothing, so the reserved pairs must be
        identical. This is what catches an absolute length or an absolute
        squared-distance threshold used where a relative one belongs.
        """
        atoms = two_grain_contact(gap=3.2)
        scaled = atoms.copy()
        scaled.set_cell(atoms.cell * scale, scale_atoms=True)
        assert reserved_pairs(scaled, cutoff=CUTOFF * scale) == reserved_pairs(atoms)

    @pytest.mark.parametrize("max_neighbors", [8, 12, 20, 30])
    def test_the_flag_only_ever_adds_edges(self, max_neighbors):
        """
        The correction must be strictly an improvement: it may add edges the
        budget dropped, never remove or replace one the budget kept, and it must
        land on exactly the untruncated component count rather than merely no
        worse than it.
        """
        atoms = two_grain_contact(gap=3.2)
        num_atoms = len(atoms)
        off = graph_for(atoms, max_neighbors, preserve=False)
        on = graph_for(atoms, max_neighbors, preserve=True)
        as_pairs = lambda ei: set(zip(ei[0].tolist(), ei[1].tolist()))  # noqa: E731
        assert as_pairs(off["edge_index"]) <= as_pairs(on["edge_index"])
        assert component_count(on["edge_index"], num_atoms) == component_count(
            graph_for(atoms, 0, preserve=False)["edge_index"], num_atoms
        )

    @pytest.mark.gpu()
    def test_reserved_set_is_bitwise_reproducible_on_gpu(self):
        """
        `scatter_reduce_` is where a nondeterministic atomic would leak in, and
        it only does so on device. The CPU twin below cannot see that.
        """
        items = [grain_chain(MONOTONE_GAPS, 6), grain_chain(3.2, 2)] + [
            build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
        ] * 4
        for atoms in items:
            atoms.pbc = [False, True, True]
        data = atomicdata_list_to_batch([AtomicData.from_ase(a) for a in items]).to(
            "cuda"
        )
        kwargs = {
            "cutoff": CUTOFF,
            "max_neighbors": 30,
            "enforce_max_neighbors_strictly": False,
            "radius_pbc_version": 1,
            "pbc": None,
            "preserve_connectivity": True,
        }
        first = generate_graph(data, **kwargs)["edge_index"]
        for _ in range(5):
            assert torch.equal(generate_graph(data, **kwargs)["edge_index"], first)

    def test_reserved_set_is_bitwise_reproducible(self):
        """
        The reserved edges feed the message-passing graph, so any run-to-run
        variation would show up as non-reproducible energies. Boruvka's scatter
        reductions are the place a nondeterministic atomic would leak in.
        """
        atoms = two_grain_contact(gap=3.2)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        kept, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
        )
        args = (target, source, distance_sq, kept, num_atoms)
        first = reconnect_mask(*args, natoms=data.natoms)
        assert bool(first.any())
        for _ in range(5):
            assert torch.equal(reconnect_mask(*args, natoms=data.natoms), first)

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


class TestWorkBoundsDoNotChangeTheAnswer:
    """
    `reconnect_mask` bounds its merge rounds from the structure of the batch
    rather than testing for convergence, because each test costs a
    device-to-host synchronization. If either bound were a heuristic, a chain of
    blocks deep enough to need more rounds than the bound allows would come back
    still fractured.
    """

    @pytest.mark.parametrize("gaps", [STAGGERED_GAPS, MONOTONE_GAPS])
    @pytest.mark.parametrize("blocks", [2, 3, 4, 5, 6])
    def test_deep_chain_is_fully_reconnected(self, blocks, gaps):
        """
        Unequal gaps, so the chain genuinely needs the bounds. With equal gaps
        every edge ties for minimal at some endpoint, one round suffices and the
        parent forest is one level deep, which exercises neither.
        """
        atoms = grain_chain(gaps[: blocks - 1], blocks)
        num_atoms = len(atoms)
        assert component_count(graph_for(atoms, 30, False)["edge_index"], num_atoms) > 1
        assert component_count(graph_for(atoms, 30, True)["edge_index"], num_atoms) == 1

    def test_deep_system_beside_shallow_ones_is_reconnected(self):
        """The rounds are bounded by the deepest system, not by the batch."""
        items = [grain_chain(STAGGERED_GAPS, 6)] + [
            build.bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
        ] * 4
        for atoms in items:
            atoms.pbc = [False, True, True]
        data = atomicdata_list_to_batch([AtomicData.from_ase(a) for a in items])
        graph = generate_graph(
            data,
            cutoff=CUTOFF,
            max_neighbors=30,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=1,
            pbc=None,
            preserve_connectivity=True,
        )
        assert component_count(graph["edge_index"], int(data.natoms.sum())) == len(
            items
        )

    def test_batched_and_single_system_calls_agree(self):
        """
        `natoms` only bounds work, so passing it must not change the result
        relative to treating the input as one undivided system.
        """
        atoms = grain_chain(3.2, 3)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        kept, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
        )
        common = (target, source, distance_sq, kept, num_atoms, 0.01)
        assert torch.equal(
            reconnect_mask(*common, natoms=data.natoms),
            reconnect_mask(*common, natoms=None),
        )


class TestOtherBranchesOfTheMask:
    """
    Paths through `get_max_neighbors_mask` and the batch that the two-grain
    case does not reach.
    """

    def test_strict_truncation_is_repaired_too(self):
        """
        `enforce_max_strictly=True` takes a different branch to build the mask
        and falls through to the same repair. Strict truncation drops more, so
        it fractures at least as readily.
        """
        atoms = two_grain_contact(gap=3.2)
        data, source, target, distance_sq = full_edge_list(atoms)
        num_atoms = int(data.natoms.sum())
        kept, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
            enforce_max_strictly=True,
        )
        assert component_count(torch.stack([source[kept], target[kept]]), num_atoms) > 1
        repaired, _ = get_max_neighbors_mask(
            natoms=data.natoms,
            index=target,
            atom_distance=distance_sq,
            max_num_neighbors_threshold=30,
            enforce_max_strictly=True,
            neighbor_index=source,
            preserve_connectivity=True,
        )
        assert (
            component_count(
                torch.stack([source[repaired], target[repaired]]), num_atoms
            )
            == 1
        )

    def test_mixed_pbc_batch(self):
        """
        Only v2 handles a batch mixing periodic and non-periodic systems, and it
        is the version whose `neighbor_index` is conditional.
        """
        periodic = two_grain_contact(gap=3.2)
        periodic.pbc = [True, True, True]
        molecule = two_grain_contact(gap=3.2)
        molecule.pbc = [False, False, False]
        items = [periodic, molecule]
        data = atomicdata_list_to_batch([AtomicData.from_ase(a) for a in items])
        graphs = {
            flag: generate_graph(
                data,
                cutoff=CUTOFF,
                max_neighbors=30,
                enforce_max_neighbors_strictly=False,
                radius_pbc_version=2,
                pbc=None,
                preserve_connectivity=flag,
            )
            for flag in (False, True)
        }
        n = int(data.natoms.sum())
        assert component_count(graphs[False]["edge_index"], n) > len(items)
        assert component_count(graphs[True]["edge_index"], n) == len(items)

    def test_single_atom_system_in_the_batch(self):
        """
        A lone atom is its own component, so it raises the component floor
        without ever contributing a bridge. The per-system search over sorted
        component roots has to place it correctly.
        """
        lone = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0])
        lone.pbc = [False, False, False]
        items = [lone, two_grain_contact(gap=3.2)]
        data = atomicdata_list_to_batch([AtomicData.from_ase(a) for a in items])
        graph = generate_graph(
            data,
            cutoff=CUTOFF,
            max_neighbors=30,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=2,
            pbc=None,
            preserve_connectivity=True,
        )
        assert component_count(graph["edge_index"], int(data.natoms.sum())) == 2


class TestBackboneWiring:
    """
    The flag is only useful if a model can set it. Without this the plumbing
    stops at `generate_graph` and nothing downstream can reach it.
    """

    @staticmethod
    def _backbone(preserve: bool):
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone

        return eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
            lmax=2,
            mmax=2,
            num_layers=1,
            otf_graph=True,
            edge_channels=5,
            num_distance_basis=7,
            use_dataset_embedding=False,
            always_use_pbc=False,
            cutoff=CUTOFF,
            max_neighbors=30,
            radius_pbc_version=1,
            preserve_connectivity=preserve,
        )

    @pytest.mark.parametrize("preserve", [False, True])
    def test_backbone_forwards_the_flag_into_graph_generation(self, preserve):
        atoms = two_grain_contact(gap=3.2)
        data = atomicdata_list_to_batch([AtomicData.from_ase(atoms)])
        graph = self._backbone(preserve)._generate_graph(data)
        expected = 1 if preserve else 2
        assert component_count(graph["edge_index"], len(atoms)) == expected

    def test_default_is_off(self):
        assert self._backbone(False).preserve_connectivity is False


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
