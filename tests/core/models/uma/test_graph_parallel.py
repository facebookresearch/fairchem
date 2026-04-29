"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.common import gp_utils
from fairchem.core.common.gp_utils import (
    gather_from_model_parallel_region_sum_grad,
    size_list_fn,
)
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.models.uma.graph_parallel import (
    _compute_send_indices,
    all_to_all_collect,
    build_gp_context,
    finish_all_to_all_collect,
    partition_atoms_index_split,
    partition_atoms_spatial,
    remap_edge_index_to_local,
    start_all_to_all_collect,
)

pytestmark = pytest.mark.serial


# =========================================================================
# Unit tests (no distributed, CPU only)
# =========================================================================


class TestPartitionAtomsIndexSplit:
    """
    Tests for partition_atoms_index_split.
    """

    def test_single_rank(self):
        result = partition_atoms_index_split(5, 1, torch.device("cpu"))
        assert result.shape == (5,)
        assert (result == 0).all()

    def test_even_split(self):
        result = partition_atoms_index_split(6, 3, torch.device("cpu"))
        assert result.shape == (6,)
        # Atoms 0,1 -> rank 0; atoms 2,3 -> rank 1; atoms 4,5 -> rank 2
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 1
        assert result[4] == 2
        assert result[5] == 2

    def test_uneven_split(self):
        result = partition_atoms_index_split(5, 2, torch.device("cpu"))
        assert result.shape == (5,)
        # 5 atoms, 2 ranks: [0,1,2] -> rank 0, [3,4] -> rank 1
        assert (result[:3] == 0).all()
        assert (result[3:] == 1).all()

    def test_more_ranks_than_atoms(self):
        result = partition_atoms_index_split(2, 5, torch.device("cpu"))
        assert result.shape == (2,)
        # Each atom gets its own rank
        for i in range(2):
            assert result[i].item() >= 0
            assert result[i].item() < 5


class TestPartitionAtomsSpatial:
    """
    Tests for partition_atoms_spatial.
    """

    def test_single_rank(self):
        pos = torch.randn(10, 3)
        result = partition_atoms_spatial(pos, 1)
        assert result.shape == (10,)
        assert (result == 0).all()

    def test_balanced_output(self):
        pos = torch.randn(100, 3)
        result = partition_atoms_spatial(pos, 4)
        assert result.shape == (100,)
        # Check all ranks are assigned
        for r in range(4):
            count = (result == r).sum()
            assert count > 0, f"Rank {r} has no atoms"
        # Check balance: each should have ~25 atoms (±1)
        for r in range(4):
            count = (result == r).sum().item()
            assert 24 <= count <= 26, f"Rank {r} has {count} atoms, expected ~25"

    def test_spatially_separated_clusters(self):
        """
        Atoms in distinct spatial clusters should be assigned to different ranks.
        """
        pos = torch.cat(
            [
                torch.randn(20, 3) + torch.tensor([0.0, 0.0, 0.0]),
                torch.randn(20, 3) + torch.tensor([100.0, 0.0, 0.0]),
            ]
        )
        result = partition_atoms_spatial(pos, 2)
        # The two clusters should be mostly on different ranks
        rank_cluster_0 = result[:20].mode()[0].item()
        rank_cluster_1 = result[20:].mode()[0].item()
        assert rank_cluster_0 != rank_cluster_1

    def test_more_ranks_than_atoms(self):
        pos = torch.randn(3, 3)
        result = partition_atoms_spatial(pos, 5)
        assert result.shape == (3,)


class TestBuildGPContext:
    """
    Tests for build_gp_context (non-distributed, simulates single rank).
    """

    def test_basic_context_building(self):
        """
        Test with a simple graph where all atoms are on rank 0.
        """
        # 4 atoms, 2 ranks, rank 0 owns [0,1], rank 1 owns [2,3]
        edge_index = torch.tensor([[0, 1, 2, 3, 2], [1, 0, 3, 2, 0]])
        rank_assignments = torch.tensor([0, 0, 1, 1])

        # Build context for rank 0 (no distributed env)
        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)

        assert ctx.rank == 0
        assert ctx.world_size == 2
        assert ctx.total_local_atoms == 2
        assert torch.equal(ctx.node_partition, torch.tensor([0, 1]))

        # Rank 0 targets: atoms 0 and 1
        # Edge (2, 0): src=2 is remote, tgt=0 is local -> need atom 2
        # Edge (1, 0): src=1 is local -> don't need
        # Edge (0, 1): src=0 is local -> don't need
        assert 2 in ctx.needed_atoms

    def test_global_to_local_mapping(self):
        """
        Verify that global_to_local correctly maps to local indices.
        """
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        rank_assignments = torch.tensor([0, 0, 1, 1])

        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)

        # Local atoms [0, 1] should map to indices [0, 1]
        assert ctx.global_to_local[0] == 0
        assert ctx.global_to_local[1] == 1

        # Remote atoms that are needed should map to indices >= total_local_atoms
        for atom in ctx.needed_atoms:
            local_idx = ctx.global_to_local[atom].item()
            assert local_idx >= ctx.total_local_atoms

    def test_no_cross_partition_edges(self):
        """
        When no edges cross partitions, no remote atoms are needed.
        """
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Only within rank 0
        rank_assignments = torch.tensor([0, 0, 1, 1])

        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        assert ctx.total_needed_atoms == 0
        assert ctx.needed_atoms.numel() == 0


class TestRemapEdgeIndex:
    """
    Tests for remap_edge_index_to_local.
    """

    def test_identity_mapping(self):
        """
        If global_to_local is identity, edge_index stays the same.
        """
        edge_index = torch.tensor([[0, 1], [1, 2]])
        global_to_local = torch.arange(3)
        result = remap_edge_index_to_local(edge_index, global_to_local)
        assert torch.equal(result, edge_index)

    def test_offset_mapping(self):
        """
        Global_to_local with offset.
        """
        edge_index = torch.tensor([[2, 3], [3, 2]])
        global_to_local = torch.tensor([-1, -1, 0, 1])
        result = remap_edge_index_to_local(edge_index, global_to_local)
        expected = torch.tensor([[0, 1], [1, 0]])
        assert torch.equal(result, expected)


# =========================================================================
# Distributed tests: all-to-all vs all-gather correctness
# =========================================================================


def _a2a_simple_layer(x, edge_index, rank_assignments, natoms):
    """
    A simple message passing layer using all-to-all communication.
    Computes same result as all-gather version but using all-to-all.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    # Build GP context
    gp_ctx = build_gp_context(edge_index, rank_assignments, rank, world_size)

    # Compute send indices
    send_indices = _compute_send_indices(gp_ctx)

    # All-to-all collect
    x_received = all_to_all_collect(x, gp_ctx, send_indices)

    # Combine local + received
    x_full = torch.cat([x, x_received], dim=0)

    # Remap edges to local space
    edge_index_local = remap_edge_index_to_local(edge_index, gp_ctx.global_to_local)

    # Simple message passing: source embeddings aggregated to targets
    x_source = x_full[edge_index_local[0]]
    x_target = x_full[edge_index_local[1]]

    edge_embeddings = (x_source + 1).pow(1.5) * (x_target + 1).pow(1.5)

    # Aggregate to local atoms only
    local_atoms = gp_ctx.total_local_atoms
    new_node_embedding = torch.zeros(
        local_atoms,
        *edge_embeddings.shape[1:],
        dtype=edge_embeddings.dtype,
        device=edge_embeddings.device,
    )
    # Target indices in local space are in [0, local_atoms) for local targets
    local_target_mask = edge_index_local[1] < local_atoms
    local_edge_idx = edge_index_local[:, local_target_mask]
    local_edge_emb = edge_embeddings[local_target_mask]
    new_node_embedding.index_add_(0, local_edge_idx[1], local_edge_emb)

    return new_node_embedding


def _allgather_simple_layer(x, edge_index, node_offset, natoms):
    """
    A simple message passing layer using all-gather (baseline).
    """
    x_full = gather_from_model_parallel_region_sum_grad(x, natoms)

    x_source = x_full[edge_index[0]]
    x_target = x_full[edge_index[1]]

    local_atoms = size_list_fn(natoms, gp_utils.get_gp_world_size())[
        gp_utils.get_gp_rank()
    ]

    edge_embeddings = (x_source + 1).pow(1.5) * (x_target + 1).pow(1.5)

    new_node_embedding = torch.zeros(
        local_atoms,
        *edge_embeddings.shape[1:],
        dtype=edge_embeddings.dtype,
        device=edge_embeddings.device,
    )
    new_node_embedding.index_add_(0, edge_index[1] - node_offset, edge_embeddings)

    return new_node_embedding


def a2a_vs_allgather_test(atomic_numbers, edge_index):
    """
    Compare all-to-all and all-gather results on the same simple layer.
    Both should produce identical output.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()
    natoms = atomic_numbers.shape[0]

    # Partition atoms (same as gp_utils does)
    node_partition = torch.tensor_split(torch.arange(natoms), world_size)[rank]
    node_offset = node_partition.min().item()

    # Create rank assignments
    rank_assignments = partition_atoms_index_split(
        natoms, world_size, torch.device("cpu")
    )

    # Filter edges: keep edges where target is in our partition
    target_in_partition = (edge_index[1] >= node_partition.min()) & (
        edge_index[1] <= node_partition.max()
    )
    local_edge_index = edge_index[:, target_in_partition]

    # Local embeddings (just use atomic numbers as embedding)
    x_local = atomic_numbers[node_partition].clone().unsqueeze(-1)

    # Run all-gather version
    result_ag = _allgather_simple_layer(x_local, local_edge_index, node_offset, natoms)

    # Run all-to-all version
    result_a2a = _a2a_simple_layer(x_local, local_edge_index, rank_assignments, natoms)

    return {
        "rank": rank,
        "allgather": result_ag.detach(),
        "all_to_all": result_a2a.detach(),
        "match": torch.allclose(result_ag, result_a2a, atol=1e-6),
    }


@pytest.mark.parametrize(
    "num_atoms, edges",
    [
        # Simple linear chain: 0-1-2-3
        (4, [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        # Star graph: 0 connected to all
        (5, [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]),
        # Dense graph: all-to-all edges (4 atoms)
        (
            4,
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            ],
        ),
    ],
)
def test_a2a_vs_allgather(num_atoms, edges):
    """
    Verify that all-to-all produces the same results as all-gather
    for a simple message passing layer.
    """
    atomic_numbers = torch.arange(
        2, 2 + num_atoms, dtype=torch.float, requires_grad=False
    )
    edge_index = torch.tensor(edges, dtype=torch.long)

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_vs_allgather_test,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"Rank {result['rank']}: all-gather and all-to-all produced "
            f"different results.\n"
            f"allgather: {result['allgather']}\n"
            f"all_to_all: {result['all_to_all']}"
        )


def a2a_backward_test(atomic_numbers, edge_index):
    """
    Test that the backward pass of all-to-all produces correct gradients
    by comparing energy and forces computed with all-gather vs all-to-all.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()
    natoms = atomic_numbers.shape[0]

    # Partition atoms
    node_partition = torch.tensor_split(torch.arange(natoms), world_size)[rank]
    node_offset = node_partition.min().item()

    rank_assignments = partition_atoms_index_split(
        natoms, world_size, torch.device("cpu")
    )

    # Filter edges
    target_in_partition = (edge_index[1] >= node_partition.min()) & (
        edge_index[1] <= node_partition.max()
    )
    local_edge_index = edge_index[:, target_in_partition]

    results = {}

    for method in ["allgather", "all_to_all"]:
        # Need fresh requires_grad for each method
        x_local = atomic_numbers[node_partition].clone().detach()
        x_local = x_local.unsqueeze(-1).requires_grad_(True)

        if method == "allgather":
            embedding = _allgather_simple_layer(
                x_local, local_edge_index, node_offset, natoms
            )
        else:
            embedding = _a2a_simple_layer(
                x_local, local_edge_index, rank_assignments, natoms
            )

        # Compute local energy contribution
        energy_part = embedding.sum()
        energy = gp_utils.reduce_from_model_parallel_region(energy_part)

        # Compute forces (gradient w.r.t. x_local)
        forces = torch.autograd.grad(
            [energy],
            [x_local],
            create_graph=False,
        )[0]

        results[f"{method}_energy"] = energy.detach()
        results[f"{method}_forces"] = forces.detach()

    results["rank"] = rank
    results["energy_match"] = torch.allclose(
        results["allgather_energy"],
        results["all_to_all_energy"],
        atol=1e-5,
    )
    results["forces_match"] = torch.allclose(
        results["allgather_forces"],
        results["all_to_all_forces"],
        atol=1e-5,
    )

    return results


def test_a2a_backward():
    """
    Verify that backward pass of all-to-all matches all-gather.
    """
    atomic_numbers = torch.tensor([2.0, 3.0, 5.0, 7.0])
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=torch.long,
    )

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_backward_test,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["energy_match"], (
            f"Rank {result['rank']}: energy mismatch. "
            f"AG={result['allgather_energy']}, A2A={result['all_to_all_energy']}"
        )
        assert result["forces_match"], (
            f"Rank {result['rank']}: forces mismatch. "
            f"AG={result['allgather_forces']}, A2A={result['all_to_all_forces']}"
        )


@pytest.mark.parametrize("world_size", [2, 3])
def test_a2a_multi_rank(world_size):
    """
    Test all-to-all vs all-gather with varying number of GP ranks.
    """
    num_atoms = 6
    # Create a ring graph
    src = list(range(num_atoms))
    dst = [(i + 1) % num_atoms for i in range(num_atoms)]
    # Bidirectional
    edge_src = src + dst
    edge_dst = dst + src
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    atomic_numbers = torch.arange(2, 2 + num_atoms, dtype=torch.float)

    config = PGConfig(
        backend="gloo",
        world_size=world_size,
        gp_group_size=world_size,
        use_gp=True,
    )
    all_rank_results = spawn_multi_process(
        config,
        a2a_vs_allgather_test,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result[
            "match"
        ], f"world_size={world_size}, rank {result['rank']}: mismatch"


def a2a_spatial_partition_test(atomic_numbers, edge_index, pos):
    """
    Test all-to-all with spatial partitioning produces correct results
    by comparing to all-gather (which always uses index-based partitioning).
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()
    natoms = atomic_numbers.shape[0]

    # --- All-gather with index-based partitioning (baseline) ---
    node_partition_idx = torch.tensor_split(torch.arange(natoms), world_size)[rank]
    node_offset_idx = node_partition_idx.min().item()

    target_in_partition_idx = (edge_index[1] >= node_partition_idx.min()) & (
        edge_index[1] <= node_partition_idx.max()
    )
    local_edge_index_idx = edge_index[:, target_in_partition_idx]

    x_local_idx = atomic_numbers[node_partition_idx].clone().unsqueeze(-1)
    result_ag = _allgather_simple_layer(
        x_local_idx, local_edge_index_idx, node_offset_idx, natoms
    )

    # --- All-to-all with spatial partitioning ---
    rank_assignments_spatial = partition_atoms_spatial(pos, world_size)
    local_mask = rank_assignments_spatial == rank
    node_partition_sp = local_mask.nonzero(as_tuple=True)[0]

    target_in_partition_sp = local_mask[edge_index[1]]
    local_edge_index_sp = edge_index[:, target_in_partition_sp]

    x_local_sp = atomic_numbers[node_partition_sp].clone().unsqueeze(-1)
    result_a2a = _a2a_simple_layer(
        x_local_sp, local_edge_index_sp, rank_assignments_spatial, natoms
    )

    # Both methods compute message passing over the SAME global graph,
    # so local atoms get the same aggregated messages regardless of
    # which partition strategy is used. However, different ranks own
    # different atoms under spatial vs index partitioning, so we
    # gather all results and compare the full output.
    # Gather all local results to rank 0 for comparison
    full_ag = gather_from_model_parallel_region_sum_grad(result_ag, natoms)
    full_a2a = gather_from_model_parallel_region_sum_grad(result_a2a, natoms)

    return {
        "rank": rank,
        "allgather_full": full_ag.detach(),
        "all_to_all_full": full_a2a.detach(),
        "match": torch.allclose(full_ag, full_a2a, atol=1e-5),
    }


def test_a2a_spatial_partition():
    """
    Verify that all-to-all with spatial partitioning produces the same
    global results as all-gather with index partitioning.
    """
    num_atoms = 8
    # Create atoms in two spatial clusters
    pos = torch.cat(
        [
            torch.randn(4, 3) + torch.tensor([0.0, 0.0, 0.0]),
            torch.randn(4, 3) + torch.tensor([100.0, 0.0, 0.0]),
        ]
    )
    atomic_numbers = torch.arange(
        2, 2 + num_atoms, dtype=torch.float, requires_grad=False
    )
    # Dense graph connecting all atoms
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_spatial_partition_test,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
        pos,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"Rank {result['rank']}: spatial partitioning produced "
            f"different global results than index partitioning"
        )


class TestEdgeClassification:
    """
    Tests for local_edge_mask precomputation in GPContext.
    """

    def test_edge_mask_types(self):
        """
        Verify that local_edge_mask is computed and has correct type/shape.
        """
        # 6 atoms, 2 ranks, edges cross the partition boundary
        rank_assignments = torch.tensor([0, 0, 0, 1, 1, 1])
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 0, 3],
                [1, 2, 0, 4, 3, 0],
            ]
        )
        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        assert ctx.local_edge_mask is not None
        assert ctx.local_edge_mask.dtype == torch.bool
        assert ctx.local_edge_mask.shape[0] == edge_index.shape[1]
        assert ctx.num_local_edges is not None
        assert ctx.num_boundary_edges is not None
        assert ctx.num_local_edges + ctx.num_boundary_edges == edge_index.shape[1]

    def test_all_local_edges(self):
        """
        When all edges are within the local partition, all should be local.
        """
        rank_assignments = torch.tensor([0, 0, 0, 1, 1, 1])
        # All edges within rank 0's partition
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        assert ctx.num_local_edges == 3
        assert ctx.num_boundary_edges == 0
        assert ctx.local_edge_mask.all()

    def test_all_boundary_edges(self):
        """
        When all edges have remote sources, all should be boundary.
        """
        rank_assignments = torch.tensor([0, 0, 0, 1, 1, 1])
        # All edges from rank 1 atoms to rank 0 atoms
        edge_index = torch.tensor([[3, 4, 5], [0, 1, 2]])
        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        assert ctx.num_local_edges == 0
        assert ctx.num_boundary_edges == 3
        assert not ctx.local_edge_mask.any()

    def test_mixed_edges(self):
        """
        Verify correct classification of mixed local and boundary edges.
        """
        rank_assignments = torch.tensor([0, 0, 0, 1, 1, 1])
        # 4 edges: 2 local (0->1, 1->2), 2 boundary (3->0, 4->1)
        edge_index = torch.tensor(
            [
                [0, 1, 3, 4],
                [1, 2, 0, 1],
            ]
        )
        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        assert ctx.num_local_edges == 2
        assert ctx.num_boundary_edges == 2
        # First 2 edges are local, last 2 are boundary
        expected_mask = torch.tensor([True, True, False, False])
        assert torch.equal(ctx.local_edge_mask, expected_mask)


# =========================================================================
# Distributed tests: async all-to-all correctness
# =========================================================================


def async_a2a_test(atomic_numbers, edge_index):
    """
    Verify that start/finish_all_to_all_collect produces same
    results as the synchronous all_to_all_collect.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()
    natoms = atomic_numbers.shape[0]

    rank_assignments = partition_atoms_index_split(
        natoms, world_size, torch.device("cpu")
    )
    gp_ctx = build_gp_context(edge_index, rank_assignments, rank, world_size)

    x = atomic_numbers[gp_ctx.node_partition].unsqueeze(1).float()
    send_indices = gp_ctx.send_indices

    # Synchronous all-to-all
    x_received_sync = all_to_all_collect(x, gp_ctx, send_indices)

    # Async all-to-all
    recv_buf, work_handles = start_all_to_all_collect(x, gp_ctx, send_indices)
    x_received_async = finish_all_to_all_collect(recv_buf, work_handles)

    match = torch.allclose(x_received_sync, x_received_async, atol=1e-6)

    return {
        "rank": rank,
        "match": match,
        "sync_shape": x_received_sync.shape,
        "async_shape": x_received_async.shape,
    }


def test_async_a2a_matches_sync():
    """
    Verify async start/finish all-to-all matches sync all_to_all_collect.
    """
    num_atoms = 6
    atomic_numbers = torch.arange(
        2, 2 + num_atoms, dtype=torch.float, requires_grad=False
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 4, 4, 5, 5, 0], [1, 2, 0, 4, 5, 3, 5, 3, 4, 3]],
        dtype=torch.long,
    )

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        async_a2a_test,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"Rank {result['rank']}: async and sync all-to-all produced "
            f"different results. sync_shape={result['sync_shape']}, "
            f"async_shape={result['async_shape']}"
        )
