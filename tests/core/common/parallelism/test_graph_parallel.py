"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from ase import Atoms

from fairchem.core.common import gp_utils
from fairchem.core.common.gp_utils import (
    gather_from_model_parallel_region_sum_grad,
    size_list_fn,
)
from fairchem.core.common.parallelism.graph_parallel_a2a import (
    all_to_all_collect,
    build_gp_context,
)
from fairchem.core.common.parallelism.graph_partition import (
    partition_atoms_index_split,
    partition_atoms_spatial,
)
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.models.uma.outputs import (
    compute_forces_and_stress,
    reduce_node_to_system,
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

        # Edge (2, 0): src=2 is remote -> should appear as remote src in edge_index_local
        # Verify edge_index_local has some remote sources (index >= total_local_atoms)
        has_remote_src = (ctx.edge_index_local[0] >= ctx.total_local_atoms).any()
        assert has_remote_src

    def test_edge_index_local_validity(self):
        """
        Verify that edge_index_local indices are valid and correctly
        separate local vs remote sources.
        """
        # 4 atoms: rank 0 owns [0,1], rank 1 owns [2,3]
        # Full graph edges, then filter to rank 0's targets
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        rank_assignments = torch.tensor([0, 0, 1, 1])

        # Filter to edges where target belongs to rank 0
        target_mask = (rank_assignments == 0)[edge_index[1]]
        rank_edge_index = edge_index[:, target_mask]

        ctx = build_gp_context(rank_edge_index, rank_assignments, rank=0, world_size=2)

        # All indices should be non-negative
        assert (ctx.edge_index_local >= 0).all()

        # All indices should be in bounds
        total = ctx.total_local_atoms + ctx.total_recv
        assert (ctx.edge_index_local < total).all()

        # All targets should be local (we filtered to rank 0 targets)
        n_local = ctx.total_local_atoms
        assert (ctx.edge_index_local[1] < n_local).all()

    def test_no_cross_partition_edges(self):
        """
        When no edges cross partitions, no remote atoms are needed.
        """
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Only within rank 0
        rank_assignments = torch.tensor([0, 0, 1, 1])

        ctx = build_gp_context(edge_index, rank_assignments, rank=0, world_size=2)
        # All sources should be local (no remote atoms needed)
        assert (ctx.edge_index_local[0] < ctx.total_local_atoms).all()
        assert ctx.recv_counts.sum() == 0

    def test_edge_split_indices(self):
        """
        Verify local_edge_idx and remote_edge_idx correctly split edges
        by source ownership.
        """
        # 4 atoms, 2 ranks: rank 0 owns [0,1], rank 1 owns [2,3]
        # Edges: (0,1) local-src, (1,0) local-src,
        #        (2,0) remote-src, (3,1) remote-src
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 0, 1]])
        rank_assignments = torch.tensor([0, 0, 1, 1])
        ctx = build_gp_context(
            edge_index,
            rank_assignments,
            rank=0,
            world_size=2,
        )

        assert ctx.local_edge_idx is not None
        assert ctx.remote_edge_idx is not None

        # Check counts: 2 local-src edges (0,1) and (1,0),
        #               2 remote-src edges (2,0) and (3,1)
        edge_index_local = ctx.edge_index_local
        n_local = ctx.total_local_atoms  # 2

        local_srcs = edge_index_local[0, ctx.local_edge_idx]
        remote_srcs = edge_index_local[0, ctx.remote_edge_idx]

        assert (local_srcs < n_local).all()
        assert (remote_srcs >= n_local).all()

        # Together they cover all edges
        assert (
            ctx.local_edge_idx.numel() + ctx.remote_edge_idx.numel()
            == edge_index_local.shape[1]
        )

    def test_edge_split_no_remote_edges(self):
        """
        When all edges are local-source, remote_edge_idx should be empty.
        """
        edge_index = torch.tensor([[0, 1], [1, 0]])
        rank_assignments = torch.tensor([0, 0, 1, 1])
        ctx = build_gp_context(
            edge_index,
            rank_assignments,
            rank=0,
            world_size=2,
        )

        assert ctx.local_edge_idx.numel() == 2
        assert ctx.remote_edge_idx.numel() == 0


def _a2a_simple_layer(x, edge_index, rank_assignments, natoms):
    """
    A simple message passing layer using all-to-all communication.
    Computes same result as all-gather version but using all-to-all.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    # Build GP context (send_indices computed inline)
    gp_ctx = build_gp_context(edge_index, rank_assignments, rank, world_size)

    send_indices = gp_ctx.send_indices

    # All-to-all collect
    x_received = all_to_all_collect(x, gp_ctx, send_indices)

    # Combine local + received
    x_full = torch.cat([x, x_received], dim=0)

    # Use precomputed local edge index
    edge_index_local = gp_ctx.edge_index_local

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
    # Target indices in local space are in [0, local_atoms)
    # for local targets
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
            f"AG={result['allgather_energy']}, "
            f"A2A={result['all_to_all_energy']}"
        )
        assert result["forces_match"], (
            f"Rank {result['rank']}: forces mismatch. "
            f"AG={result['allgather_forces']}, "
            f"A2A={result['all_to_all_forces']}"
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
        assert result["match"], (
            f"world_size={world_size}, " f"rank {result['rank']}: mismatch"
        )


def a2a_spatial_partition_test(atomic_numbers, edge_index, pos):
    """
    Test all-to-all with spatial partitioning produces correct results
    by comparing to all-gather (which always uses index-based
    partitioning).
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
        x_local_sp,
        local_edge_index_sp,
        rank_assignments_spatial,
        natoms,
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


# =========================================================================
# Energy / forces / stress GP correctness tests
# =========================================================================


def _make_water_box(
    num_molecules: int = 10,
    box_length: float = 12.0,
    seed: int = 42,
) -> Atoms:
    """
    Create a periodic box of water molecules.
    """
    rng = np.random.RandomState(seed)
    positions, symbols = [], []
    for _ in range(num_molecules):
        center = rng.uniform(0, box_length, size=3)
        h1 = center + np.array([0.96, 0.0, 0.0])
        h2 = center + np.array([-0.24, 0.93, 0.0])
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        rot = np.array(
            [
                [cos_t * cos_p, -sin_t, cos_t * sin_p],
                [sin_t * cos_p, cos_t, sin_t * sin_p],
                [-sin_p, 0, cos_p],
            ]
        )
        for p in [center, h1, h2]:
            positions.append(center + rot @ (p - center))
            symbols.append("O" if len(symbols) % 3 == 0 else "H")
    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        cell=[box_length] * 3,
        pbc=True,
    )


class _ToyEnergyModel(nn.Module):
    """
    Minimal model: per-atom MLP energy from positions -> stress via autograd.

    Isolates the outputs.py stress computation from the full backbone,
    making the test focused on the GP reduction bug.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pos, cell, batch, num_systems):
        # Route pos through cell so dE/dcell exists for stress computation.
        # Mimics real backbone where cell enters via edge shifts.
        cell_inv = torch.linalg.inv(cell[batch])
        pos_frac = torch.einsum("bi,bij->bj", pos, cell_inv)
        pos_used = torch.einsum("bi,bij->bj", pos_frac, cell[batch])

        # Partition nodes across GP ranks, mimicking the real backbone's
        # graph partitioning where each rank only processes a subset of nodes.
        if gp_utils.initialized():
            node_partition = torch.tensor_split(
                torch.arange(len(pos), device=pos.device),
                gp_utils.get_gp_world_size(),
            )[gp_utils.get_gp_rank()]
            pos_local = pos_used[node_partition]
            batch_local = batch[node_partition]
        else:
            pos_local = pos_used
            batch_local = batch

        node_energy = self.mlp(pos_local).squeeze(-1)
        energy, energy_part = reduce_node_to_system(
            node_energy, batch_local, num_systems
        )
        forces, stress = compute_forces_and_stress(
            energy_part,
            pos,
            cell,
            batch,
            training=self.training,
        )
        return energy, forces, stress


def energy_forces_stress_gp_worker(world_size, natoms, seed=42):
    """
    Worker run on each gloo rank.

    Returns energy, forces, stress from a toy model.
    """
    rank = dist.get_rank()

    torch.manual_seed(seed)
    model = _ToyEnergyModel(hidden=32)
    model.eval()

    atoms = _make_water_box(num_molecules=max(natoms // 3, 1), seed=seed)
    natoms_actual = len(atoms)

    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, requires_grad=True)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32).unsqueeze(0)
    cell.requires_grad_(True)
    batch = torch.zeros(natoms_actual, dtype=torch.long)

    energy, forces, stress = model(pos, cell, batch, num_systems=1)

    return {
        "energy": energy.detach(),
        "forces": forces.detach(),
        "stress": stress.detach(),
        "rank": rank,
    }


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_energy_forces_stress_gp(world_size):
    """
    Verify energy, forces, and stress match between single-process
    reference and N-process graph-parallel execution.

    Catches the double-reduction bug where pos_virial was computed
    from already all-reduced gradients, then re-reduced inside
    reduce_node_to_system.
    """
    natoms = 30

    # 1-process reference (no GP)
    ref_config = PGConfig(backend="gloo", world_size=1, gp_group_size=1, use_gp=False)
    ref_results = spawn_multi_process(
        ref_config,
        energy_forces_stress_gp_worker,
        init_pg_and_rank_and_launch_test,
        1,
        natoms,
    )
    ref = ref_results[0]

    # N-process with GP
    gp_config = PGConfig(
        backend="gloo",
        world_size=world_size,
        gp_group_size=world_size,
        use_gp=True,
    )
    gp_results = spawn_multi_process(
        gp_config,
        energy_forces_stress_gp_worker,
        init_pg_and_rank_and_launch_test,
        world_size,
        natoms,
    )
    gp = gp_results[0]

    assert torch.allclose(ref["energy"], gp["energy"], atol=1e-5), (
        f"Energy mismatch: max_diff="
        f"{(ref['energy'] - gp['energy']).abs().max().item():.6e}"
    )
    assert torch.allclose(ref["forces"], gp["forces"], atol=1e-4), (
        f"Forces mismatch: max_diff="
        f"{(ref['forces'] - gp['forces']).abs().max().item():.6e}"
    )
    assert torch.allclose(ref["stress"], gp["stress"], atol=1e-5), (
        f"Stress mismatch: max_diff="
        f"{(ref['stress'] - gp['stress']).abs().max().item():.6e}"
    )


# =========================================================================
# GPU tests (NCCL, 2 processes)
# =========================================================================

_skip_if_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Multi-GPU test, skipped in CI",
)


def _to_cuda(*tensors):
    device = torch.device(f"cuda:{gp_utils.get_gp_rank()}")
    return tuple(t.to(device) for t in tensors)


def a2a_vs_allgather_test_gpu(atomic_numbers, edge_index):
    (atomic_numbers, edge_index) = _to_cuda(atomic_numbers, edge_index)
    return a2a_vs_allgather_test(atomic_numbers, edge_index)


def a2a_backward_test_gpu(atomic_numbers, edge_index):
    (atomic_numbers, edge_index) = _to_cuda(atomic_numbers, edge_index)
    return a2a_backward_test(atomic_numbers, edge_index)


def a2a_spatial_partition_test_gpu(atomic_numbers, edge_index, pos):
    (atomic_numbers, edge_index, pos) = _to_cuda(atomic_numbers, edge_index, pos)
    return a2a_spatial_partition_test(atomic_numbers, edge_index, pos)


@_skip_if_ci
@pytest.mark.parametrize(
    "num_atoms, edges",
    [
        (4, [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        (5, [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]),
        (
            4,
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            ],
        ),
    ],
)
def test_a2a_vs_allgather_gpu(num_atoms, edges):
    atomic_numbers = torch.arange(
        2, 2 + num_atoms, dtype=torch.float, requires_grad=False
    )
    edge_index = torch.tensor(edges, dtype=torch.long)

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_vs_allgather_test_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"Rank {result['rank']}: all-gather and all-to-all produced "
            f"different results on GPU.\n"
            f"allgather: {result['allgather']}\n"
            f"all_to_all: {result['all_to_all']}"
        )


@_skip_if_ci
def test_a2a_backward_gpu():
    atomic_numbers = torch.tensor([2.0, 3.0, 5.0, 7.0])
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=torch.long,
    )

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_backward_test_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["energy_match"], (
            f"Rank {result['rank']}: energy mismatch on GPU. "
            f"AG={result['allgather_energy']}, "
            f"A2A={result['all_to_all_energy']}"
        )
        assert result["forces_match"], (
            f"Rank {result['rank']}: forces mismatch on GPU. "
            f"AG={result['allgather_forces']}, "
            f"A2A={result['all_to_all_forces']}"
        )


@_skip_if_ci
@pytest.mark.parametrize("world_size", [2, 3])
def test_a2a_multi_rank_gpu(world_size):
    num_atoms = 6
    src = list(range(num_atoms))
    dst = [(i + 1) % num_atoms for i in range(num_atoms)]
    edge_src = src + dst
    edge_dst = dst + src
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    atomic_numbers = torch.arange(2, 2 + num_atoms, dtype=torch.float)

    config = PGConfig(
        backend="nccl",
        world_size=world_size,
        gp_group_size=world_size,
        use_gp=True,
    )
    all_rank_results = spawn_multi_process(
        config,
        a2a_vs_allgather_test_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"world_size={world_size}, " f"rank {result['rank']}: mismatch on GPU"
        )


@_skip_if_ci
def test_a2a_spatial_partition_gpu():
    num_atoms = 8
    pos = torch.cat(
        [
            torch.randn(4, 3) + torch.tensor([0.0, 0.0, 0.0]),
            torch.randn(4, 3) + torch.tensor([100.0, 0.0, 0.0]),
        ]
    )
    atomic_numbers = torch.arange(
        2, 2 + num_atoms, dtype=torch.float, requires_grad=False
    )
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        a2a_spatial_partition_test_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
        pos,
    )

    for result in all_rank_results:
        assert result["match"], (
            f"Rank {result['rank']}: spatial partitioning produced "
            f"different global results than index partitioning on GPU"
        )
