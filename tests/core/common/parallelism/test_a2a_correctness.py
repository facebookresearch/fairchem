"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

A2A (all-to-all) graph parallel correctness tests.

Verifies that the A2A graph parallel implementation produces
correct results via multi-process Gloo tests (CPU).

Run via pytest:
    pytest test_a2a_correctness.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

from fairchem.core.common import gp_utils
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

# =========================================================================
# Pytest-compatible distributed tests (CPU, Gloo, 2 processes)
# =========================================================================


def _correctness_test_inner(
    atomic_numbers,
    pos,
    edge_index,
    num_atoms,
    partition_strategy,
):
    """
    Inner test function run on each rank.

    Builds GPContext with both BL-style (index_split) and A2A (spatial)
    partitioning, runs all-to-all collect, and verifies that the received
    embeddings are correct by checking that each received atom's value
    matches the expected value from the global embedding tensor.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    # Create rank assignments
    if partition_strategy == "spatial":
        rank_assignments = partition_atoms_spatial(pos, world_size)
    else:
        rank_assignments = partition_atoms_index_split(
            num_atoms, world_size, pos.device
        )

    # Get this rank's partition
    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]

    # Filter edges to this rank's partition
    target_mask = (rank_assignments == rank)[edge_index[1]]
    rank_edge_index = edge_index[:, target_mask]

    # Build GP context
    gp_ctx = build_gp_context(
        rank_edge_index,
        rank_assignments,
        rank,
        world_size,
        node_partition=node_partition,
    )

    # Create a global embedding where each atom's embedding is its
    # atomic number (unique per atom). This makes it trivial to verify
    # that the right atoms were received.
    x_global = atomic_numbers.unsqueeze(1).float()
    x_local = x_global[node_partition]

    send_indices = gp_ctx.send_indices

    # Test collect function
    x_recv_autograd = all_to_all_collect(x_local, gp_ctx, send_indices)

    # Verify edge_index_local is valid
    x_full = torch.cat([x_local, x_recv_autograd], dim=0)
    edge_valid = (gp_ctx.edge_index_local >= 0).all().item()
    edge_in_bounds = (gp_ctx.edge_index_local < x_full.shape[0]).all().item()

    # Verify message passing produces the same result as
    # non-distributed. Simple sum aggregation: for each local target,
    # sum source embeddings.
    x_source = x_full[gp_ctx.edge_index_local[0]]
    local_result = torch.zeros(
        gp_ctx.total_local_atoms,
        x_source.shape[1],
        dtype=x_source.dtype,
        device=x_source.device,
    )
    local_result.index_add_(0, gp_ctx.edge_index_local[1], x_source)

    # Reference: compute the same aggregation on the full graph
    x_source_ref = x_global[rank_edge_index[0]]
    ref_result = torch.zeros(
        num_atoms,
        x_source_ref.shape[1],
        dtype=x_source_ref.dtype,
        device=x_source_ref.device,
    )
    ref_result.index_add_(0, rank_edge_index[1], x_source_ref)
    ref_local = ref_result[node_partition]

    mp_match = torch.allclose(local_result, ref_local, atol=1e-6)

    return {
        "rank": rank,
        "partition_strategy": partition_strategy,
        "world_size": world_size,
        "local_atoms": gp_ctx.total_local_atoms,
        "num_edges": rank_edge_index.shape[1],
        "edge_valid": edge_valid,
        "edge_in_bounds": edge_in_bounds,
        "mp_match": mp_match,
    }


@pytest.mark.parametrize(
    "strategy,num_atoms",
    [
        ("index_split", 8),
        ("index_split", 20),
        ("spatial", 8),
        ("spatial", 20),
    ],
)
def test_a2a_correctness_gloo(strategy, num_atoms):
    """
    Verify A2A correctness at 2 GPUs using Gloo backend.

    Creates a dense graph (all atoms connected) and verifies that:
    1. Edge indices are valid (non-negative, in-bounds)
    2. Message passing produces correct aggregation vs reference
    """
    # Create dense graph
    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(2, 2 + num_atoms, dtype=torch.float)
    pos = torch.randn(num_atoms, 3) * 10  # spread out for spatial

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["edge_valid"], f"Rank {r}: edge_index_local has negative entries"
        assert result[
            "edge_in_bounds"
        ], f"Rank {r}: edge_index_local has out-of-bounds entries"
        assert result["mp_match"], (
            f"Rank {r}: message passing result differs " f"from reference"
        )


@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_consistency_across_graph_sizes(strategy):
    """
    Verify A2A correctness with sparse graphs (not all-to-all
    connected).

    Uses a chain graph (each atom connected to its neighbors within
    distance 2) to test the case where not every rank needs atoms
    from every other rank.
    """
    num_atoms = 16

    # Chain graph: atom i connected to i-1, i+1, i-2, i+2
    src, dst = [], []
    for i in range(num_atoms):
        for d in [-2, -1, 1, 2]:
            j = (i + d) % num_atoms  # wrap around
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(10, 10 + num_atoms, dtype=torch.float)
    # Linear arrangement for clear spatial partitioning
    pos = torch.zeros(num_atoms, 3)
    pos[:, 0] = torch.arange(num_atoms, dtype=torch.float)

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["mp_match"], (
            f"Rank {r}: message passing result differs " f"from reference"
        )


def _multidim_test_inner(x_global, pos, edge_index, num_atoms, strategy):
    """
    Test A2A correctness with multi-dimensional embeddings.
    Defined at module level for pickle compatibility with
    multiprocessing.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    if strategy == "spatial":
        rank_assignments = partition_atoms_spatial(pos, world_size)
    else:
        rank_assignments = partition_atoms_index_split(
            num_atoms, world_size, pos.device
        )

    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    target_mask = (rank_assignments == rank)[edge_index[1]]
    rank_edge_index = edge_index[:, target_mask]

    gp_ctx = build_gp_context(
        rank_edge_index,
        rank_assignments,
        rank,
        world_size,
        node_partition=node_partition,
    )

    x_local = x_global[node_partition]
    send_indices = gp_ctx.send_indices

    x_recv = all_to_all_collect(x_local, gp_ctx, send_indices)

    # Verify message passing produces the same result as non-distributed.
    x_full = torch.cat([x_local, x_recv], dim=0)
    x_source = x_full[gp_ctx.edge_index_local[0]]
    local_result = torch.zeros(
        gp_ctx.total_local_atoms,
        x_source.shape[1],
        dtype=x_source.dtype,
        device=x_source.device,
    )
    local_result.index_add_(0, gp_ctx.edge_index_local[1], x_source)

    # Reference: compute the same aggregation on the full graph
    x_source_ref = x_global[rank_edge_index[0]]
    ref_result = torch.zeros(
        num_atoms,
        x_source_ref.shape[1],
        dtype=x_source_ref.dtype,
        device=x_source_ref.device,
    )
    ref_result.index_add_(0, rank_edge_index[1], x_source_ref)
    ref_local = ref_result[node_partition]

    mp_match = torch.allclose(local_result, ref_local, atol=1e-6)

    return {
        "rank": rank,
        "mp_match": mp_match,
        "recv_shape": x_recv.shape,
    }


@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_multidim_embeddings(strategy):
    """
    Verify correctness with multi-dimensional embeddings (not just
    scalars).

    Uses 16-dim embeddings to match the typical sphere_channels
    in UMA.
    """
    num_atoms = 12
    embed_dim = 16

    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Use random embeddings instead of scalar atomic numbers
    torch.manual_seed(42)
    x_global = torch.randn(num_atoms, embed_dim)
    pos = torch.randn(num_atoms, 3) * 10

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)

    all_rank_results = spawn_multi_process(
        config,
        _multidim_test_inner,
        init_pg_and_rank_and_launch_test,
        x_global,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["mp_match"], (
            f"Rank {r}: multidim mp mismatch, " f"recv_shape={result['recv_shape']}"
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


def _correctness_test_inner_gpu(
    atomic_numbers, pos, edge_index, num_atoms, partition_strategy
):
    (atomic_numbers, pos, edge_index) = _to_cuda(atomic_numbers, pos, edge_index)
    return _correctness_test_inner(
        atomic_numbers, pos, edge_index, num_atoms, partition_strategy
    )


def _multidim_test_inner_gpu(x_global, pos, edge_index, num_atoms, strategy):
    (x_global, pos, edge_index) = _to_cuda(x_global, pos, edge_index)
    return _multidim_test_inner(x_global, pos, edge_index, num_atoms, strategy)


@_skip_if_ci
@pytest.mark.parametrize(
    "strategy,num_atoms",
    [
        ("index_split", 8),
        ("index_split", 20),
        ("spatial", 8),
        ("spatial", 20),
    ],
)
def test_a2a_correctness_gpu(strategy, num_atoms):
    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(2, 2 + num_atoms, dtype=torch.float)
    pos = torch.randn(num_atoms, 3) * 10

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result[
            "edge_valid"
        ], f"Rank {r}: edge_index_local has negative entries on GPU"
        assert result[
            "edge_in_bounds"
        ], f"Rank {r}: edge_index_local has out-of-bounds entries on GPU"
        assert result[
            "mp_match"
        ], f"Rank {r}: message passing result differs from reference on GPU"


@_skip_if_ci
@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_consistency_across_graph_sizes_gpu(strategy):
    num_atoms = 16

    src, dst = [], []
    for i in range(num_atoms):
        for d in [-2, -1, 1, 2]:
            j = (i + d) % num_atoms
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(10, 10 + num_atoms, dtype=torch.float)
    pos = torch.zeros(num_atoms, 3)
    pos[:, 0] = torch.arange(num_atoms, dtype=torch.float)

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner_gpu,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result[
            "mp_match"
        ], f"Rank {r}: message passing result differs from reference on GPU"


@_skip_if_ci
@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_multidim_embeddings_gpu(strategy):
    num_atoms = 12
    embed_dim = 16

    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    torch.manual_seed(42)
    x_global = torch.randn(num_atoms, embed_dim)
    pos = torch.randn(num_atoms, 3) * 10

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)

    all_rank_results = spawn_multi_process(
        config,
        _multidim_test_inner_gpu,
        init_pg_and_rank_and_launch_test,
        x_global,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["mp_match"], (
            f"Rank {r}: multidim mp mismatch on GPU, "
            f"recv_shape={result['recv_shape']}"
        )
