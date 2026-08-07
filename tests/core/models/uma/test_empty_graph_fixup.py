"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the isolated-atoms / empty-global-graph code path.

Covers:
    1. ``add_n_empty_edges(..., fake_atom_idx=N)`` produces self-loops on
       the requested atom, not on global atom 0.
    2. Single-rank (no GP): a 1-atom system and a 2-atom-no-edges system
       both survive the empty-graph fixup.
    3. Two-rank Gloo GP + A2A: a 2-atom-no-edges system where atom 0 lives
       on rank 0 and atom 1 lives on rank 1. Each rank must pad with a
       self-loop on its LOCAL atom so that build_gp_context includes the
       fake edge in edge_index_local. Verifies the fix from
       ``_generate_graph``.
    4. Two-rank Gloo GP + A2A with a 1-atom system: the underpopulated
       rank should hit the ``node_partition.numel() > 0`` assertion, not
       a silent shape mismatch.
"""

from __future__ import annotations

import torch

from fairchem.core.common import gp_utils
from fairchem.core.common.parallelism.graph_parallel_a2a import build_gp_context
from fairchem.core.common.parallelism.graph_partition import partition_atoms_index_split
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.models.uma.escn_md import add_n_empty_edges

# ---------------------------------------------------------------------------
# 1. add_n_empty_edges — fake_atom_idx
# ---------------------------------------------------------------------------


def _empty_graph_dict(device="cpu"):
    return {
        "edge_index": torch.empty(2, 0, dtype=torch.long, device=device),
        "edge_distance": torch.empty(0, device=device),
        "edge_distance_vec": torch.empty(0, 3, device=device),
    }


def test_add_n_empty_edges_defaults_to_atom_0():
    g = _empty_graph_dict()
    add_n_empty_edges(g, 1, cutoff=6.0)
    assert g["edge_index"].shape == (2, 1)
    assert g["edge_index"].tolist() == [[0], [0]]


def test_add_n_empty_edges_uses_fake_atom_idx():
    g = _empty_graph_dict()
    add_n_empty_edges(g, 1, cutoff=6.0, fake_atom_idx=7)
    assert g["edge_index"].tolist() == [[7], [7]]
    assert g["edge_distance"].shape == (1,)
    assert g["edge_distance_vec"].shape == (1, 3)


def test_add_n_empty_edges_multiple():
    g = _empty_graph_dict()
    add_n_empty_edges(g, 3, cutoff=6.0, fake_atom_idx=2)
    assert g["edge_index"].shape == (2, 3)
    assert (g["edge_index"] == 2).all()


def test_add_n_empty_edges_preserves_existing_edges():
    g = {
        "edge_index": torch.tensor([[1, 2], [3, 4]]),
        "edge_distance": torch.tensor([1.0, 2.0]),
        "edge_distance_vec": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float),
    }
    add_n_empty_edges(g, 1, cutoff=6.0, fake_atom_idx=5)
    # New edges prepend
    assert g["edge_index"].shape == (2, 3)
    assert g["edge_index"][:, 0].tolist() == [5, 5]
    assert g["edge_index"][:, 1:].tolist() == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# 2. Single-rank behaviour of build_gp_context under the padded-graph pattern
#    (this is the state _generate_graph produces after the fixup).
# ---------------------------------------------------------------------------


def test_1_atom_padded_graph_is_size_1():
    """1 atom in the world, no GP. After padding, edge_index is size 1."""
    g = _empty_graph_dict()
    add_n_empty_edges(g, 1, cutoff=6.0, fake_atom_idx=0)
    assert g["edge_index"].shape == (2, 1)
    assert g["edge_index"].tolist() == [[0], [0]]


def test_2_atoms_no_edges_padded_graph_is_size_1():
    """2 atoms, no neighbours. After padding, edge_index is size 1."""
    g = _empty_graph_dict()
    add_n_empty_edges(g, 1, cutoff=6.0, fake_atom_idx=0)
    assert g["edge_index"].shape == (2, 1)


# ---------------------------------------------------------------------------
# 3. Two-rank Gloo tests — the important A2A case.
# ---------------------------------------------------------------------------


def _simulate_rank_padded_context(num_atoms, rank, world_size):
    """
    Simulate one rank's view: compute node_partition via index_split,
    then apply the new empty-graph fixup (self-loop on
    ``node_partition[0]``), then call build_gp_context.

    Returns None if this rank has no atoms — that's the case where
    _generate_graph would trip its ``node_partition.numel() > 0``
    assertion in production.

    Runs in a single process — build_gp_context takes rank+world_size as
    parameters and only actually communicates when gp_utils.initialized()
    is True (mirroring the pattern in TestBuildGPContext).
    """
    rank_assignments = partition_atoms_index_split(
        num_atoms, world_size, torch.device("cpu")
    )
    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    if node_partition.numel() == 0:
        return None

    # Empty local edge_index (globally isolated)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    # Simulate the new fixup: self-loop on this rank's first local atom
    fake_atom_idx = int(node_partition[0].item())
    padded = torch.full((2, 1), fake_atom_idx, dtype=torch.long)
    edge_index = torch.cat([padded, edge_index], dim=1)

    ctx = build_gp_context(
        edge_index=edge_index,
        rank_assignments=rank_assignments,
        rank=rank,
        world_size=world_size,
        node_partition=node_partition,
    )
    return ctx


def test_2_atoms_2_ranks_no_edges_a2a_pads_each_rank():
    """
    2 atoms across 2 ranks; both isolated (no real neighbours). Each
    rank's fake self-loop must be on its own local atom so it survives
    the target-in-node_partition filter inside build_gp_context. Result
    on each rank should be edge_index_local of size 1 pointing at
    local-atom-0.
    """
    for rank in [0, 1]:
        ctx = _simulate_rank_padded_context(num_atoms=2, rank=rank, world_size=2)
        assert ctx is not None, f"rank {rank}: unexpectedly empty partition"
        assert (
            ctx.total_local_atoms == 1
        ), f"rank {rank}: total_local_atoms={ctx.total_local_atoms}, expected 1"
        assert ctx.edge_index_local.shape == (2, 1), (
            f"rank {rank}: edge_index_local shape {ctx.edge_index_local.shape}, "
            f"expected (2, 1) — this is the bug that gp_ctx used to have "
            f"shape (2, 0) while wigner had shape (1,)."
        )
        # The fake edge target must be the single local atom (index 0 in
        # local coords).
        assert ctx.edge_index_local[1].tolist() == [0], (
            f"rank {rank}: fake edge target {ctx.edge_index_local[1].tolist()}, "
            f"expected [0]"
        )


def test_1_atom_2_ranks_yields_one_empty_partition():
    """
    1 atom globally + 2 ranks means one rank's partition is empty. In
    the real _generate_graph the ``node_partition.numel() > 0``
    assertion fires on that rank. We simulate both ranks here and
    verify exactly one gets an empty partition.
    """
    ctxs = [
        _simulate_rank_padded_context(num_atoms=1, rank=r, world_size=2) for r in [0, 1]
    ]
    empties = [i for i, c in enumerate(ctxs) if c is None]
    populated = [i for i, c in enumerate(ctxs) if c is not None]
    assert len(empties) == 1, (
        f"expected exactly 1 empty rank, got empties={empties}, "
        f"populated={populated}"
    )
    assert (
        len(populated) == 1
    ), f"expected exactly 1 populated rank, got populated={populated}"
    ctx = ctxs[populated[0]]
    # The populated rank has 1 local atom and 1 padded edge on itself.
    assert ctx.total_local_atoms == 1
    assert ctx.edge_index_local.shape == (2, 1)


# ---------------------------------------------------------------------------
# 4. Two-rank Gloo distributed A2A collect on the padded graph.
#    This is a real multi-process test to catch any collective-level
#    breakage in the padded-empty case.
# ---------------------------------------------------------------------------


def _empty_a2a_gloo_inner(num_atoms):
    from fairchem.core.common.parallelism.graph_parallel_a2a import (
        all_to_all_collect,
    )

    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    rank_assignments = partition_atoms_index_split(
        num_atoms, world_size, torch.device("cpu")
    )
    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    if node_partition.numel() == 0:
        return {"rank": rank, "skipped_no_atoms": True}

    edge_index = torch.empty(2, 0, dtype=torch.long)
    fake_atom_idx = int(node_partition[0].item())
    padded = torch.full((2, 1), fake_atom_idx, dtype=torch.long)
    edge_index = torch.cat([padded, edge_index], dim=1)

    gp_ctx = build_gp_context(
        edge_index=edge_index,
        rank_assignments=rank_assignments,
        rank=rank,
        world_size=world_size,
        node_partition=node_partition,
    )

    # Run an all_to_all collect — under the new fixup no remote atoms
    # are needed since each fake edge is a self-loop, so we expect an
    # empty x_recv but no crash.
    x_local = torch.arange(node_partition.numel() * 3, dtype=torch.float).reshape(-1, 3)
    x_recv = all_to_all_collect(x_local, gp_ctx)
    return {
        "rank": rank,
        "skipped_no_atoms": False,
        "n_local_atoms": int(node_partition.numel()),
        "edge_index_local_shape": list(gp_ctx.edge_index_local.shape),
        "x_recv_shape": list(x_recv.shape),
    }


def test_2_atoms_2_ranks_gloo_a2a_collect():
    """
    Multi-process Gloo test: the padded empty-graph path survives a
    real A2A build + collect roundtrip. Each rank pads with a self-loop
    on its own atom → no cross-rank data movement → collect returns an
    empty tensor.
    """
    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    results = spawn_multi_process(
        config,
        _empty_a2a_gloo_inner,
        init_pg_and_rank_and_launch_test,
        2,  # num_atoms
    )
    for r in results:
        assert not r["skipped_no_atoms"], f"rank {r['rank']} was empty"
        assert r["n_local_atoms"] == 1
        assert r["edge_index_local_shape"] == [2, 1]
        # No remote sources needed
        assert r["x_recv_shape"][0] == 0
