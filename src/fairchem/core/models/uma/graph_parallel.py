"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import torch
from torch import distributed as dist
from torch.profiler import record_function

from fairchem.core.common import gp_utils


def _safe_all_to_all(
    output_list: list[torch.Tensor],
    input_list: list[torch.Tensor],
    group: dist.ProcessGroup,
) -> None:
    """
    All-to-all with fallback for backends that don't support it (e.g. Gloo).

    When the backend supports all_to_all natively (NCCL), uses it directly.
    Otherwise, falls back to pairwise isend/irecv which works on any backend.

    Args:
        output_list: List of output tensors, one per rank.
        input_list: List of input tensors, one per rank.
        group: Process group.
    """
    backend = dist.get_backend(group)
    if backend == "nccl":
        dist.all_to_all(output_list, input_list, group=group)
    else:
        # Gloo fallback: use pairwise send/recv
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        ops = []
        for r in range(world_size):
            if r == rank:
                # Local copy
                if input_list[r].numel() > 0:
                    output_list[r].copy_(input_list[r])
            elif input_list[r].numel() > 0 or output_list[r].numel() > 0:
                # Skip zero-length P2P ops to avoid potential hangs
                ops.append(dist.P2POp(dist.isend, input_list[r], r, group=group))
                ops.append(dist.P2POp(dist.irecv, output_list[r], r, group=group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()


class PartitionStrategy(Enum):
    """
    Strategy for partitioning atoms across GP ranks.

    INDEX_SPLIT: Simple contiguous index split (existing behavior).
    SPATIAL: Spatial domain decomposition using fast k-means.
    """

    INDEX_SPLIT = "index_split"
    SPATIAL = "spatial"


@dataclass
class GPContext:
    """
    Graph parallel context holding per-rank atom assignments
    and communication metadata for all-to-all.

    This replaces the all-gather approach by tracking which atoms
    each rank needs from other ranks for its local edge computations.

    Attributes:
        rank: Current GP rank.
        world_size: Number of GP ranks.
        node_partition: Global indices of atoms owned by this rank.
        rank_assignments: For every atom in the global graph, which rank owns it.
            Shape: (total_atoms,), dtype: int.
        needed_atoms: Global indices of non-local atoms this rank needs
            (sources of edges whose targets are in this rank's partition).
        needed_from_ranks: For each atom in needed_atoms, which rank owns it.
        send_counts: Number of atoms to send to each rank. Shape: (world_size,).
        recv_counts: Number of atoms to receive from each rank. Shape: (world_size,).
        global_to_local: Mapping from global atom index to position in the
            local concatenated tensor [local_atoms | received_atoms].
            Shape: (total_atoms,), with -1 for atoms not accessible.
        total_local_atoms: Number of atoms in this rank's partition.
        total_needed_atoms: Total atoms needed from other ranks.
        send_indices: Precomputed local indices of atoms to send, ordered by
            destination rank. Computed once at build time to avoid per-forward
            all-to-all index exchange. None if not yet computed.
        edge_index_local: Precomputed edge index remapped to local indices.
            None if not yet computed (set by build_gp_context when edge_index
            is provided).
        local_edge_mask: Boolean mask identifying fully-local edges (both src
            and tgt are local atoms). Used for comm-compute overlap. Shape:
            (num_edges,). None if not yet computed.
        num_local_edges: Number of fully-local edges (precomputed from
            local_edge_mask). None if not yet computed.
        num_boundary_edges: Number of boundary edges (src is remote). None
            if not yet computed.
        edge_reorder: Permutation that sorts edges so local edges come first,
            then boundary edges. Shape: (num_edges,). Applied once to per-edge
            tensors (wigner, x_edge, etc.) in the backbone forward. Enables
            compile-friendly overlap via split() instead of boolean indexing.
    """

    rank: int
    world_size: int
    node_partition: torch.Tensor
    rank_assignments: torch.Tensor
    needed_atoms: torch.Tensor
    needed_from_ranks: torch.Tensor
    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    global_to_local: torch.Tensor
    total_local_atoms: int
    total_needed_atoms: int
    send_indices: torch.Tensor | None = None
    edge_index_local: torch.Tensor | None = None
    local_edge_mask: torch.Tensor | None = None
    num_local_edges: int | None = None
    num_boundary_edges: int | None = None
    edge_reorder: torch.Tensor | None = None
    # Precomputed Python lists to avoid repeated .tolist() in AllToAllCollect
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None
    total_recv: int | None = None
    # Precomputed sparse neighbor lists for P2P communication
    # Only includes ranks with non-zero send/recv counts, avoiding
    # O(world_size) iteration in the communication hot path.
    send_neighbors: list[int] | None = None  # Ranks we send to (non-zero send)
    recv_neighbors: list[int] | None = None  # Ranks we recv from (non-zero recv)
    # Pre-allocated receive buffer (lazily initialized).
    # Reused across layers within a step to avoid per-layer CUDA malloc.
    _recv_buf: torch.Tensor | None = None


def _expand_bits_10(v: torch.Tensor) -> torch.Tensor:
    """
    Expand a 10-bit integer so each bit is spaced by 2 zero bits.

    Maps bit at position i to position 3*i, producing a 30-bit
    output suitable for interleaving with two other axes to form
    a Morton Z-order code.

    Args:
        v: Integer tensor with values in [0, 1023].

    Returns:
        Tensor with bits expanded (each input bit at position 3*i).
    """
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def partition_atoms_spatial(
    pos: torch.Tensor,
    num_ranks: int,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    Spatial partitioning via Morton Z-order curve on GPU.

    Computes a 30-bit Morton code per atom by interleaving 10 bits
    from each spatial axis. Sorting by Morton code groups spatially
    nearby atoms together, minimizing boundary edges. Atoms are
    then split into num_ranks equal chunks in sorted order.

    Runs entirely on GPU with zero CPU transfers or sync points
    (unlike recursive coordinate bisection which requires CPU
    round-trips). The Morton curve provides O(N^{2/3}) surface
    fraction per partition, similar to recursive bisection.

    Args:
        pos: Atom positions, shape (N, 3).
        num_ranks: Number of partitions (GP world size).
        num_iters: Unused (kept for API compatibility).

    Returns:
        rank_assignments: Tensor of shape (N,) with rank index
            for each atom, on the same device as pos.
    """
    N = pos.shape[0]
    device = pos.device

    if num_ranks == 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    if num_ranks >= N:
        return torch.arange(N, dtype=torch.long, device=device)

    # Normalize positions to [0, 1023] using a SINGLE global scale
    # factor (the largest bounding-box extent).  Per-dimension
    # normalization would amplify noise in short dimensions, breaking
    # Morton locality (e.g. a 100-unit x-gap becomes indistinguishable
    # from a 2-unit y-gap after independent rescaling).
    min_pos = pos.min(0)[0]
    extent = (pos.max(0)[0] - min_pos).max().clamp(min=1e-8)
    norm = ((pos - min_pos) / extent * 1023).long().clamp(0, 1023)

    # 30-bit Morton Z-order code: interleave x, y, z bits
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]
    morton = _expand_bits_10(x) | (_expand_bits_10(y) << 1) | (_expand_bits_10(z) << 2)

    # Sort by Morton code and assign to ranks in balanced chunks.
    # Use ``i * P // N`` mapping (not ``i // ceil(N/P)``) to ensure
    # EVERY rank receives at least ``floor(N/P)`` atoms.  The ceil-based
    # formula leaves trailing ranks empty when N is not a multiple of P
    # (e.g. 1000 atoms / 64 ranks → rank 63 gets 0 atoms, causing a
    # hang in collective communication).
    _, sorted_indices = morton.sort()
    assignments = torch.empty(N, dtype=torch.long, device=device)
    assignments[sorted_indices] = torch.arange(N, device=device) * num_ranks // N

    return assignments


def partition_atoms_index_split(
    total_atoms: int,
    num_ranks: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Simple contiguous index split (matches existing fairchem behavior).

    Args:
        total_atoms: Total number of atoms.
        num_ranks: Number of GP ranks.
        device: Device for the output tensor.

    Returns:
        rank_assignments: Tensor of shape (total_atoms,) with rank for each atom.
    """
    assignments = torch.empty(total_atoms, dtype=torch.long, device=device)
    partitions = torch.tensor_split(torch.arange(total_atoms, device=device), num_ranks)
    for rank, partition in enumerate(partitions):
        assignments[partition] = rank
    return assignments


@torch.compiler.disable
def build_gp_context(
    edge_index: torch.Tensor,
    rank_assignments: torch.Tensor,
    rank: int,
    world_size: int,
    send_info: dict | None = None,
    node_partition: torch.Tensor | None = None,
) -> GPContext:
    """
    Build the GP context from edge connectivity and atom assignments.

    Determines which non-local atoms this rank needs (edge sources from
    other ranks), exchanges atom indices via all-to-all, and computes
    all communication metadata.

    When send_info is provided (pre-computed during graph filtering in
    filter_edges_by_node_partition), the NCCL index-exchange collective
    is skipped entirely — send_counts and send_indices_global are taken
    directly from send_info.

    Args:
        edge_index: Edge index filtered to edges whose targets are in
            this rank's partition, shape (2, num_local_edges).
            Row 0 = source, row 1 = target.
        rank_assignments: Rank assignment for each atom, shape (total_atoms,).
        rank: This rank's GP rank.
        world_size: GP world size.
        send_info: Pre-computed send/recv metadata from graph filtering.
            If provided, must contain:
            - send_counts: Tensor of shape (world_size,) with count of
              atoms to send to each rank.
            - send_indices_global: Tensor of global atom indices to send,
              sorted by destination rank.
            Optionally:
            - recv_counts: Override recv_counts (world_size,).
            - recv_indices_global: Override needed_atoms.
            When provided, _sparse_index_exchange is skipped.
        node_partition: Pre-computed atom indices in this rank's partition.
            If provided, avoids recomputing from rank_assignments.

    Returns:
        GPContext with all metadata needed for all-to-all communication.
    """
    total_atoms = rank_assignments.shape[0]
    device = rank_assignments.device

    # Atoms owned by this rank (reuse pre-computed if available)
    if node_partition is None:
        node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    total_local_atoms = node_partition.shape[0]

    # Find which non-local atoms this rank needs as edge sources.
    # Since edge_index is already filtered to edges whose targets are
    # in this rank's partition, every edge has a local target. We only
    # need to find edges where the SOURCE is remote (not in our partition).
    local_mask = rank_assignments == rank  # (total_atoms,) bool
    src_is_remote = ~local_mask[edge_index[0]]

    # Remote sources needed for local targets
    # Use boolean mask + nonzero instead of .unique(sorted=True) on raw
    # edge sources — O(N) scatter + scan vs O(E log E) sort.
    needed_mask = torch.zeros(total_atoms, dtype=torch.bool, device=device)
    needed_mask[edge_index[0, src_is_remote]] = True
    needed_mask &= ~local_mask  # exclude local atoms (safety)
    needed_atoms = needed_mask.nonzero(as_tuple=True)[0]

    total_needed_atoms = needed_atoms.shape[0]
    needed_from_ranks = rank_assignments[needed_atoms]

    # Compute recv_counts: how many atoms we receive from each rank
    if total_needed_atoms > 0:
        recv_counts = torch.bincount(needed_from_ranks, minlength=world_size).to(
            dtype=torch.long, device=device
        )
    else:
        recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    recv_counts[rank] = 0  # Never receive from self

    # CRITICAL: Sort needed_atoms by source rank to match recv_buf ordering.
    # all_to_all fills recv_buf by source rank: [atoms from rank 0 | atoms
    # from rank 1 | ...]. Within each rank, atoms are in the order we
    # requested them (global index order, since argsort is stable).
    # global_to_local must assign local indices in this SAME order,
    # otherwise local index i maps to recv_buf[i] which has a DIFFERENT
    # atom's embedding.
    # With index_split, global index order == rank order (no-op sort).
    # With spatial, global index order != rank order → sort is essential.
    sort_order = needed_from_ranks.argsort(stable=True)
    needed_atoms = needed_atoms[sort_order]
    needed_from_ranks = needed_from_ranks[sort_order]

    # Use pre-computed send_info when available to skip the
    # _sparse_index_exchange NCCL collectives entirely.
    #
    # When send_info contains recv_counts/recv_indices_global (AABB),
    # it's a SUPERSET of edge-based needed_atoms (AABB is conservative).
    # The extra atoms waste some bandwidth/memory but don't affect
    # correctness since they are never referenced by any edge.
    if send_info is not None and "recv_counts" in send_info:
        # AABB send_info — use fully (send + recv), skip NCCL.
        send_counts = send_info["send_counts"]
        send_indices_global = send_info["send_indices_global"]
        recv_counts = send_info["recv_counts"]
        needed_atoms = send_info["recv_indices_global"]
        needed_from_ranks = rank_assignments[needed_atoms]
        total_needed_atoms = needed_atoms.shape[0]
        # AABB provides needed_atoms in global-index order, but
        # all_to_all fills recv_buf in source-rank order. Sort to match.
        aabb_sort = needed_from_ranks.argsort(stable=True)
        needed_atoms = needed_atoms[aabb_sort]
        needed_from_ranks = needed_from_ranks[aabb_sort]
    elif send_info is not None:
        # Pre-computed during graph filtering (no recv_counts) — use it.
        send_counts = send_info["send_counts"]
        send_indices_global = send_info["send_indices_global"]
    else:
        with record_function("a2a_sparse_index_exchange"):
            send_counts, send_indices_global = _sparse_index_exchange(
                needed_atoms=needed_atoms,
                needed_from_ranks=needed_from_ranks,
                recv_counts=recv_counts,
                rank=rank,
                world_size=world_size,
                device=device,
            )

    # Build global_to_local mapping:
    # Local atoms: index 0..total_local_atoms-1 (in order of node_partition)
    # Received atoms: index total_local_atoms..total_local_atoms+total_needed
    # IMPORTANT: needed_atoms is sorted by source rank (not global index)
    # to match the recv_buf ordering from all_to_all. This ensures that
    # local index (total_local + i) maps to recv_buf[i], which contains
    # the embedding of needed_atoms[i].
    global_to_local = torch.full((total_atoms,), -1, dtype=torch.long, device=device)
    # Map local atoms
    global_to_local[node_partition] = torch.arange(
        total_local_atoms, dtype=torch.long, device=device
    )
    # Map needed remote atoms (in recv_buf order = source rank order)
    global_to_local[needed_atoms] = torch.arange(
        total_local_atoms,
        total_local_atoms + total_needed_atoms,
        dtype=torch.long,
        device=device,
    )

    # Convert send_indices from global to local
    send_indices = None
    has_send = send_indices_global is not None
    if has_send and send_indices_global.numel() > 0:
        send_indices = global_to_local[send_indices_global]
    elif has_send:
        send_indices = torch.empty(0, dtype=torch.long, device=device)

    # Precompute edge_index_local
    edge_index_local = global_to_local[edge_index]

    # Classify edges: fully-local (both endpoints local) vs boundary
    # (source is remote). Used for communication-computation overlap.
    src_is_local = edge_index_local[0] < total_local_atoms
    tgt_is_local_edge = edge_index_local[1] < total_local_atoms
    local_edge_mask = src_is_local & tgt_is_local_edge

    # Pre-compute edge reorder permutation: local edges first, boundary
    # edges last. This enables compile-friendly overlap via split()
    # instead of boolean indexing. The reorder is applied in the
    # backbone forward to all per-edge tensors simultaneously.
    edge_reorder = torch.argsort((~local_edge_mask).to(torch.int32), stable=True)

    # Batch ALL GPU→CPU scalar extractions into a single transfer.
    # This batches send_counts, recv_counts, local_edge_count, AND
    # validation scalars into ONE .cpu() call, eliminating 2 extra
    # GPU→CPU syncs from separate .all()/.any() validation checks.
    local_edge_count = local_edge_mask.sum().unsqueeze(0).to(torch.long)
    bad_edge_count = (edge_index_local < 0).sum().unsqueeze(0).to(torch.long)
    send_valid = (
        torch.ones(1, dtype=torch.long, device=device)
        if send_indices is None or send_indices.numel() == 0
        else (
            ((send_indices >= 0) & (send_indices < total_local_atoms))
            .all()
            .unsqueeze(0)
            .to(torch.long)
        )
    )
    all_cpu = torch.cat(
        [send_counts, recv_counts, local_edge_count, bad_edge_count, send_valid]
    ).cpu()
    send_splits = all_cpu[:world_size].tolist()
    recv_splits = all_cpu[world_size : 2 * world_size].tolist()
    total_recv = sum(recv_splits)
    num_local_edges = int(all_cpu[2 * world_size].item())
    num_boundary_edges = edge_index_local.shape[1] - num_local_edges
    n_bad = int(all_cpu[2 * world_size + 1].item())
    send_ok = int(all_cpu[2 * world_size + 2].item())

    # Validate AFTER the batched CPU transfer (no extra GPU syncs).
    if not send_ok:
        # Diagnostic: identify which send_indices are out of range.
        bad_mask = (send_indices < 0) | (send_indices >= total_local_atoms)
        n_bad_send = bad_mask.sum().item()
        n_total_send = send_indices.numel()
        bad_global = send_indices_global[bad_mask][:10].tolist()
        bad_ra = rank_assignments[send_indices_global[bad_mask][:10]].tolist()
        raise RuntimeError(
            f"Rank {rank}: received requests for atoms not in our "
            f"partition ({n_bad_send}/{n_total_send} OOB). "
            f"bad_global={bad_global}, bad_ranks={bad_ra}. "
            f"This usually means rank_assignments differs across "
            f"ranks (e.g. non-deterministic crystal generation)."
        )
    if n_bad > 0:
        # Only compute diagnostics in the error path (rare).
        bad_cols = (edge_index_local < 0).any(dim=0)
        bad_globals = edge_index[:, bad_cols].unique()
        bad_ranks = rank_assignments[bad_globals]

        # Compute edge-based needed atoms for comparison.
        edge_src_remote = ~local_mask[edge_index[0]]
        edge_needed_mask = torch.zeros(total_atoms, dtype=torch.bool, device=device)
        edge_needed_mask[edge_index[0, edge_src_remote]] = True
        edge_needed_mask &= ~local_mask
        edge_needed_count = edge_needed_mask.sum().item()

        # Check which edge-needed atoms are NOT in our needed_atoms.
        needed_set = torch.zeros(total_atoms, dtype=torch.bool, device=device)
        needed_set[needed_atoms] = True
        missing = edge_needed_mask & ~needed_set
        missing_count = missing.sum().item()

        # Check for local atoms in needed_atoms.
        local_in_needed = (local_mask & needed_set).sum().item()

        # Check for bad local atoms (local atoms mapped to -1).
        local_is_bad = local_mask[bad_globals]
        n_local_bad = local_is_bad.sum().item()

        logging.error(
            f"Rank {rank}: AABB DIAGNOSTIC — "
            f"{n_bad} entries in edge_index_local are -1. "
            f"edge_needed={edge_needed_count}, "
            f"aabb_needed={total_needed_atoms}, "
            f"missing_from_aabb={missing_count}, "
            f"local_in_needed={local_in_needed}, "
            f"n_local_bad={n_local_bad}/{len(bad_globals)} bad globals, "
            f"total_atoms={total_atoms}, "
            f"total_local={total_local_atoms}, "
            f"node_partition_range=[{node_partition.min().item()}, "
            f"{node_partition.max().item()}], "
            f"send_info_provided={send_info is not None}, "
            f"bad_globals[:20]={bad_globals.tolist()[:20]}, "
            f"bad_ranks[:20]={bad_ranks.tolist()[:20]}"
        )
        if missing_count > 0:
            missing_indices = missing.nonzero(as_tuple=True)[0][:10]
            logging.error(
                f"Rank {rank}: Missing atoms (edge-needed but "
                f"not in AABB): {missing_indices.tolist()}"
            )
        raise RuntimeError(
            f"Rank {rank}: edge_index has {n_bad} endpoints not in "
            f"global_to_local mapping. AABB may not be conservative enough."
        )

    return GPContext(
        rank=rank,
        world_size=world_size,
        node_partition=node_partition,
        rank_assignments=rank_assignments,
        needed_atoms=needed_atoms,
        needed_from_ranks=needed_from_ranks,
        send_counts=send_counts,
        recv_counts=recv_counts,
        global_to_local=global_to_local,
        total_local_atoms=total_local_atoms,
        total_needed_atoms=total_needed_atoms,
        send_indices=send_indices,
        edge_index_local=edge_index_local,
        local_edge_mask=local_edge_mask,
        num_local_edges=num_local_edges,
        num_boundary_edges=num_boundary_edges,
        edge_reorder=edge_reorder,
        # Precompute Python lists once (avoids .tolist() per layer per forward)
        send_splits=send_splits,
        recv_splits=recv_splits,
        total_recv=total_recv,
        # Precompute sparse neighbor lists for P2P communication
        send_neighbors=[
            r for r in range(world_size) if send_splits[r] > 0 and r != rank
        ],
        recv_neighbors=[
            r for r in range(world_size) if recv_splits[r] > 0 and r != rank
        ],
    )


def _sparse_index_exchange(
    needed_atoms: torch.Tensor,
    needed_from_ranks: torch.Tensor,
    recv_counts: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Variable-split index exchange using two small all-to-alls.

    Step 1: Exchange recv_counts to get send_counts (P ints).
    Step 2: Exchange actual atom indices with variable split sizes.

    Sends only the exact number of indices needed (no padding),
    keeping communication volume minimal.

    Args:
        needed_atoms: Global indices of atoms this rank needs, sorted.
        needed_from_ranks: Source rank for each needed atom.
        recv_counts: Number of atoms needed from each rank.
        rank: This rank's GP rank.
        world_size: GP world size.
        device: Tensor device.

    Returns:
        Tuple of (send_counts, send_indices_global).
    """
    if not gp_utils.initialized():
        return (
            torch.zeros(world_size, dtype=torch.long, device=device),
            None,
        )

    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)

    # Step 1: Exchange counts.
    # What rank A calls recv_counts[B] is what rank B must send to A.
    # So rank B's send_counts[A] = rank A's recv_counts[B].
    # all_to_all on a (world_size,) tensor transposes the count matrix.
    send_counts = torch.empty(world_size, dtype=torch.long, device=device)
    if backend == "nccl":
        dist.all_to_all_single(send_counts, recv_counts.contiguous(), group=gp_group)
    else:
        # Gloo fallback: use pairwise send/recv
        send_list = list(recv_counts.split(1))
        recv_list = list(send_counts.split(1))
        _safe_all_to_all(recv_list, send_list, group=gp_group)

    # Step 2: Exchange actual atom indices with variable splits.
    # Build send buffer: needed_atoms sorted by source rank.
    if needed_atoms.numel() > 0:
        sort_order = needed_from_ranks.argsort(stable=True)
        send_buf = needed_atoms[sort_order].contiguous()
    else:
        send_buf = torch.empty(0, dtype=torch.long, device=device)

    total_recv_indices = send_counts.sum().item()
    recv_buf = torch.empty(total_recv_indices, dtype=torch.long, device=device)

    send_splits = recv_counts.tolist()  # what we send = what others need from us
    recv_splits = send_counts.tolist()  # what we recv = what we need from others

    if backend == "nccl":
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=gp_group,
        )
    else:
        # Gloo fallback: use pairwise send/recv
        send_list = list(send_buf.split(send_splits))
        recv_list = list(recv_buf.split(recv_splits))
        _safe_all_to_all(recv_list, send_list, group=gp_group)

    # recv_buf now contains the global indices of atoms we must SEND,
    # ordered by destination rank.
    send_indices_global = recv_buf

    return send_counts, send_indices_global


class AllToAllCollect(torch.autograd.Function):
    """
    Autograd function that uses all-to-all to collect only the needed
    remote atom embeddings, replacing the all-gather approach.

    Forward: Sends local atom embeddings to ranks that need them,
    receives remote atom embeddings that we need. Returns only the
    received remote embeddings (NOT concatenated with local).

    Backward: Reverses the communication — sends gradient of received
    embeddings back to their owners, receives gradient of sent embeddings.

    Optimizations over naive all-to-all:
    - Uses ``all_to_all_single`` on NCCL to avoid Python list creation
      from ``split()`` — communicates packed tensors directly.
    - Returns the pre-allocated receive buffer directly instead of
      ``torch.cat(recv_list)`` — avoids a redundant copy.
    - Accepts precomputed ``send_splits``/``recv_splits`` to avoid
      repeated ``.tolist()`` calls per layer.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        x_local: torch.Tensor,
        send_indices: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        gp_group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        precomputed_send_splits: list[int] | None = None,
        precomputed_recv_splits: list[int] | None = None,
        precomputed_total_recv: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_local: Local atom embeddings, shape (local_atoms, *feature_dims).
            send_indices: Local indices of atoms to send, ordered by dest rank.
            send_counts: Number of atoms to send to each rank.
            recv_counts: Number of atoms to receive from each rank.
            gp_group: GP process group.
            rank: GP rank.
            world_size: GP world size.
            precomputed_send_splits: Optional cached send_counts.tolist().
            precomputed_recv_splits: Optional cached recv_counts.tolist().
            precomputed_total_recv: Optional cached sum(recv_splits).

        Returns:
            Received remote embeddings, shape (sum(recv_counts), *feature_dims).
        """
        ctx.send_indices = send_indices
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.gp_group = gp_group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_size = x_local.shape[0]
        # Cache precomputed splits for backward (avoids .tolist() per layer)
        ctx.precomputed_send_splits = precomputed_send_splits
        ctx.precomputed_recv_splits = precomputed_recv_splits

        feature_shape = x_local.shape[1:]

        # Gather atoms to send (index_select into contiguous buffer)
        if send_indices.numel() > 0:
            x_send = x_local[send_indices].contiguous()
        else:
            x_send = torch.empty(
                0, *feature_shape, device=x_local.device, dtype=x_local.dtype
            )

        # Use precomputed splits if available (avoids .tolist() per layer)
        send_splits = (
            precomputed_send_splits
            if precomputed_send_splits is not None
            else send_counts.tolist()
        )
        recv_splits = (
            precomputed_recv_splits
            if precomputed_recv_splits is not None
            else recv_counts.tolist()
        )
        total_recv = (
            precomputed_total_recv
            if precomputed_total_recv is not None
            else sum(recv_splits)
        )
        x_recv = torch.empty(
            total_recv, *feature_shape, device=x_local.device, dtype=x_local.dtype
        )

        # Perform all-to-all communication
        backend = dist.get_backend(gp_group)
        if backend == "nccl":
            # Use all_to_all_single for NCCL — avoids Python list creation
            # from split(). Operates on packed contiguous tensors directly.
            dist.all_to_all_single(
                x_recv,
                x_send,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=gp_group,
            )
        else:
            # Gloo fallback: use list-based pairwise send/recv
            send_list = list(x_send.split(send_splits))
            recv_list = list(x_recv.split(recv_splits))
            _safe_all_to_all(recv_list, send_list, group=gp_group)

        # x_recv already contains all received data in rank order —
        # no need for torch.cat (recv_list are views into x_recv).
        return x_recv

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_received: torch.Tensor):
        """
        Reverse the all-to-all: send gradients back to the ranks that
        originally sent us the embeddings.
        """
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        send_indices = ctx.send_indices
        gp_group = ctx.gp_group
        local_size = ctx.local_size

        feature_shape = grad_received.shape[1:]

        # In backward, the roles are reversed:
        # What we received in forward, we now send back gradients for
        # What we sent in forward, we now receive gradients for
        bwd_send_splits = (
            ctx.precomputed_recv_splits
            if ctx.precomputed_recv_splits is not None
            else recv_counts.tolist()
        )
        bwd_recv_splits = (
            ctx.precomputed_send_splits
            if ctx.precomputed_send_splits is not None
            else send_counts.tolist()
        )

        total_bwd_recv = sum(bwd_recv_splits)
        grad_send_back = torch.empty(
            total_bwd_recv,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        # Reverse all-to-all
        backend = dist.get_backend(gp_group)
        if backend == "nccl":
            # Use all_to_all_single for NCCL — packed tensor, no list
            dist.all_to_all_single(
                grad_send_back,
                grad_received.contiguous(),
                output_split_sizes=bwd_recv_splits,
                input_split_sizes=bwd_send_splits,
                group=gp_group,
            )
        else:
            # Gloo fallback
            bwd_send_list = list(grad_received.split(bwd_send_splits))
            bwd_recv_list = list(grad_send_back.split(bwd_recv_splits))
            _safe_all_to_all(bwd_recv_list, bwd_send_list, group=gp_group)

        # Scatter received gradients back to local positions
        grad_local = torch.zeros(
            local_size,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        if total_bwd_recv > 0:
            # grad_send_back already contains data — no cat needed
            grad_local.index_add_(0, send_indices, grad_send_back)

        # Return gradients for x_local only; None for all other inputs
        return grad_local, None, None, None, None, None, None, None, None, None


# Module-level storage for async work handles.
# Keyed by id(recv_buffer) → work handle. Cleaned up after wait.
_ASYNC_A2A_WORK: dict[int, dist.Work | list[dist.Work]] = {}


class AsyncAllToAllCollect(torch.autograd.Function):
    """
    Autograd-compatible ASYNC all-to-all collect.

    Like AllToAllCollect but uses ``async_op=True`` so the NCCL
    operation runs in the background. The caller MUST call
    ``wait_async_all_to_all_collect(x_recv)`` before using x_recv.

    This enables overlapping communication with local edge computation
    in the ``_forward_overlap`` path while still supporting autograd
    (needed for autograd-based forces/stress).

    The backward is identical to AllToAllCollect — a synchronous
    reverse all-to-all. Only the forward is async.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        x_local: torch.Tensor,
        send_indices: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        gp_group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        precomputed_send_splits: list[int] | None = None,
        precomputed_recv_splits: list[int] | None = None,
        precomputed_total_recv: int | None = None,
    ) -> torch.Tensor:
        """
        Start async all-to-all. Returns x_recv (NOT yet complete).

        Caller MUST call wait_async_all_to_all_collect(x_recv)
        before reading x_recv values.
        """
        ctx.send_indices = send_indices
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.gp_group = gp_group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_size = x_local.shape[0]
        ctx.precomputed_send_splits = precomputed_send_splits
        ctx.precomputed_recv_splits = precomputed_recv_splits

        feature_shape = x_local.shape[1:]

        # Gather atoms to send
        if send_indices.numel() > 0:
            x_send = x_local[send_indices].contiguous()
        else:
            x_send = torch.empty(
                0, *feature_shape, device=x_local.device, dtype=x_local.dtype
            )

        send_splits = (
            precomputed_send_splits
            if precomputed_send_splits is not None
            else send_counts.tolist()
        )
        recv_splits = (
            precomputed_recv_splits
            if precomputed_recv_splits is not None
            else recv_counts.tolist()
        )
        total_recv = (
            precomputed_total_recv
            if precomputed_total_recv is not None
            else sum(recv_splits)
        )
        x_recv = torch.empty(
            total_recv,
            *feature_shape,
            device=x_local.device,
            dtype=x_local.dtype,
        )

        # Launch async all-to-all
        backend = dist.get_backend(gp_group)
        if backend == "nccl":
            work = dist.all_to_all_single(
                x_recv,
                x_send,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=gp_group,
                async_op=True,
            )
            _ASYNC_A2A_WORK[id(x_recv)] = work
        else:
            # Gloo: no async support, fall back to sync
            send_list = list(x_send.split(send_splits))
            recv_list = list(x_recv.split(recv_splits))
            _safe_all_to_all(recv_list, send_list, group=gp_group)
            # No work handle needed — already complete

        return x_recv

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_received: torch.Tensor):
        """
        Reverse A2A (synchronous). Same as AllToAllCollect.backward.
        """
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        send_indices = ctx.send_indices
        gp_group = ctx.gp_group
        local_size = ctx.local_size

        feature_shape = grad_received.shape[1:]

        bwd_send_splits = (
            ctx.precomputed_recv_splits
            if ctx.precomputed_recv_splits is not None
            else recv_counts.tolist()
        )
        bwd_recv_splits = (
            ctx.precomputed_send_splits
            if ctx.precomputed_send_splits is not None
            else send_counts.tolist()
        )

        total_bwd_recv = sum(bwd_recv_splits)
        grad_send_back = torch.empty(
            total_bwd_recv,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        backend = dist.get_backend(gp_group)
        if backend == "nccl":
            dist.all_to_all_single(
                grad_send_back,
                grad_received.contiguous(),
                output_split_sizes=bwd_recv_splits,
                input_split_sizes=bwd_send_splits,
                group=gp_group,
            )
        else:
            bwd_send_list = list(grad_received.split(bwd_send_splits))
            bwd_recv_list = list(grad_send_back.split(bwd_recv_splits))
            _safe_all_to_all(bwd_recv_list, bwd_send_list, group=gp_group)

        grad_local = torch.zeros(
            local_size,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        if total_bwd_recv > 0:
            grad_local.index_add_(0, send_indices, grad_send_back)

        return (
            grad_local,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def start_all_to_all_collect_autograd(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Start async A2A with autograd support. Returns x_recv (not ready).

    The caller MUST call ``wait_async_all_to_all_collect(x_recv)``
    before using x_recv values. Between start and wait, the caller
    can run local computation to overlap with the communication.

    Unlike ``start_all_to_all_collect``, this version participates in
    autograd — gradients flow through the communication in backward.

    Args:
        x_local: Local embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context.
        send_indices: Local indices of atoms to send.

    Returns:
        x_recv: NOT-YET-READY tensor. Must wait before use.
    """
    if send_indices is None:
        raise ValueError(
            "send_indices is None — build_gp_context should always "
            "compute send_indices. Check GP setup."
        )
    return AsyncAllToAllCollect.apply(
        x_local,
        send_indices,
        gp_ctx.send_counts,
        gp_ctx.recv_counts,
        gp_utils.get_gp_group(),
        gp_ctx.rank,
        gp_ctx.world_size,
        gp_ctx.send_splits,
        gp_ctx.recv_splits,
        gp_ctx.total_recv,
    )


@torch.compiler.disable
def wait_async_all_to_all_collect(x_recv: torch.Tensor) -> None:
    """
    Wait for async A2A started by start_all_to_all_collect_autograd.

    After this returns, x_recv contains valid data.

    Args:
        x_recv: The tensor returned by start_all_to_all_collect_autograd.
    """
    key = id(x_recv)
    work = _ASYNC_A2A_WORK.pop(key, None)
    if work is not None:
        work.wait()


def all_to_all_collect(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> torch.Tensor:
    """
    High-level function to collect remote embeddings via all-to-all.

    Returns the received remote embeddings (NOT including local).
    The caller should concatenate [x_local, received] and use
    gp_ctx.global_to_local to index into this combined tensor.

    Args:
        x_local: Local atom embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context.
        send_indices: Local indices of atoms to send.

    Returns:
        x_received: Remote atom embeddings, shape (total_needed, *features).
    """
    if send_indices is None:
        raise ValueError(
            "send_indices is None — build_gp_context should always "
            "compute send_indices. Check GP setup."
        )
    return AllToAllCollect.apply(
        x_local,
        send_indices,
        gp_ctx.send_counts,
        gp_ctx.recv_counts,
        gp_utils.get_gp_group(),
        gp_ctx.rank,
        gp_ctx.world_size,
        gp_ctx.send_splits,
        gp_ctx.recv_splits,
        gp_ctx.total_recv,
    )


def all_to_all_collect_compiled(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compile-friendly all-to-all collect using functional collectives.

    Uses ``torch.distributed._functional_collectives.all_to_all_single``
    which is a registered PyTorch op — torch.compile can trace through it
    WITHOUT creating a graph break. This eliminates the per-layer graph
    break from the ``@torch.compiler.disable`` on
    ``AllToAllCollect.forward()``.

    This function does NOT support autograd — gradients will not flow
    through the communication. When gradients are needed (e.g., autograd
    forces via ``torch.autograd.grad(energy, pos)``), use
    ``all_to_all_collect`` instead, which uses an autograd.Function with
    proper backward support.

    NOTE: ``all_to_all_single_autograd`` (the funcoll autograd variant)
    crashes with torch.compile because it doesn't handle symbolic split
    sizes (SymInt). Both BL (all-gather) and A2A have a graph break when
    autograd is needed, so this is not a regression vs the baseline.

    For MD simulation with spatial partitioning, the split sizes are
    effectively constant for hundreds of steps (atoms barely move per
    timestep), so torch.compile guards on the split sizes will pass
    without recompilation.

    NOTE: Requires NCCL backend. Functional collectives are not
    supported on Gloo. CPU/Gloo tests should use ``all_to_all_collect``.

    Args:
        x_local: Local atom embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context.
        send_indices: Local indices of atoms to send.

    Returns:
        x_received: Remote atom embeddings, shape (total_needed, *features).
    """
    if send_indices is None:
        raise ValueError(
            "send_indices is None — build_gp_context should always "
            "compute send_indices. Check GP setup."
        )

    # Gather atoms to send (compile-friendly indexing)
    x_send = x_local[send_indices].contiguous()

    # Use functional collective — no graph break, no autograd
    gp_group = gp_utils.get_gp_group()

    from torch.distributed._functional_collectives import (
        all_to_all_single as functional_a2a,
    )

    x_recv = functional_a2a(
        x_send,
        output_split_sizes=gp_ctx.recv_splits,
        input_split_sizes=gp_ctx.send_splits,
        group=gp_group,
    )

    return x_recv


@torch.compiler.disable
def all_to_all_collect_p2p(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Collect remote embeddings using sparse P2P communication.

    Instead of ``all_to_all_single`` (which creates P-1 send/recv pairs
    even for zero-length messages), this uses ``batch_isend_irecv`` with
    only the non-zero neighbors. At 64 GPUs with spatial partitioning,
    each rank typically has ~10-15 actual neighbors, so this reduces
    the number of NCCL operations from ~63 to ~25.

    Uses pre-allocated send/recv buffers stored on ``gp_ctx`` to avoid
    per-layer CUDA memory allocation overhead (4 allocations/step saved).

    Does NOT participate in autograd — intended for eval-mode inference
    only. For training, use ``all_to_all_collect`` instead.

    Args:
        x_local: Local atom embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context with precomputed neighbor lists.
        send_indices: Local indices of atoms to send.

    Returns:
        x_received: Remote atom embeddings, shape (total_needed, *features).
    """
    if send_indices is None:
        raise ValueError(
            "send_indices is None — build_gp_context should always "
            "compute send_indices. Check GP setup."
        )
    feature_shape = x_local.shape[1:]
    send_splits = gp_ctx.send_splits
    recv_splits = gp_ctx.recv_splits
    total_recv = gp_ctx.total_recv

    # Gather atoms to send
    # Note: cannot use torch.index_select with out= because x_local
    # may require grad (for force computation), and out= doesn't
    # support autograd. Use regular indexing which creates a new tensor
    # but supports the backward pass. The send buffer pre-allocation
    # is not worth the autograd complexity.
    if send_indices.numel() > 0:
        x_send = x_local[send_indices].contiguous()
    else:
        x_send = torch.empty(
            0, *feature_shape, device=x_local.device, dtype=x_local.dtype
        )

    # Reuse pre-allocated recv buffer if available and correct size
    recv_shape = (total_recv, *feature_shape)
    if (
        gp_ctx._recv_buf is not None
        and gp_ctx._recv_buf.shape == recv_shape
        and gp_ctx._recv_buf.dtype == x_local.dtype
    ):
        x_recv = gp_ctx._recv_buf
    else:
        x_recv = torch.empty(recv_shape, device=x_local.device, dtype=x_local.dtype)
        gp_ctx._recv_buf = x_recv

    # Sparse P2P communication: only talk to actual neighbors
    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)

    if backend == "nccl":
        # Split into per-rank chunks (views into contiguous buffers)
        send_chunks = list(x_send.split(send_splits))
        recv_chunks = list(x_recv.split(recv_splits))

        ops = []
        # Only create ops for non-zero neighbors
        for r in gp_ctx.send_neighbors:
            ops.append(dist.P2POp(dist.isend, send_chunks[r], r, group=gp_group))
        for r in gp_ctx.recv_neighbors:
            ops.append(dist.P2POp(dist.irecv, recv_chunks[r], r, group=gp_group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
    else:
        # Gloo fallback: use pairwise send/recv
        send_chunks = list(x_send.split(send_splits))
        recv_chunks = list(x_recv.split(recv_splits))
        _safe_all_to_all(recv_chunks, send_chunks, group=gp_group)

    return x_recv


@torch.compiler.disable
def start_all_to_all_collect(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> tuple[torch.Tensor, list[dist.Work]]:
    """
    Start async all-to-all communication for comm-compute overlap.

    Launches the all-to-all without waiting for completion. Returns
    the pre-allocated receive buffer and work handles. The caller
    should do useful compute, then call ``finish_all_to_all_collect``
    to wait for completion and get the received embeddings.

    This function does NOT participate in autograd. For differentiable
    all-to-all, use ``all_to_all_collect`` instead. This async variant
    is intended for the overlap path where gradients are handled
    separately.

    Uses ``all_to_all_single`` on NCCL for efficiency (avoids Python
    list creation from ``split()``).

    Args:
        x_local: Local atom embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context.
        send_indices: Local indices of atoms to send.

    Returns:
        Tuple of (recv_buffer, work_handles):
            recv_buffer: Pre-allocated tensor for received embeddings.
            work_handles: List of dist.Work handles to wait on.
    """
    feature_shape = x_local.shape[1:]

    # Gather atoms to send
    if send_indices.numel() > 0:
        x_send = x_local[send_indices].contiguous()
    else:
        x_send = torch.empty(
            0, *feature_shape, device=x_local.device, dtype=x_local.dtype
        )

    # Use precomputed splits if available (avoids .tolist() per layer)
    send_splits = (
        gp_ctx.send_splits
        if gp_ctx.send_splits is not None
        else gp_ctx.send_counts.tolist()
    )
    recv_splits = (
        gp_ctx.recv_splits
        if gp_ctx.recv_splits is not None
        else gp_ctx.recv_counts.tolist()
    )
    total_recv = (
        gp_ctx.total_recv if gp_ctx.total_recv is not None else sum(recv_splits)
    )
    x_recv = torch.empty(
        total_recv, *feature_shape, device=x_local.device, dtype=x_local.dtype
    )

    # Launch async all-to-all
    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)

    work_handles = []
    if backend == "nccl":
        # Use all_to_all_single for NCCL — packed tensor, no list creation
        work = dist.all_to_all_single(
            x_recv,
            x_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=gp_group,
            async_op=True,
        )
        work_handles.append(work)
    else:
        # Gloo fallback: use pairwise send/recv
        send_list = list(x_send.split(send_splits))
        recv_list = list(x_recv.split(recv_splits))
        rank = dist.get_rank(gp_group)
        world_size = dist.get_world_size(gp_group)
        ops = []
        for r in range(world_size):
            if r == rank:
                if send_list[r].numel() > 0:
                    recv_list[r].copy_(send_list[r])
            elif send_list[r].numel() > 0 or recv_list[r].numel() > 0:
                # Skip zero-length P2P ops to avoid potential hangs
                ops.append(dist.P2POp(dist.isend, send_list[r], r, group=gp_group))
                ops.append(dist.P2POp(dist.irecv, recv_list[r], r, group=gp_group))
        if ops:
            work_handles = dist.batch_isend_irecv(ops)

    return x_recv, work_handles


@torch.compiler.disable
def finish_all_to_all_collect(
    recv_buffer: torch.Tensor,
    work_handles: list[dist.Work],
) -> torch.Tensor:
    """
    Wait for async all-to-all to complete and return received embeddings.

    Args:
        recv_buffer: Pre-allocated receive buffer from start_all_to_all_collect.
        work_handles: Work handles from start_all_to_all_collect.

    Returns:
        x_received: Received remote atom embeddings.
    """
    for work in work_handles:
        work.wait()
    return recv_buffer
