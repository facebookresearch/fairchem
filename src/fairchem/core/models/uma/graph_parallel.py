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


def _compute_send_info_aabb(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor | None,
    rank_assignments: torch.Tensor,
    node_partition: torch.Tensor,
    rank: int,
    world_size: int,
) -> dict:
    """
    Compute send and recv info from AABB geometry without NCCL collectives.

    For each remote rank R, computes the axis-aligned bounding box (AABB)
    of atoms assigned to R, expands it by cutoff, then checks which local
    atoms fall within that expanded AABB (considering periodic images for
    PBC systems). Local atoms within cutoff of rank R's domain are sent
    to rank R.

    Also computes recv info: for each remote rank R, checks which of R's
    atoms fall within THIS rank's expanded AABB. This is the symmetric
    computation and ensures send/recv counts are consistent across ranks
    (rank A's send_counts[B] == rank B's recv_counts[A]) without any
    NCCL communication.

    Args:
        pos: All atom positions, shape (N, 3).
        cell: Unit cell, shape (1, 3, 3) or (B, 3, 3).
        cutoff: Neighbor cutoff radius.
        pbc: Periodic boundary conditions, shape (B, 3) or None.
        rank_assignments: Rank for each atom, shape (N,).
        node_partition: Indices of atoms in this rank's partition.
        rank: This rank's GP rank.
        world_size: GP world size.

    Returns:
        dict with keys:
            send_counts: Tensor (world_size,) — atoms to send to each rank.
            send_indices_global: Tensor of global atom indices to send,
                sorted by destination rank.
            recv_counts: Tensor (world_size,) — atoms to receive from
                each rank (what they will send to us based on AABB).
            recv_indices_global: Tensor of global atom indices we will
                receive, sorted by source rank.
    """
    device = pos.device
    local_pos = pos[node_partition]  # (n_local, 3)

    # Compute AABB for each rank using scatter min/max.
    # Initialize with extreme values so empty ranks stay excluded.
    aabb_min = torch.full((world_size, 3), float("inf"), device=device, dtype=pos.dtype)
    aabb_max = torch.full(
        (world_size, 3), float("-inf"), device=device, dtype=pos.dtype
    )

    # Expand rank_assignments for broadcasting: (N, 1)
    ra_expanded = rank_assignments.unsqueeze(1).expand(-1, 3)
    aabb_min.scatter_reduce_(0, ra_expanded, pos, reduce="amin")
    aabb_max.scatter_reduce_(0, ra_expanded, pos, reduce="amax")

    # Build PBC shift vectors. For PBC systems, atoms near box edges
    # may be within cutoff of a remote rank's domain through a periodic
    # image. We check all 27 shifts (including identity).
    use_pbc = pbc is not None and pbc.any()
    if use_pbc:
        # cell is (B, 3, 3) — use first system's cell for shifts
        c = cell[0]  # (3, 3)
        offsets = torch.tensor([-1, 0, 1], device=device, dtype=pos.dtype)
        # Create 27 shift combinations
        shifts_ijk = torch.stack(
            torch.meshgrid(offsets, offsets, offsets, indexing="ij"),
            dim=-1,
        ).reshape(27, 3)  # (27, 3)
        shift_vectors = shifts_ijk @ c  # (27, 3)
    else:
        # No PBC: only check identity (no periodic images)
        shift_vectors = torch.zeros(1, 3, device=device, dtype=pos.dtype)

    # --- PRE-FILTER: Identify neighbor ranks via AABB overlap ---
    # Instead of checking all atoms against all ranks' AABBs (O(N*P*S)),
    # first determine which remote ranks could possibly overlap with our
    # local AABB (expanded by cutoff). With spatial partitioning at 64+
    # GPUs, only ~10-15 ranks are actual neighbors; pre-filtering reduces
    # per-atom checks by 4-5x.
    my_min_raw = aabb_min[rank] + cutoff  # undo expansion → raw AABB
    my_max_raw = aabb_max[rank] + cutoff
    my_min_expanded = my_min_raw - cutoff
    my_max_expanded = my_max_raw + cutoff

    # Check which remote rank AABBs overlap with our expanded AABB
    # across any PBC shift. Two AABBs overlap if they overlap in all
    # 3 dimensions.
    # Remote raw AABBs: (world_size, 3)
    remote_min_raw = aabb_min + cutoff  # undo expansion
    remote_max_raw = aabb_max + cutoff

    # For each shift, check AABB overlap (vectorized, no Python loop):
    #   shifted_remote = remote + shift → check overlap with our AABB
    # remote_min/max_raw: (world_size, 3)
    # shift_vectors: (n_shifts, 3)
    # Broadcast: (n_shifts, world_size, 3)
    shifted_rmin = remote_min_raw.unsqueeze(0) + shift_vectors.unsqueeze(1)  # (S, W, 3)
    shifted_rmax = remote_max_raw.unsqueeze(0) + shift_vectors.unsqueeze(1)  # (S, W, 3)
    overlap = (
        (shifted_rmax >= my_min_expanded) & (shifted_rmin <= my_max_expanded)
    ).all(dim=-1)  # (S, W)
    is_neighbor = overlap.any(dim=0)  # (W,)

    is_neighbor[rank] = False  # exclude self

    # Convert to list of neighbor ranks for efficient indexing
    neighbor_ranks = is_neighbor.nonzero(as_tuple=True)[0]  # (n_neighbors,)

    # Expand AABBs by cutoff (for the actual atom-level checks below)
    aabb_min_exp = aabb_min - cutoff
    aabb_max_exp = aabb_max + cutoff

    # --- SEND side: check local atoms against NEIGHBOR ranks' AABBs ---
    # Only check against n_neighbors ranks instead of world_size.
    # shifted_pos: (n_shifts, n_local, 3)
    shifted_pos = local_pos.unsqueeze(0) + shift_vectors.unsqueeze(1)

    # Select only neighbor AABBs
    neighbor_lo = aabb_min_exp[neighbor_ranks]  # (n_neighbors, 3)
    neighbor_hi = aabb_max_exp[neighbor_ranks]  # (n_neighbors, 3)

    sp = shifted_pos.unsqueeze(2)  # (n_shifts, n_local, 1, 3)
    lo = neighbor_lo.unsqueeze(0).unsqueeze(0)  # (1, 1, n_neighbors, 3)
    hi = neighbor_hi.unsqueeze(0).unsqueeze(0)  # (1, 1, n_neighbors, 3)

    # in_box: True if atom is within AABB in all 3 dimensions
    in_box = ((sp >= lo) & (sp <= hi)).all(dim=-1)  # (n_shifts, n_local, n_neighbors)

    # Collapse shifts: atom is reachable if ANY periodic image is in box
    reachable_nbr = in_box.any(dim=0)  # (n_local, n_neighbors)

    # Reconstruct full send_counts (world_size,) from neighbor subset
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    nbr_counts = reachable_nbr.sum(dim=0).to(torch.long)  # (n_neighbors,)
    send_counts[neighbor_ranks] = nbr_counts

    # Build send_indices_global sorted by rank
    atom_idx, nbr_idx = reachable_nbr.nonzero(as_tuple=True)
    global_atoms = node_partition[atom_idx]
    dest_ranks = neighbor_ranks[nbr_idx]
    sort_order = dest_ranks.argsort(stable=True)
    send_indices_global = global_atoms[sort_order]

    # --- RECV side: check NEIGHBOR atoms against OUR expanded AABB ---
    # Only process atoms from neighbor ranks, not all N atoms.
    my_lo = aabb_min_exp[rank]  # (3,)
    my_hi = aabb_max_exp[rank]  # (3,)

    # Gather atoms from neighbor ranks only
    neighbor_mask = is_neighbor[rank_assignments]  # (N,) — True for neighbor atoms
    neighbor_atom_indices = neighbor_mask.nonzero(as_tuple=True)[0]
    neighbor_pos = pos[neighbor_atom_indices]  # (n_neighbor_atoms, 3)

    # shifted_nbr: (n_shifts, n_neighbor_atoms, 3)
    shifted_nbr = neighbor_pos.unsqueeze(0) + shift_vectors.unsqueeze(1)
    in_my_box = ((shifted_nbr >= my_lo) & (shifted_nbr <= my_hi)).all(
        dim=-1
    )  # (n_shifts, n_neighbor_atoms)
    reachable_to_me = in_my_box.any(dim=0)  # (n_neighbor_atoms,)

    # Map back to global indices and build recv_counts
    recv_global = neighbor_atom_indices[reachable_to_me]
    if recv_global.numel() > 0:
        recv_from_ranks = rank_assignments[recv_global]
        recv_counts = torch.bincount(recv_from_ranks, minlength=world_size).to(
            torch.long
        )
        recv_counts[rank] = 0
        sort_order_recv = recv_from_ranks.argsort(stable=True)
        recv_indices_global = recv_global[sort_order_recv]
    else:
        recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        recv_indices_global = torch.empty(0, dtype=torch.long, device=device)

    return {
        "send_counts": send_counts,
        "send_indices_global": send_indices_global,
        "recv_counts": recv_counts,
        "recv_indices_global": recv_indices_global,
    }


def _recursive_bisect(
    pos: torch.Tensor,
    assignments: torch.Tensor,
    indices: torch.Tensor,
    rank_offset: int,
    num_parts: int,
) -> None:
    """
    Recursively bisect a group of atoms along the longest axis.

    Args:
        pos: Full position tensor (N, 3).
        assignments: Output rank assignment tensor (N,), modified in-place.
        indices: Global indices of atoms in this group.
        rank_offset: Starting rank ID for this group.
        num_parts: Number of partitions to create from this group.
    """
    if num_parts == 1:
        assignments[indices] = rank_offset
        return

    # Find longest axis of this group's bounding box
    group_pos = pos[indices]
    ranges = group_pos.max(dim=0)[0] - group_pos.min(dim=0)[0]
    axis = ranges.argmax().item()

    # Sort along this axis and split in half
    axis_vals = group_pos[:, axis]
    sorted_order = axis_vals.argsort()
    sorted_indices = indices[sorted_order]

    mid = len(sorted_indices) // 2
    left_half = num_parts // 2
    right_half = num_parts - left_half

    _recursive_bisect(pos, assignments, sorted_indices[:mid], rank_offset, left_half)
    _recursive_bisect(
        pos, assignments, sorted_indices[mid:], rank_offset + left_half, right_half
    )


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
    other ranks), exchanges atom indices via a single fused all-to-all,
    and computes all communication metadata.

    When send_info is provided (pre-computed during graph filtering in
    filter_edges_by_node_partition), the NCCL index-exchange collective
    is skipped entirely — send_counts and send_indices_global are taken
    directly from send_info. This eliminates the most expensive collective
    in the setup path.

    Args:
        edge_index: Edge index filtered to edges whose targets are in
            this rank's partition, shape (2, num_local_edges).
            Row 0 = source, row 1 = target.
        rank_assignments: Rank assignment for each atom, shape (total_atoms,).
        rank: This rank's GP rank.
        world_size: GP world size.
        send_info: Pre-computed send/recv metadata from AABB geometry.
            If provided, must contain:
            - send_counts: Tensor of shape (world_size,) with count of
              atoms to send to each rank.
            - send_indices_global: Tensor of global atom indices to send,
              sorted by destination rank.
            - recv_counts: Tensor of shape (world_size,) with count of
              atoms to receive from each rank (AABB-based).
            - recv_indices_global: Tensor of global atom indices to
              receive, sorted by source rank.
            When provided, _fused_index_exchange is skipped and AABB
            recv data overrides edge-based recv_counts/needed_atoms.
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

    # Use AABB send_info when available to skip the _fused_index_exchange
    # NCCL collective entirely. AABB computes send_counts and recv_counts
    # symmetrically (rank A's send_counts[B] == rank B's recv_counts[A])
    # without any communication.
    #
    # AABB recv_indices_global is a SUPERSET of edge-based needed_atoms
    # (AABB is conservative — L2 ≤ L-inf for each axis). The extra atoms
    # waste some bandwidth/memory but don't affect correctness since they
    # are never referenced by any edge.
    if send_info is not None and "recv_counts" in send_info:
        # AABB send_info — use fully (send + recv), skip NCCL.
        send_counts = send_info["send_counts"]
        send_indices_global = send_info["send_indices_global"]
        recv_counts = send_info["recv_counts"]
        needed_atoms = send_info["recv_indices_global"]
        needed_from_ranks = rank_assignments[needed_atoms]
        total_needed_atoms = needed_atoms.shape[0]
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
        else (send_indices >= 0).all().unsqueeze(0).to(torch.long)
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
        raise RuntimeError(
            f"Rank {rank}: received requests for atoms not in our partition"
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


def _fused_index_exchange(
    needed_atoms: torch.Tensor,
    needed_from_ranks: torch.Tensor,
    recv_counts: torch.Tensor,
    rank: int,
    world_size: int,
    total_atoms: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fused count + index exchange using a single padded all-to-all.

    Replaces the old 2-step approach (count exchange via all_to_all +
    index exchange via all_to_all) with a single all_to_all_single using
    fixed-size padded chunks. Each rank pads its needed atom indices to
    ceil(N/P) per destination rank, sends everything in one collective,
    then derives send_counts from the received data by counting
    non-sentinel entries.

    This halves the number of collective operations in the GP setup path.

    Args:
        needed_atoms: Global indices of atoms this rank needs, sorted.
        needed_from_ranks: Source rank for each needed atom.
        recv_counts: Number of atoms needed from each rank.
        rank: This rank's GP rank.
        world_size: GP world size.
        total_atoms: Total number of atoms in the system.
        device: Tensor device.

    Returns:
        Tuple of (send_counts, send_indices_global):
            send_counts: Number of atoms to send to each rank (world_size,).
            send_indices_global: Global indices of atoms to send, ordered
                by destination rank. None if distributed is not initialized.
    """
    if not gp_utils.initialized():
        return (
            torch.zeros(world_size, dtype=torch.long, device=device),
            None,
        )

    # Max atoms per rank chunk in the padded buffer.
    # Upper bound: no rank can need more than ceil(N/P) atoms from us
    # (our entire partition).
    max_per_rank = (total_atoms + world_size - 1) // world_size

    # Build padded send buffer: needed_atoms sorted by source rank,
    # each rank-chunk padded to max_per_rank with sentinel value.
    total_padded = world_size * max_per_rank
    # Sentinel = total_atoms (guaranteed > any valid atom index)
    send_buf = torch.full((total_padded,), total_atoms, dtype=torch.long, device=device)

    if needed_atoms.numel() > 0:
        # Sort needed atoms by source rank for contiguous chunks
        sort_order = needed_from_ranks.argsort()
        needed_sorted = needed_atoms[sort_order]

        # Compute buffer position for each atom:
        # position = source_rank * max_per_rank + within_rank_offset
        cumcounts = torch.arange(len(needed_sorted), device=device)
        recv_cumsum = recv_counts.cumsum(0)
        rank_starts = torch.zeros(world_size, dtype=torch.long, device=device)
        rank_starts[1:] = recv_cumsum[:-1]
        ranks_sorted = needed_from_ranks[sort_order]
        within_rank_idx = cumcounts - rank_starts[ranks_sorted]
        buf_idx = ranks_sorted * max_per_rank + within_rank_idx
        send_buf[buf_idx] = needed_sorted

    # Single all-to-all with equal split sizes (no count exchange needed)
    recv_buf = torch.empty(total_padded, dtype=torch.long, device=device)

    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)
    if backend == "nccl":
        # Equal splits → NCCL can optimize the all-to-all schedule
        dist.all_to_all_single(recv_buf, send_buf, group=gp_group)
    else:
        # Gloo fallback: pairwise send/recv of fixed-size chunks
        M = max_per_rank
        ops = []
        for r in range(world_size):
            if r == rank:
                recv_buf[r * M : (r + 1) * M].copy_(send_buf[r * M : (r + 1) * M])
            else:
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        send_buf[r * M : (r + 1) * M],
                        r,
                        group=gp_group,
                    )
                )
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_buf[r * M : (r + 1) * M],
                        r,
                        group=gp_group,
                    )
                )
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    # Extract send_counts and send_indices from the padded recv buffer.
    # Valid entries have value < total_atoms (sentinel = total_atoms).
    recv_2d = recv_buf.view(world_size, max_per_rank)
    valid_mask = recv_2d < total_atoms
    send_counts = valid_mask.sum(dim=1).to(torch.long)

    # Extract valid global indices in row-major order (rank 0 first, etc.)
    # Since padding is at the end of each chunk, within-chunk order is
    # preserved, matching what the old 2-step approach produced.
    send_indices_global = recv_2d[valid_mask]

    return send_counts, send_indices_global


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

    Step 1: Exchange recv_counts to get send_counts (64 ints = 0.5KB).
    Step 2: Exchange actual atom indices with variable split sizes.

    This sends 12.8x less data than _fused_index_exchange at 64 GPUs
    (40KB vs 512KB) because it avoids padding empty rank-chunks to
    ceil(N/P) with sentinel values.

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


def _compute_send_indices(
    gp_ctx: GPContext,
) -> torch.Tensor:
    """
    Compute which local atoms to send to other ranks, and in what order.

    Uses a single all-to-all exchange of atom indices (vectorized,
    no Python loop over ranks).

    Returns:
        send_indices: Local indices of atoms to send, ordered by dest rank.
            Shape: (sum(send_counts),).
    """
    if not gp_utils.initialized():
        return torch.empty(0, dtype=torch.long, device=gp_ctx.node_partition.device)

    device = gp_ctx.node_partition.device

    # Sort needed_atoms by source rank so the buffer is contiguous
    # per destination rank (required for all_to_all_single split sizes).
    sort_order = gp_ctx.needed_from_ranks.argsort()
    needed_sorted = gp_ctx.needed_atoms[sort_order]

    # recv_counts = how many atoms we need FROM each rank (our send buffer)
    # send_counts = how many atoms each rank needs FROM us (our recv buffer)
    send_splits = gp_ctx.recv_counts.tolist()
    recv_splits = gp_ctx.send_counts.tolist()
    total_recv = int(gp_ctx.send_counts.sum().item())

    recv_buf = torch.empty(total_recv, dtype=torch.long, device=device)

    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)
    if backend == "nccl":
        dist.all_to_all_single(
            recv_buf,
            needed_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=gp_group,
        )
    else:
        # Gloo fallback: pairwise send/recv
        send_list = list(needed_sorted.split(send_splits))
        recv_list = list(recv_buf.split(recv_splits))
        rank = gp_ctx.rank
        world_size = gp_ctx.world_size
        ops = []
        for r in range(world_size):
            if r == rank:
                if send_list[r].numel() > 0:
                    recv_list[r].copy_(send_list[r])
            elif send_list[r].numel() > 0 or recv_list[r].numel() > 0:
                ops.append(dist.P2POp(dist.isend, send_list[r], r, group=gp_group))
                ops.append(dist.P2POp(dist.irecv, recv_list[r], r, group=gp_group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    # recv_buf now contains global indices that other ranks need from us.
    # Convert to local indices.
    send_indices = gp_ctx.global_to_local[recv_buf]

    if not (send_indices >= 0).all():
        raise RuntimeError(
            f"Rank {gp_ctx.rank}: received requests for atoms " f"not in our partition"
        )

    return send_indices


class AllToAllCollect(torch.autograd.Function):
    """
    Autograd function that uses all-to-all to collect only the needed
    remote atom embeddings, replacing the all-gather approach.

    Forward: Sends local atom embeddings to ranks that need them,
    receives remote atom embeddings that we need. Returns a concatenated
    tensor [local_embeddings | received_embeddings].

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
        send_indices: Local indices of atoms to send (from _compute_send_indices).

    Returns:
        x_received: Remote atom embeddings, shape (total_needed, *features).
    """
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
def start_all_to_all_collect_p2p(
    x_local: torch.Tensor,
    gp_ctx: GPContext,
    send_indices: torch.Tensor,
) -> tuple[torch.Tensor, list[dist.Work]]:
    """
    Start async sparse P2P communication for comm-compute overlap.

    Combines the benefits of sparse P2P (fewer NCCL ops) with async
    overlap (hiding latency behind local edge computation). Uses
    ``batch_isend_irecv`` with only non-zero neighbors and returns
    immediately without waiting.

    The caller should do useful compute, then call
    ``finish_all_to_all_collect`` to wait for completion.

    Does NOT participate in autograd — eval-mode only.

    Args:
        x_local: Local atom embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context with precomputed neighbor lists.
        send_indices: Local indices of atoms to send.

    Returns:
        Tuple of (recv_buffer, work_handles):
            recv_buffer: Pre-allocated tensor for received embeddings.
            work_handles: List of dist.Work handles to wait on.
    """
    feature_shape = x_local.shape[1:]
    send_splits = gp_ctx.send_splits
    recv_splits = gp_ctx.recv_splits
    total_recv = gp_ctx.total_recv

    # Gather atoms to send
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

    # Launch sparse P2P ops without waiting
    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)

    work_handles = []
    if backend == "nccl":
        send_chunks = list(x_send.split(send_splits))
        recv_chunks = list(x_recv.split(recv_splits))

        ops = []
        for r in gp_ctx.send_neighbors:
            ops.append(dist.P2POp(dist.isend, send_chunks[r], r, group=gp_group))
        for r in gp_ctx.recv_neighbors:
            ops.append(dist.P2POp(dist.irecv, recv_chunks[r], r, group=gp_group))

        if ops:
            work_handles = dist.batch_isend_irecv(ops)
    else:
        # Gloo fallback: use pairwise send/recv
        send_chunks = list(x_send.split(send_splits))
        recv_chunks = list(x_recv.split(recv_splits))
        _safe_all_to_all(recv_chunks, send_chunks, group=gp_group)

    return x_recv, work_handles


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


def remap_edge_index_to_local(
    edge_index: torch.Tensor,
    global_to_local: torch.Tensor,
) -> torch.Tensor:
    """
    Remap edge_index from global atom indices to local indices
    in the combined [local | received] tensor.

    Args:
        edge_index: Edge index in global space, shape (2, num_edges).
        global_to_local: Mapping from global to local index, shape (total_atoms,).

    Returns:
        Edge index in local space, shape (2, num_edges).
    """
    return global_to_local[edge_index]
