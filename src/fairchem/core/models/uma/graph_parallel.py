"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import distributed as dist

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
    # Precomputed Python lists to avoid repeated .tolist() in AllToAllCollect
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None
    total_recv: int | None = None


def partition_atoms_spatial(
    pos: torch.Tensor,
    num_ranks: int,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    Spatial partitioning via recursive coordinate bisection.

    Recursively halves the atom set along the longest axis of each
    sub-group until the desired number of partitions is reached.
    Guarantees balanced partition sizes (differ by at most 1) and
    runs in O(N log P) time with no iterative refinement.

    Falls back to the longest-axis sort-and-split for non-power-of-2
    ranks, which is O(N log N).

    All computation is done on CPU to avoid GPU→CPU sync points
    from .argmax().item() calls in the recursive bisection (up to
    P-1 sync points eliminated). The position tensor is small
    (N x 3 floats, ~768KB at 64k atoms), so the transfer is fast.

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

    # Move to CPU to avoid GPU->CPU sync points in recursive bisection.
    # Each recursion level calls .argmax().item() which forces a sync.
    # For P=64, that's 63 sync points (~10-20us each = 0.6-1.2ms).
    # CPU sort on 64k atoms takes <0.1ms, so this is a net win.
    pos_cpu = pos.detach().cpu()
    assignments = torch.zeros(N, dtype=torch.long)

    # Recursive coordinate bisection: O(N log P), no iterations.
    # Handles both power-of-2 and non-power-of-2 rank counts via
    # uneven splits (right_half = num_parts - left_half).
    _recursive_bisect(pos_cpu, assignments, torch.arange(N), 0, num_ranks)

    return assignments.to(device)


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


def build_gp_context(
    edge_index: torch.Tensor,
    rank_assignments: torch.Tensor,
    rank: int,
    world_size: int,
) -> GPContext:
    """
    Build the GP context from edge connectivity and atom assignments.

    Determines which non-local atoms this rank needs (edge sources from
    other ranks), exchanges atom indices via a single fused all-to-all,
    and computes all communication metadata.

    Uses a single padded all-to-all collective (instead of the previous
    2-step approach of count exchange + index exchange) by padding atom
    index lists to a fixed size per rank. This halves the number of
    collective operations in the setup path.

    Args:
        edge_index: Full graph edge index, shape (2, num_edges).
            Row 0 = source, row 1 = target.
        rank_assignments: Rank assignment for each atom, shape (total_atoms,).
        rank: This rank's GP rank.
        world_size: GP world size.

    Returns:
        GPContext with all metadata needed for all-to-all communication.
    """
    total_atoms = rank_assignments.shape[0]
    device = rank_assignments.device

    # Atoms owned by this rank
    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    total_local_atoms = node_partition.shape[0]

    # Find which non-local atoms this rank needs as edge sources.
    # An edge (src, tgt) with tgt in our partition needs src's embedding.
    # If src is NOT in our partition, we need to receive it.
    local_mask = rank_assignments == rank  # (total_atoms,) bool
    tgt_is_local = local_mask[edge_index[1]]  # edges whose target is local
    src_is_remote = ~local_mask[edge_index[0]]  # edges whose source is remote

    # Remote sources needed for local targets
    # Use boolean mask + nonzero instead of .unique(sorted=True) on raw
    # edge sources — O(N) scatter + scan vs O(E log E) sort.
    remote_edges_mask = tgt_is_local & src_is_remote
    needed_mask = torch.zeros(total_atoms, dtype=torch.bool, device=device)
    needed_mask[edge_index[0, remote_edges_mask]] = True
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

    # Fused count + index exchange: single padded all-to-all replaces
    # the old 2-step approach (count exchange + index exchange).
    send_counts, send_indices_global = _fused_index_exchange(
        needed_atoms=needed_atoms,
        needed_from_ranks=needed_from_ranks,
        recv_counts=recv_counts,
        rank=rank,
        world_size=world_size,
        total_atoms=total_atoms,
        device=device,
    )

    # Build global_to_local mapping:
    # Local atoms: index 0..total_local_atoms-1 (in order of node_partition)
    # Received atoms: index total_local_atoms..total_local_atoms+total_needed_atoms-1
    global_to_local = torch.full((total_atoms,), -1, dtype=torch.long, device=device)
    # Map local atoms
    global_to_local[node_partition] = torch.arange(
        total_local_atoms, dtype=torch.long, device=device
    )
    # Map needed remote atoms
    global_to_local[needed_atoms] = torch.arange(
        total_local_atoms,
        total_local_atoms + total_needed_atoms,
        dtype=torch.long,
        device=device,
    )

    # Convert send_indices from global to local
    send_indices = None
    if send_indices_global is not None and send_indices_global.numel() > 0:
        send_indices = global_to_local[send_indices_global]
        if not (send_indices >= 0).all():
            raise RuntimeError(
                f"Rank {rank}: received requests for atoms not in our partition"
            )
    elif send_indices_global is not None:
        send_indices = torch.empty(0, dtype=torch.long, device=device)

    # Precompute edge_index_local
    edge_index_local = global_to_local[edge_index]

    # Classify edges: fully-local (both endpoints local) vs boundary
    # (source is remote). Used for communication-computation overlap.
    src_is_local = edge_index_local[0] < total_local_atoms
    tgt_is_local_edge = edge_index_local[1] < total_local_atoms
    local_edge_mask = src_is_local & tgt_is_local_edge

    # Batch scalar extractions to minimize GPU→CPU sync points:
    # move counts to CPU in one transfer instead of per-element syncs.
    send_recv_cpu = torch.stack([send_counts, recv_counts]).cpu()
    send_splits = send_recv_cpu[0].tolist()
    recv_splits = send_recv_cpu[1].tolist()
    total_recv = sum(recv_splits)
    num_local_edges = int(local_edge_mask.sum().cpu().item())
    num_boundary_edges = edge_index_local.shape[1] - num_local_edges

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
        # Precompute Python lists once (avoids .tolist() per layer per forward)
        send_splits=send_splits,
        recv_splits=recv_splits,
        total_recv=total_recv,
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
