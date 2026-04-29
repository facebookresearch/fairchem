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
                output_list[r].copy_(input_list[r])
            else:
                # Send our data to rank r
                ops.append(dist.P2POp(dist.isend, input_list[r], r, group=group))
                # Receive data from rank r
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
    Spatial partitioning using k-means-style clustering.

    Divides atoms into num_ranks groups by spatial proximity.
    Uses up to num_iters iterations with early convergence check.

    Args:
        pos: Atom positions, shape (N, 3).
        num_ranks: Number of partitions (GP world size).
        num_iters: Maximum number of k-means iterations. Default 10.

    Returns:
        rank_assignments: Tensor of shape (N,) with rank index for each atom.
    """
    N = pos.shape[0]
    device = pos.device

    if num_ranks == 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    if num_ranks >= N:
        # Edge case: fewer atoms than ranks
        assignments = torch.arange(N, dtype=torch.long, device=device)
        return assignments

    # Initialize centroids by evenly sampling from sorted positions
    # Sort along longest axis for better initial seeding
    ranges = pos.max(dim=0)[0] - pos.min(dim=0)[0]
    longest_axis = ranges.argmax().item()
    sorted_indices = pos[:, longest_axis].argsort()
    step = N // num_ranks
    centroid_indices = sorted_indices[step // 2 :: step][:num_ranks]
    centroids = pos[centroid_indices].clone()  # (num_ranks, 3)

    # K-means iterations with early convergence
    prev_assignments = None
    for _ in range(num_iters):
        # Assign each atom to nearest centroid
        # Use chunked distance computation to avoid O(N * num_ranks) memory
        chunk_size = max(1, min(65536, N))
        assignments = torch.empty(N, dtype=torch.long, device=device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # (chunk, 3) vs (num_ranks, 3) -> (chunk, num_ranks)
            dists = torch.cdist(pos[start:end], centroids)
            assignments[start:end] = dists.argmin(dim=1)

        # Early convergence check
        if prev_assignments is not None and torch.equal(assignments, prev_assignments):
            break
        prev_assignments = assignments.clone()

        # Update centroids
        for k in range(num_ranks):
            mask = assignments == k
            if mask.any():
                centroids[k] = pos[mask].mean(dim=0)

    # Balance: ensure each rank has at least one atom and roughly equal counts
    assignments = _balance_assignments(assignments, num_ranks, N, pos, centroids)

    return assignments


def _balance_assignments(
    assignments: torch.Tensor,
    num_ranks: int,
    total_atoms: int,
    pos: torch.Tensor | None = None,
    centroids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Post-process assignments to ensure balanced partition sizes.

    Moves atoms from overloaded ranks to underloaded ranks.
    When positions and centroids are provided, preferentially moves
    atoms that are closest to the destination centroid (preserving
    spatial locality).

    Target size per rank is total_atoms // num_ranks (± 1).
    """
    target_size = total_atoms // num_ranks
    remainder = total_atoms % num_ranks

    # Count atoms per rank
    counts = torch.zeros(num_ranks, dtype=torch.long, device=assignments.device)
    for k in range(num_ranks):
        counts[k] = (assignments == k).sum()

    # Target sizes: first `remainder` ranks get target_size + 1
    target_sizes = torch.full(
        (num_ranks,), target_size, dtype=torch.long, device=assignments.device
    )
    target_sizes[:remainder] += 1

    # Iterative rebalancing: move atoms from over-full to under-full
    for _ in range(num_ranks):
        over = (counts - target_sizes).clamp(min=0)
        under = (target_sizes - counts).clamp(min=0)
        if over.sum() == 0:
            break

        # Find most overloaded and most underloaded
        src_rank = over.argmax().item()
        dst_rank = under.argmax().item()
        if over[src_rank] == 0 or under[dst_rank] == 0:
            break

        # Move atoms
        n_move = min(over[src_rank].item(), under[dst_rank].item())
        src_atoms = (assignments == src_rank).nonzero(as_tuple=True)[0]

        if pos is not None and centroids is not None:
            # Move atoms closest to the destination centroid
            # to preserve spatial locality
            dists_to_dst = torch.cdist(
                pos[src_atoms].unsqueeze(0),
                centroids[dst_rank].unsqueeze(0).unsqueeze(0),
            ).squeeze()
            _, closest_order = dists_to_dst.sort()
            atoms_to_move = src_atoms[closest_order[:n_move]]
        else:
            atoms_to_move = src_atoms[:n_move]

        assignments[atoms_to_move] = dst_rank
        counts[src_rank] -= n_move
        counts[dst_rank] += n_move

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


def build_gp_context(
    edge_index: torch.Tensor,
    rank_assignments: torch.Tensor,
    rank: int,
    world_size: int,
) -> GPContext:
    """
    Build the GP context from edge connectivity and atom assignments.

    Determines which non-local atoms this rank needs (edge sources from
    other ranks), and computes send/recv counts for all-to-all.

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
    remote_edges_mask = tgt_is_local & src_is_remote
    needed_atoms_raw = edge_index[0, remote_edges_mask]
    needed_atoms = needed_atoms_raw.unique(sorted=True)

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

    # Compute send_counts: how many atoms we send to each rank.
    # We need to know what OTHER ranks need from us. This requires communication.
    # Use all_to_all on the counts themselves.
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    if gp_utils.initialized():
        # Exchange counts via all_to_all
        send_counts_list = list(send_counts.reshape(world_size, 1).split(1))
        recv_counts_list = list(recv_counts.reshape(world_size, 1).split(1))
        _safe_all_to_all(
            send_counts_list,
            recv_counts_list,
            group=gp_utils.get_gp_group(),
        )
        send_counts = torch.cat(send_counts_list).squeeze(1)

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

    # Precompute edge_index_local
    edge_index_local = global_to_local[edge_index]

    # Classify edges: fully-local (both endpoints local) vs boundary
    # (source is remote). Used for communication-computation overlap.
    src_is_local = edge_index_local[0] < total_local_atoms
    tgt_is_local = edge_index_local[1] < total_local_atoms
    local_edge_mask = src_is_local & tgt_is_local
    num_local_edges = int(local_edge_mask.sum().item())
    num_boundary_edges = edge_index_local.shape[1] - num_local_edges

    ctx = GPContext(
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
        edge_index_local=edge_index_local,
        local_edge_mask=local_edge_mask,
        num_local_edges=num_local_edges,
        num_boundary_edges=num_boundary_edges,
        # Precompute Python lists once (avoids .tolist() per layer per forward)
        send_splits=send_counts.tolist(),
        recv_splits=recv_counts.tolist(),
        total_recv=int(recv_counts.sum().item()),
    )

    # Precompute send_indices if distributed is initialized.
    # This avoids an all-to-all index exchange on every forward pass.
    if gp_utils.initialized():
        ctx.send_indices = _compute_send_indices(ctx)

    return ctx


def _compute_send_indices(
    gp_ctx: GPContext,
) -> torch.Tensor:
    """
    Compute which of our local atoms need to be sent to other ranks,
    and in what order (sorted by destination rank).

    This requires knowing what other ranks need from us.
    We exchange the needed atom indices via all-to-all.

    Returns:
        send_indices: Local indices of atoms to send, ordered by dest rank.
            Shape: (sum(send_counts),).
    """
    if not gp_utils.initialized():
        return torch.empty(0, dtype=torch.long, device=gp_ctx.node_partition.device)

    device = gp_ctx.node_partition.device
    world_size = gp_ctx.world_size

    # Prepare what we need: for each rank r, the global indices of atoms
    # we need from r. We send these indices TO rank r so it knows what to send us.
    send_idx_lists = []
    recv_idx_lists = []

    for r in range(world_size):
        if r == gp_ctx.rank:
            # Self: no communication needed
            send_idx_lists.append(torch.empty(0, dtype=torch.long, device=device))
            recv_idx_lists.append(torch.empty(0, dtype=torch.long, device=device))
        else:
            # We need to tell rank r which atoms we need from it
            mask = gp_ctx.needed_from_ranks == r
            needed_from_r = gp_ctx.needed_atoms[mask.nonzero(as_tuple=True)[0]]
            send_idx_lists.append(needed_from_r)

            # We'll receive from rank r the indices it needs from us
            recv_count = gp_ctx.send_counts[r].item()
            recv_idx_lists.append(
                torch.empty(recv_count, dtype=torch.long, device=device)
            )

    # Exchange indices via all_to_all
    _safe_all_to_all(
        recv_idx_lists,
        send_idx_lists,
        group=gp_utils.get_gp_group(),
    )

    # Now recv_idx_lists[r] contains the global indices that rank r needs from us.
    # Convert to local indices and concatenate in rank order.
    send_indices_parts = []
    for r in range(world_size):
        if r != gp_ctx.rank and recv_idx_lists[r].numel() > 0:
            # Convert global indices to local indices
            local_idxs = gp_ctx.global_to_local[recv_idx_lists[r]]
            assert (local_idxs >= 0).all(), (
                f"Rank {gp_ctx.rank}: rank {r} requested atoms " f"not in our partition"
            )
            send_indices_parts.append(local_idxs)

    if send_indices_parts:
        return torch.cat(send_indices_parts)
    return torch.empty(0, dtype=torch.long, device=device)


class AllToAllCollect(torch.autograd.Function):
    """
    Autograd function that uses all-to-all to collect only the needed
    remote atom embeddings, replacing the all-gather approach.

    Forward: Sends local atom embeddings to ranks that need them,
    receives remote atom embeddings that we need. Returns a concatenated
    tensor [local_embeddings | received_embeddings].

    Backward: Reverses the communication — sends gradient of received
    embeddings back to their owners, receives gradient of sent embeddings.
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

        # Split send tensor by destination rank
        send_list = list(x_send.split(send_splits))
        recv_list = list(x_recv.split(recv_splits))

        # Perform all-to-all
        _safe_all_to_all(recv_list, send_list, group=gp_group)

        # Concatenate received parts
        if recv_list:
            x_received = torch.cat(recv_list, dim=0)
        else:
            x_received = torch.empty(
                0, *feature_shape, device=x_local.device, dtype=x_local.dtype
            )

        return x_received

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
        bwd_send_splits = recv_counts.tolist()
        bwd_recv_splits = send_counts.tolist()

        total_bwd_recv = sum(bwd_recv_splits)
        grad_send_back = torch.empty(
            total_bwd_recv,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        # Split grad_received by source rank (same order as recv in forward)
        bwd_send_list = list(grad_received.split(bwd_send_splits))
        bwd_recv_list = list(grad_send_back.split(bwd_recv_splits))

        # Reverse all-to-all
        _safe_all_to_all(bwd_recv_list, bwd_send_list, group=gp_group)

        # Scatter received gradients back to local positions
        grad_local = torch.zeros(
            local_size,
            *feature_shape,
            device=grad_received.device,
            dtype=grad_received.dtype,
        )

        if total_bwd_recv > 0:
            grad_from_others = torch.cat(bwd_recv_list, dim=0)
            # Accumulate gradients at the send_indices positions
            grad_local.index_add_(0, send_indices, grad_from_others)

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

    send_list = list(x_send.split(send_splits))
    recv_list = list(x_recv.split(recv_splits))

    # Launch async all-to-all (NCCL only)
    gp_group = gp_utils.get_gp_group()
    backend = dist.get_backend(gp_group)

    work_handles = []
    if backend == "nccl":
        # NCCL supports async all-to-all
        work = dist.all_to_all(recv_list, send_list, group=gp_group, async_op=True)
        work_handles.append(work)
    else:
        # Gloo fallback: use pairwise send/recv (already async via batch_isend_irecv)
        rank = dist.get_rank(gp_group)
        world_size = dist.get_world_size(gp_group)
        ops = []
        for r in range(world_size):
            if r == rank:
                recv_list[r].copy_(send_list[r])
            else:
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
