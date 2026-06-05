"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            (sources of edges whose targets are in this rank's partition),
            sorted by source rank.
        needed_from_ranks: For each atom in needed_atoms, which rank owns it
            (sorted to match needed_atoms ordering).
        send_counts: Number of atoms to send to each rank. Shape: (world_size,).
        recv_counts: Number of atoms to receive from each rank.
            Shape: (world_size,).
        global_to_local: Mapping from global atom index to position in the
            local concatenated tensor [local_atoms | received_atoms].
            Shape: (total_atoms,), with -1 for atoms not accessible.
        total_local_atoms: Number of atoms in this rank's partition.
        total_needed_atoms: Total atoms needed from other ranks.
        send_indices: Local indices of atoms to send, ordered by
            destination rank.
        edge_index_local: Edge index remapped to local indices.
        send_splits: Per-rank split sizes for the embedding send buffer.
        recv_splits: Per-rank split sizes for the embedding recv buffer.
        total_recv: Total number of embeddings to receive (sum of recv_splits).
        local_edge_idx: Indices into edge_index_local where source is a local atom.
        remote_edge_idx: Indices into edge_index_local where source is a remote atom.
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
    send_indices: torch.Tensor
    edge_index_local: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    total_recv: int
    local_edge_idx: torch.Tensor
    remote_edge_idx: torch.Tensor


def _sparse_index_exchange(
    needed_atoms: torch.Tensor,
    recv_counts: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Variable-split index exchange using two small all-to-alls.

    Step 1: Exchange recv_counts to get send_counts (P ints).
    Step 2: Exchange actual atom indices with variable split sizes.

    Sends only the exact number of indices needed (no padding),
    keeping communication volume minimal: O(sum of needed counts).

    We considered a single-allgather alternative that packs counts +
    padded indices into one buffer and calls all_gather once, but it
    sends O(P * natoms_per_rank) regardless of sparsity. Benchmarks
    on H200 at GP=64 across 8 nodes (256K atoms, UMA-S) showed:
      2x A2A (this fn):   0.593 ns/day  (+22.8% vs baseline AG-GP)
      1x all-gather:      0.584 ns/day  (+20.9% vs baseline AG-GP)
    The 2x A2A is ~1.5% faster because the variable-split second
    A2A transfers less data when each rank only needs atoms from
    nearby neighbors, not all P ranks.

    Args:
        needed_atoms: Global indices of atoms this rank needs,
            pre-sorted by source rank (done by the caller).
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
            torch.empty(0, dtype=torch.long, device=device),
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
    # needed_atoms is already sorted by source rank (done by the
    # caller in build_gp_context), so use it directly as send buffer.
    if needed_atoms.numel() > 0:
        send_buf = needed_atoms.contiguous()
    else:
        send_buf = torch.empty(0, dtype=torch.long, device=device)

    # Batch send_counts and recv_counts into a single GPU→CPU transfer.
    # This eliminates 2 extra GPU→CPU syncs vs separate .tolist() calls.
    counts_cpu = torch.stack([send_counts, recv_counts]).cpu()
    recv_splits = counts_cpu[0].tolist()  # what we recv = what we need
    send_splits = counts_cpu[1].tolist()  # what we send = what others need
    total_recv_indices = sum(recv_splits)
    recv_buf = torch.empty(total_recv_indices, dtype=torch.long, device=device)

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


@torch.compiler.disable
def build_gp_context(
    edge_index: torch.Tensor,
    rank_assignments: torch.Tensor,
    rank: int,
    world_size: int,
    node_partition: torch.Tensor | None = None,
) -> GPContext:
    """
    Build the GP context from edge connectivity and atom assignments.

    Args:
        edge_index: Edge index filtered to this rank's partition,
            shape (2, num_edges). Row 0 = source, row 1 = target.
        rank_assignments: Rank owner for each atom,
            shape (total_atoms,).
        rank: This rank's GP rank.
        world_size: GP world size.
        node_partition: Pre-computed local atom indices.

    Returns:
        GPContext with all communication metadata.
    """
    total_atoms = rank_assignments.shape[0]
    device = rank_assignments.device

    if node_partition is None:
        node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    total_local_atoms = node_partition.shape[0]

    # Find remote atoms needed as edge sources.
    remote_mask = rank_assignments != rank
    src_is_remote = remote_mask[edge_index[0]]
    needed_mask = torch.zeros(total_atoms, dtype=torch.bool, device=device)
    needed_mask[edge_index[0, src_is_remote]] = True
    needed_atoms = needed_mask.nonzero(as_tuple=True)[0]

    total_needed_atoms = needed_atoms.shape[0]
    needed_from_ranks = rank_assignments[needed_atoms]

    if total_needed_atoms > 0:
        recv_counts = torch.bincount(needed_from_ranks, minlength=world_size).to(
            dtype=torch.long, device=device
        )
    else:
        recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    recv_counts[rank] = 0

    # Sort needed_atoms by source rank to match A2A recv_buf ordering.
    sort_order = needed_from_ranks.argsort(stable=True)
    needed_atoms = needed_atoms[sort_order]
    needed_from_ranks_sorted = needed_from_ranks[sort_order]

    # Exchange send metadata via collective.
    with record_function("a2a_sparse_index_exchange"):
        send_counts, send_indices_global = _sparse_index_exchange(
            needed_atoms=needed_atoms,
            recv_counts=recv_counts,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    # Build global-to-local index mapping.
    # Local atoms: [0, total_local_atoms)
    # Remote atoms: [total_local_atoms, total_local_atoms + needed)
    global_to_local = torch.full((total_atoms,), -1, dtype=torch.long, device=device)
    global_to_local[node_partition] = torch.arange(
        total_local_atoms, dtype=torch.long, device=device
    )
    global_to_local[needed_atoms] = torch.arange(
        total_local_atoms,
        total_local_atoms + total_needed_atoms,
        dtype=torch.long,
        device=device,
    )

    # Convert send_indices from global to local.
    if send_indices_global.numel() > 0:
        send_indices = global_to_local[send_indices_global]
    else:
        send_indices = torch.empty(0, dtype=torch.long, device=device)

    # Remap edge_index to local indices.
    edge_index_local = global_to_local[edge_index]

    # Single GPU-to-CPU transfer for send/recv splits.
    splits_cpu = torch.stack([send_counts, recv_counts]).cpu()
    send_splits = splits_cpu[0].tolist()
    recv_splits = splits_cpu[1].tolist()
    total_recv = sum(recv_splits)

    # Validate mappings (async — no device-to-host sync).
    torch._assert_async(
        ~(edge_index_local < 0).any(),
        "edge_index_local has negative entries — graph edges "
        "reference atoms not in global_to_local mapping.",
    )
    if send_indices.numel() > 0:
        torch._assert_async(
            ~((send_indices < 0) | (send_indices >= total_local_atoms)).any(),
            "send_indices out of bounds — remote rank requested "
            "atoms not in our partition.",
        )

    # Precompute local/remote edge indices for comm-compute overlap.
    local_edge_mask = edge_index_local[0] < total_local_atoms
    local_edge_idx = local_edge_mask.nonzero(as_tuple=True)[0]
    remote_edge_idx = (~local_edge_mask).nonzero(as_tuple=True)[0]

    return GPContext(
        rank=rank,
        world_size=world_size,
        node_partition=node_partition,
        rank_assignments=rank_assignments,
        needed_atoms=needed_atoms,
        needed_from_ranks=needed_from_ranks_sorted,
        send_counts=send_counts,
        recv_counts=recv_counts,
        global_to_local=global_to_local,
        total_local_atoms=total_local_atoms,
        total_needed_atoms=total_needed_atoms,
        send_indices=send_indices,
        edge_index_local=edge_index_local,
        send_splits=send_splits,
        recv_splits=recv_splits,
        total_recv=total_recv,
        local_edge_idx=local_edge_idx,
        remote_edge_idx=remote_edge_idx,
    )


class AllToAllCollect(torch.autograd.Function):
    """
    Autograd function that uses all-to-all to collect only the needed
    remote atom embeddings, replacing the all-gather approach.

    Forward: Sends local atom embeddings to ranks that need them,
    receives remote atom embeddings that we need. Returns only the
    received remote embeddings (NOT concatenated with local).

    Backward: Reverses the communication — sends gradient of received
    embeddings back to their owners, receives gradient of sent
    embeddings.

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
        Forward all-to-all embedding collection.

        Args:
            x_local: Local atom embeddings,
                shape (local_atoms, *feature_dims).
            send_indices: Local indices of atoms to send,
                ordered by dest rank.
            send_counts: Number of atoms to send to each rank.
            recv_counts: Number of atoms to receive from each rank.
            gp_group: GP process group.
            rank: GP rank.
            world_size: GP world size.
            precomputed_send_splits: Optional cached
                send_counts.tolist().
            precomputed_recv_splits: Optional cached
                recv_counts.tolist().
            precomputed_total_recv: Optional cached
                sum(recv_splits).

        Returns:
            Received remote embeddings,
                shape (sum(recv_counts), *feature_dims).
        """
        ctx.send_indices = send_indices
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.gp_group = gp_group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_size = x_local.shape[0]
        # Cache precomputed splits for backward
        ctx.precomputed_send_splits = precomputed_send_splits
        ctx.precomputed_recv_splits = precomputed_recv_splits

        feature_shape = x_local.shape[1:]

        # Gather atoms to send (index_select into contiguous buffer)
        if send_indices.numel() > 0:
            x_send = x_local[send_indices].contiguous()
        else:
            x_send = torch.empty(
                0,
                *feature_shape,
                device=x_local.device,
                dtype=x_local.dtype,
            )

        # Use precomputed splits if available
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

        # Perform all-to-all communication
        backend = dist.get_backend(gp_group)
        if backend == "nccl":
            # Use all_to_all_single for NCCL
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

        # x_recv already contains all received data in rank order
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

        # In backward, the roles are reversed
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
            grad_local.index_add_(0, send_indices, grad_send_back)

        # Return gradients for x_local only; None for all other inputs
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
        x_received: Remote atom embeddings,
            shape (total_needed, *features).
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
