"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
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
            (sources of edges whose targets are in this rank's partition).
        needed_from_ranks: For each atom in needed_atoms, which rank owns it.
        send_counts: Number of atoms to send to each rank. Shape: (world_size,).
        recv_counts: Number of atoms to receive from each rank.
            Shape: (world_size,).
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
    # Precomputed Python lists to avoid repeated .tolist() in AllToAllCollect
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None
    total_recv: int | None = None
    # Precomputed integer indices for local/remote edges (for
    # comm-compute overlap).  Local edges have source atoms owned by
    # this rank (edge_index_local[0] < total_local_atoms), remote
    # edges have sources from other ranks.  Using integer indices
    # instead of boolean masks for compile-friendly indexing (avoids
    # dynamic-shape boolean masking in compiled graphs).
    local_edge_idx: torch.Tensor | None = None
    remote_edge_idx: torch.Tensor | None = None


def _sparse_index_exchange(
    needed_atoms: torch.Tensor,
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
            this rank's partition, shape (2, num_edges).
            Row 0 = source, row 1 = target.
        rank_assignments: Rank assignment for each atom,
            shape (total_atoms,).
        rank: This rank's GP rank.
        world_size: GP world size.
        send_info: Pre-computed send/recv metadata from graph filtering.
            If provided, must contain:
            - send_counts: Tensor of shape (world_size,) with count of
              atoms to send to each rank.
            - send_indices_global: Tensor of global atom indices to send,
              sorted by destination rank.
            When provided, _sparse_index_exchange is skipped.
        node_partition: Pre-computed atom indices in this rank's
            partition. If provided, avoids recomputing from
            rank_assignments.

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
    if send_info is not None:
        # Pre-computed during graph filtering — use it.
        send_counts = send_info["send_counts"]
        send_indices_global = send_info["send_indices_global"]
    else:
        with record_function("a2a_sparse_index_exchange"):
            send_counts, send_indices_global = _sparse_index_exchange(
                needed_atoms=needed_atoms,
                recv_counts=recv_counts,
                rank=rank,
                world_size=world_size,
                device=device,
            )

    # Build global_to_local mapping:
    # Local atoms: index 0..total_local_atoms-1 (in order of
    # node_partition)
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

    # Batch ALL GPU→CPU scalar extractions into a single transfer.
    # This batches send_counts, recv_counts, AND validation scalars
    # into ONE .cpu() call, eliminating extra GPU→CPU syncs.
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
    all_cpu = torch.cat([send_counts, recv_counts, bad_edge_count, send_valid]).cpu()
    send_splits = all_cpu[:world_size].tolist()
    recv_splits = all_cpu[world_size : 2 * world_size].tolist()
    total_recv = sum(recv_splits)
    n_bad = int(all_cpu[2 * world_size].item())
    send_ok = int(all_cpu[2 * world_size + 1].item())

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
            f"Rank {rank}: GP DIAGNOSTIC — "
            f"{n_bad} entries in edge_index_local are -1. "
            f"edge_needed={edge_needed_count}, "
            f"needed_atoms_count={total_needed_atoms}, "
            f"missing_from_needed={missing_count}, "
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
                f"not in needed_atoms): {missing_indices.tolist()}"
            )
        raise RuntimeError(
            f"Rank {rank}: edge_index has {n_bad} endpoints not in "
            f"global_to_local mapping. This indicates a mismatch "
            f"between graph edges and partition assignments."
        )

    # Precompute local/remote edge indices for comm-compute overlap.
    # An edge is "local-source" if its source atom is owned by this
    # rank (index < total_local_atoms in the remapped edge_index_local).
    local_edge_mask = edge_index_local[0] < total_local_atoms
    local_edge_idx = local_edge_mask.nonzero(as_tuple=True)[0]
    remote_edge_idx = (~local_edge_mask).nonzero(as_tuple=True)[0]

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
        # Precompute Python lists once (avoids .tolist() per layer per fwd)
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
    which is a registered PyTorch op — torch.compile can trace through
    it WITHOUT creating a graph break. This eliminates the per-layer
    graph break from the ``@torch.compiler.disable`` on
    ``AllToAllCollect.forward()``.

    This function does NOT support autograd — gradients will not flow
    through the communication. When gradients are needed (e.g., autograd
    forces via ``torch.autograd.grad(energy, pos)``), use
    ``all_to_all_collect`` instead, which uses an autograd.Function
    with proper backward support.

    NOTE: ``all_to_all_single_autograd`` (the funcoll autograd variant)
    crashes with torch.compile because it doesn't handle symbolic split
    sizes (SymInt). Both BL (all-gather) and A2A have a graph break
    when autograd is needed, so this is not a regression vs baseline.

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
        x_received: Remote atom embeddings,
            shape (total_needed, *features).
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
