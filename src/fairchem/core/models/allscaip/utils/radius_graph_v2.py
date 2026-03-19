"""
Modified from MinDScAIP: Minimally biased Differentiable Scaled Attention Interatomic Potential
Credit: Ryan Liu
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData

from fairchem.core.models.escaip.utils.radius_graph import (
    hard_rank,
    safe_norm,
    soft_rank,
)


def segment_argsort(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    Compute argsort within each segment.

    Args:
        values: (E,) values to sort
        segment_ids: (E,) segment ID for each value (must be sorted!)
        num_segments: total number of segments

    Returns:
        ranks: (E,) rank of each value within its segment (0-indexed)
    """
    E = values.size(0)
    device = values.device

    # Add large offset per segment to make global sort work per-segment
    # offset[i] = segment_ids[i] * max_possible_value
    max_val = values.max() - values.min() + 1
    offset_values = values + segment_ids.float() * max_val

    # Global argsort
    global_order = torch.argsort(offset_values)

    # Compute rank within segment using scatter
    ranks = torch.zeros(E, device=device, dtype=torch.long)

    # Count elements per segment up to each position
    # For each position in sorted order, its rank is its position within segment
    segment_counts = torch.zeros(num_segments, device=device, dtype=torch.long)
    sorted_segments = segment_ids[global_order]

    # Compute cumulative count per segment
    for i in range(E):
        seg = sorted_segments[i].item()
        ranks[global_order[i]] = segment_counts[seg]
        segment_counts[seg] += 1

    return ranks


def compute_soft_ranks_single_segment(
    dist_padded: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
    delta: int,
    sigmoid_scale: float,
    use_low_mem: bool,
) -> torch.Tensor:
    """
    Compute soft ranks for a single padded segment.
    vmap-compatible: uses only out-of-place operations.

    Args:
        dist_padded: (max_seg_size,) padded distances
        valid_mask: (max_seg_size,) bool mask for valid entries
        k, delta, sigmoid_scale: ranking parameters
        use_low_mem: whether to use low memory mode

    Returns:
        ranks: (max_seg_size,) ranks (inf for invalid entries)
    """
    max_size = dist_padded.size(0)
    kd = k + delta

    # Set invalid distances to inf so they sort to the end
    dist_masked = torch.where(
        valid_mask, dist_padded, torch.full_like(dist_padded, float("inf"))
    )

    if use_low_mem:
        # Sort and only consider k+delta nearest
        sorted_dist, sort_idx = torch.sort(dist_masked)

        # Compute soft ranks for top kd (handle case where max_size < kd)
        actual_kd = min(kd, max_size)
        top_dist = sorted_dist[:actual_kd]
        diff = top_dist[:, None] - top_dist[None, :]
        top_ranks = bump_function(diff / sigmoid_scale).sum(dim=-1)

        # Create full-size ranks with inf for entries beyond kd
        # Note: sorted_dist entries beyond actual valid count are inf, so they get sorted to end
        full_sorted_ranks = torch.full(
            (max_size,),
            float("inf"),
            device=dist_padded.device,
            dtype=dist_padded.dtype,
        )
        # We can't use indexing, so we use where with position mask
        pos_mask = torch.arange(max_size, device=dist_padded.device) < actual_kd
        full_sorted_ranks = torch.where(
            pos_mask,
            torch.cat(
                [
                    top_ranks,
                    torch.zeros(
                        max_size - actual_kd,
                        device=dist_padded.device,
                        dtype=dist_padded.dtype,
                    ),
                ]
            ),
            full_sorted_ranks,
        )

        # Map back to original order using inverse permutation
        inv_sort_idx = torch.argsort(sort_idx)
        ranks = full_sorted_ranks[inv_sort_idx]

        # Mask out invalid entries
        ranks = torch.where(valid_mask, ranks, torch.full_like(ranks, float("inf")))
    else:
        # Full soft ranking
        diff = dist_masked[:, None] - dist_masked[None, :]
        ranks = torch.sigmoid(diff / sigmoid_scale).sum(dim=-1)
        ranks = torch.where(valid_mask, ranks, torch.full_like(ranks, float("inf")))

    return ranks


def segment_argsort_vectorized(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    Vectorized version of segment argsort using cumsum trick.

    Args:
        values: (E,) values to sort
        segment_ids: (E,) segment ID for each value
        num_segments: total number of segments

    Returns:
        ranks: (E,) rank of each value within its segment (0-indexed)
    """
    E = values.size(0)
    device = values.device

    # Add large offset per segment
    max_val = values.max() - values.min() + 1
    offset_values = values + segment_ids.to(values.dtype) * max_val

    # Global argsort
    global_order = torch.argsort(offset_values)
    sorted_segments = segment_ids[global_order]

    # Create position-in-segment using cumsum
    # When segment changes, reset counter; otherwise increment
    segment_change = torch.cat(
        [
            torch.ones(1, device=device, dtype=torch.long),
            (sorted_segments[1:] != sorted_segments[:-1]).long(),
        ]
    )

    # Cumsum gives position, but resets at segment boundaries
    cumsum = torch.cumsum(torch.ones(E, device=device, dtype=torch.long), dim=0)
    # Subtract the cumsum value at segment start
    # segment_starts = cumsum * segment_change
    segment_start_values = torch.zeros(E, device=device, dtype=torch.long)
    segment_start_values[segment_change.bool()] = cumsum[segment_change.bool()]

    # Forward fill segment start values
    segment_start_filled = torch.cummax(segment_start_values, dim=0)[0]

    # Rank within segment = cumsum - segment_start
    sorted_ranks = cumsum - segment_start_filled

    # Map back to original order
    ranks = torch.zeros(E, device=device, dtype=torch.long)
    ranks[global_order] = sorted_ranks

    return ranks


def bump_function(x: torch.Tensor) -> torch.Tensor:
    """
    Bump function for the low memory soft knn. Designed such that the behavior
    matches sigmoid for small x and the step function for large x. Since torch.where
    propagates gradients, we need to mask the input tensor x.
    Args:
        x: the input tensor
    Returns:
        y: the bump function value
    """
    mask = x.abs() < 4
    step = torch.where(x < 0, 0.0, 1.0)
    x = x.div(4.0).masked_fill_(~mask, 0)
    bump = torch.exp(-2.0 / (x + 1)) / (
        torch.exp(-2.0 / (x + 1)) + torch.exp(-2.0 / (1 - x))
    )
    return torch.where(mask, bump, step)


def soft_rank_low_mem(
    dist: torch.Tensor,
    k: int,
    scale: float,
    delta: int = 20,
) -> torch.Tensor:
    """
    calculate the soft rankings for the soft knn. Approximate with low memory by
    truncating the distance matrix to be [0, k + delta]. This is not exact but is a good
    approximation. It is valid when the difference of distance at k+delta and k is
    larger than pi * scale.
    Args:
        dist: the pairwise distance tensor
        k: the number of neighbors
        scale: the scale factor for the shifted sine function (Å).
        delta: the delta factor for the truncation
    Returns:
        ranks: the soft rankings
    """
    sorted_dist, indicies = torch.sort(dist, dim=-1)
    ranks_T = bump_function(
        (sorted_dist[:, : k + delta, None] - sorted_dist[:, None, : k + delta]) / scale
    ).sum(dim=-1)
    ranks = torch.full_like(dist, torch.inf)
    ranks[
        torch.arange(dist.size(0), device=dist.device)[:, None],
        indicies[:, : k + delta],
    ] = ranks_T
    return ranks


def build_radius_graph_chunked(
    pos: torch.Tensor,
    cell: torch.Tensor,
    image_id: torch.Tensor,
    cutoff: float,
    start_index: int,
    device: torch.device,
    k: int = 30,
    soft: bool = False,
    sigmoid_scale: float = 0.2,
    lse_scale: float = 0.1,
    use_low_mem: bool = False,
    delta: int = 20,
    compute_dist_pairwise: bool = True,
    chunk_size: int = 512,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Chunked version of build_radius_graph to reduce peak memory.

    This processes the NxNxM distance computation in chunks of source atoms,
    computing rankings correctly by:
    1. Phase 1: For each source chunk, compute distances and src_ranks for ALL edges
    2. Phase 2: Group collected edges by destination and compute dst_ranks
    3. Phase 3: Compute envelope and filter edges

    Memory is reduced from O(N²M) to O(chunk_size x N x M) for the main tensor.
    """
    N = pos.size(0)
    M = image_id.size(0)

    # Precompute image offsets
    image_offsets = torch.mm(image_id, cell)  # (M, 3)

    # Phase 1: Process in chunks, using vmap within each chunk
    # Chunking is necessary for memory - we can't materialize full NxNxM at once

    # Precompute destination positions with all images: (N, M, 3)
    dst_pos_all = pos[:, None, :] + image_offsets[None, :, :]  # (N, M, 3)

    def compute_distances_single_source(src_pos_single):
        """Compute distances from one source to all N*M destinations - vmap compatible."""
        disp_single = dst_pos_all - src_pos_single[None, None, :]  # (N, M, 3)
        dist_single = safe_norm(disp_single, dim=-1)  # (N, M)
        return disp_single, dist_single

    vmapped_distances = torch.vmap(compute_distances_single_source)

    all_src_idx = []
    all_dst_idx = []
    all_disp = []
    all_dist = []
    all_src_ranks = []
    dist_pairwise_chunks = [] if compute_dist_pairwise else None

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_N = chunk_end - chunk_start
        pos_chunk = pos[chunk_start:chunk_end]  # (chunk_N, 3)

        # Use vmap to compute distances for all sources in chunk
        disp_chunk, dist_chunk = vmapped_distances(
            pos_chunk
        )  # (chunk_N, N, M, 3), (chunk_N, N, M)

        if compute_dist_pairwise:
            dist_pairwise_chunks.append(dist_chunk.min(dim=2)[0])

        # Compute src_ranks for this chunk
        dist_flat = dist_chunk.view(chunk_N, N * M)
        if soft:
            if use_low_mem:
                src_ranks_chunk = soft_rank_low_mem(
                    dist_flat, k, sigmoid_scale, delta
                ).view(chunk_N, N, M)
            else:
                src_ranks_chunk = soft_rank(dist_flat, sigmoid_scale).view(
                    chunk_N, N, M
                )
        else:
            src_ranks_chunk = hard_rank(dist_flat).view(chunk_N, N, M)

        # Filter and collect edges
        potential_mask = (dist_chunk < cutoff) | (dist_chunk == 0)
        i_local, j_global, img = torch.where(potential_mask)
        i_global = i_local + chunk_start

        all_src_idx.append(i_global)
        all_dst_idx.append(j_global)
        all_disp.append(disp_chunk[i_local, j_global, img])
        all_dist.append(dist_chunk[i_local, j_global, img])
        all_src_ranks.append(src_ranks_chunk[i_local, j_global, img])

        del disp_chunk, dist_chunk, dist_flat, src_ranks_chunk

    # Concatenate phase 1 results
    src_idx = torch.cat(all_src_idx)
    dst_idx = torch.cat(all_dst_idx)
    disp = torch.cat(all_disp)
    dist = torch.cat(all_dist)
    src_ranks = torch.cat(all_src_ranks)
    num_edges = src_idx.size(0)

    # Phase 2: Compute dst_ranks using vectorized segment operations
    # Sort edges by destination for segment-wise operations
    dst_sort_order = torch.argsort(dst_idx)
    sorted_dst_idx = dst_idx[dst_sort_order]
    sorted_dist = dist[dst_sort_order]

    if soft:
        # For soft ranking, use vmap over padded segments
        # Get segment boundaries
        unique_dsts, counts = torch.unique_consecutive(
            sorted_dst_idx, return_counts=True
        )
        num_segments = len(unique_dsts)
        segment_ends = torch.cumsum(counts, dim=0)
        segment_starts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), segment_ends[:-1]]
        )

        # Use actual max segment size - can't truncate or we lose edges
        max_seg_size = int(counts.max().item())

        # Create padded distance tensor and validity mask
        padded_dists = torch.full(
            (num_segments, max_seg_size), float("inf"), device=device, dtype=dist.dtype
        )
        valid_masks = torch.zeros(
            (num_segments, max_seg_size), device=device, dtype=torch.bool
        )

        for seg_idx in range(num_segments):
            start = segment_starts[seg_idx].item()
            end = segment_ends[seg_idx].item()
            seg_len = end - start
            padded_dists[seg_idx, :seg_len] = sorted_dist[start:end]
            valid_masks[seg_idx, :seg_len] = True

        # Define vmap function
        def compute_ranks_for_segment(dist_seg, mask_seg):
            return compute_soft_ranks_single_segment(
                dist_seg, mask_seg, k, delta, sigmoid_scale, use_low_mem
            )

        # Apply vmap over all segments
        vmapped_compute = torch.vmap(compute_ranks_for_segment)
        all_ranks = vmapped_compute(
            padded_dists, valid_masks
        )  # (num_segments, max_seg_size)

        # Scatter results back to original positions
        dst_ranks = torch.full(
            (num_edges,), float("inf"), device=device, dtype=dist.dtype
        )
        for seg_idx in range(num_segments):
            start = segment_starts[seg_idx].item()
            end = segment_ends[seg_idx].item()
            seg_len = end - start
            dst_ranks[dst_sort_order[start:end]] = all_ranks[seg_idx, :seg_len]
    else:
        # Hard ranking: fully vectorized using segment_argsort
        sorted_ranks = segment_argsort_vectorized(sorted_dist, sorted_dst_idx, N)
        dst_ranks = torch.zeros(num_edges, device=device, dtype=dist.dtype)
        dst_ranks[dst_sort_order] = sorted_ranks.to(dist.dtype)

    # Phase 3: Compute envelope and filter
    env_src = src_ranks / k
    env_dst = dst_ranks / k
    env_dist = dist / cutoff

    if soft:
        env_stack = torch.stack([env_src, env_dst, env_dist], dim=0)
        env = lse_scale * torch.logsumexp(env_stack / lse_scale, dim=0)
    else:
        env = torch.maximum(torch.maximum(env_src, env_dst), env_dist)

    # Set envelope to 0 for self-loops (shouldn't exist but safety check)
    env = torch.where(dist == 0, torch.zeros_like(env), env)

    # Filter by envelope
    final_mask = env < 1.0
    src_idx = src_idx[final_mask]
    dst_idx = dst_idx[final_mask]
    disp = disp[final_mask]
    env = env[final_mask]

    # Compute index rankings using vectorized segment argsort
    num_final_edges = src_idx.size(0)

    # index1_rank: rank of each edge among source's outgoing edges (by envelope)
    src_sort_order = torch.argsort(src_idx)
    sorted_src_idx = src_idx[src_sort_order]
    sorted_env_src = env[src_sort_order]
    sorted_index1_rank = segment_argsort_vectorized(sorted_env_src, sorted_src_idx, N)
    index1_rank = torch.zeros(num_final_edges, device=device, dtype=torch.long)
    index1_rank[src_sort_order] = sorted_index1_rank

    # index2_rank: rank of each edge among destination's incoming edges (by envelope)
    dst_sort_order = torch.argsort(dst_idx)
    sorted_dst_idx = dst_idx[dst_sort_order]
    sorted_env_dst = env[dst_sort_order]
    sorted_index2_rank = segment_argsort_vectorized(sorted_env_dst, sorted_dst_idx, N)
    index2_rank = torch.zeros(num_final_edges, device=device, dtype=torch.long)
    index2_rank[dst_sort_order] = sorted_index2_rank

    if compute_dist_pairwise:
        dist_pairwise = torch.cat(dist_pairwise_chunks, dim=0)
    else:
        dist_pairwise = None

    index1 = src_idx + start_index
    index2 = dst_idx + start_index

    return (
        index1,
        index2,
        index1_rank,
        index2_rank,
        disp,
        env,
        dist_pairwise,
    )


def build_radius_graph(
    pos: torch.Tensor,
    cell: torch.Tensor,
    image_id: torch.Tensor,
    cutoff: float,
    start_index: int,
    device: torch.device,
    k: int = 30,
    soft: bool = False,
    sigmoid_scale: float = 0.2,
    lse_scale: float = 0.1,
    use_low_mem: bool = False,
    delta: int = 20,
    compute_dist_pairwise: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    construct the biknn radius graph for one system.
    Args:
        pos: the atomic positions tensor
        cell: the cell tensor for the periodic boundary condition
        image_id: the image identifier for different PBC images
        cutoff: the cutoff distance in Angstrom
        start_index: the starting index of the system in the batch
        device: the device on which the tensors are allocated
        k: the number of neighbors
        soft: the flag for the soft knn
        sigmoid_scale: the scale factor for the sigmoid function
        lse_scale: the scale factor for the log-sum-exp function
        use_low_mem: the flag for the low memory soft knn
        delta: the delta factor for the low memory soft knn
    Returns:
        index1: the source index of the neighbors
        index2: the destination index of the neighbors
        index1_rank: the rank of the edge in source neighbors by envelope function
        index2_rank: the rank of the edge in destination neighbors by envelope function
        disp: the displacement vector of the neighbors
        env: the envelope vector of the neighbors
    """
    N = pos.size(0)
    M = image_id.size(0)
    # calculate the displacements while taking into account the PBC
    src_pos = pos[:, None] + torch.mm(image_id, cell)[None, :]
    disp = src_pos[None, :, :, :] - pos[:, None, None, :]
    dist = safe_norm(disp, dim=-1)
    dist_T = dist.transpose(0, 1).contiguous()
    # get the pairwise distance between all atoms (only if needed)
    # pairwise minimum-image distance matrix for this system (N, N)
    dist_pairwise = dist.min(dim=2)[0] if compute_dist_pairwise else None
    # compute the rankings, depending on the soft or hard knn
    if soft:
        # calculate the rankings in a soft manner
        if use_low_mem:
            # use low memory soft knn
            src_ranks = soft_rank_low_mem(
                dist.view(N, N * M), k, sigmoid_scale, delta
            ).view(N, N, M)
            dst_ranks = (
                soft_rank_low_mem(dist_T.view(N, N * M), k, sigmoid_scale, delta)
                .view(N, N, M)
                .transpose(0, 1)
            )
        else:
            # use full soft knn
            src_ranks = soft_rank(dist.view(N, N * M), sigmoid_scale).view(N, N, M)
            dst_ranks = (
                (soft_rank(dist_T.view(N, N * M), sigmoid_scale))
                .view(N, N, M)
                .transpose(0, 1)
            )
        # env is the soft maximum of the source and destination rankings and the
        # distance normalized by the radius cutoff.
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = lse_scale * torch.logsumexp(env / lse_scale, dim=0)
    else:
        # calculate the rankings in a hard manner
        src_ranks = hard_rank((dist).view(N, N * M)).view(N, N, M)
        dst_ranks = hard_rank((dist_T).view(N, N * M)).view(N, N, M).transpose(0, 1)
        # env is the hard maximum of the source and destination rankings and the
        # distance normalized by the radius cutoff.
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = torch.amax(env, dim=0)
    # set the envelope to zero for self-loops
    env.masked_fill_(dist == 0.0, 0.0)
    # sort the distances of source and destintion neighbors
    index = torch.arange(N, device=device)[:, None]
    ranks = torch.arange(M * N, device=device)[None, :]
    # ranks are the ranks of the atoms within the neighbors, i.e. the j-th source
    # neighbor of i-th atom should have index1 of i and index1_rank of j.
    index1_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    src_argsort = torch.argsort(env.view(N, N * M), dim=1)
    index1_rank[index, src_argsort // M, src_argsort % M] = ranks
    index2_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    dst_argsort = torch.argsort(env.transpose(0, 1).reshape(N, N * M), dim=1)
    index2_rank[dst_argsort // M, index, dst_argsort % M] = ranks
    assert (index1_rank >= 0).all(), "Index1_rank contains negative indices."
    assert (index2_rank >= 0).all(), "Index2_rank contains negative indices."
    # compute the mask of the neighbors
    mask = env < 1.0
    # select the neighbors within the cutoff
    index1, index2, index3 = torch.where(mask)
    index1_rank = index1_rank[index1, index2, index3]
    index2_rank = index2_rank[index1, index2, index3]
    disp = disp[index1, index2, index3]
    env = env[index1, index2, index3]
    # add the start index
    index1 = index1 + start_index
    index2 = index2 + start_index
    return (
        index1,
        index2,
        index1_rank,
        index2_rank,
        disp,
        env,
        dist_pairwise,
    )


def batched_radius_graph(
    pos_list: list[torch.Tensor],
    cell_list: list[torch.Tensor],
    image_id_list: list[torch.Tensor],
    num_atoms: int,
    max_atoms: int | None,
    slices: list[int],
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    knn_use_low_mem: bool,
    knn_pad_size: int | None,
    cutoff: float,
    device: torch.device,
    compute_dist_pairwise: bool = True,
    use_chunked: bool = False,
    chunk_size: int = 512,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    calculate the biknn radius graph for the batch of systems
    Args:
        pos_list: the list of atomic positions tensors
        anum_list: the list of atomic number tensors
        cell_list: the list of cell tensors
        image_id_list: the list of image identifier tensors
        num_atoms: the number of atoms in the batch
        max_atoms: the size to pad to for the number of atoms
        slices: the slices index for the batch
        knn_params: the parameters for the knn algorithm
        cutoff: the cutoff distance in Angstrom
        device: the device on which the tensors are allocated
        use_chunked: whether to use chunked graph construction
        chunk_size: size of chunks for chunked graph construction
    Returns:
        padded_disp: the padded displacement tensor
        src_env: the source envelope tensor
        dst_env: the destination envelope tensor
        src_index: the destination layout to source layout index tensor
        dst_index: the source layout to destination layout index tensor
        edge_index: the edge index tensor
    """
    # Choose build function based on chunking mode
    if use_chunked:
        build_fn = build_radius_graph_chunked
    else:
        build_fn = build_radius_graph

    # build the biknn radius graph for each system
    results = [
        build_fn(
            pos,
            cell,
            image_id,
            cutoff,
            start_idx,
            device,
            knn_k,
            knn_soft,
            knn_sigmoid_scale,
            knn_lse_scale,
            knn_use_low_mem,
            compute_dist_pairwise=compute_dist_pairwise,
            **({"chunk_size": chunk_size} if use_chunked else {}),
        )
        for pos, cell, image_id, start_idx in zip(
            pos_list, cell_list, image_id_list, slices
        )
    ]
    # Unzip the per-system results
    (
        index1_list,
        index2_list,
        index1_rank_list,
        index2_rank_list,
        disp_list,
        env_list,
        dist_blocks,
    ) = zip(*results)

    # Concatenate tensors that share the same size across systems
    index1 = torch.cat(index1_list)
    index2 = torch.cat(index2_list)
    index1_rank = torch.cat(index1_rank_list)
    index2_rank = torch.cat(index2_rank_list)
    disp = torch.cat(disp_list)
    env = torch.cat(env_list)

    # Assemble a block-diagonal pairwise distance matrix (N, N) only if computed
    if compute_dist_pairwise:
        dist_pairwise = torch.block_diag(*dist_blocks)
    else:
        dist_pairwise = None

    # if soft knn, pad the tensors with the maximum number of neighbors.
    if knn_pad_size is None:
        knn_pad_size = int(max(index1_rank.max().item(), index2_rank.max().item())) + 1
    # flag if the number of neighbors is larger than the maximum number of neighbors
    elif (index1_rank >= knn_pad_size).any() or (index2_rank >= knn_pad_size).any():
        warnings.warn(
            "The number of neighbors is larger than the maximum number of neighbors."
            "Removing the excess neighbors."
        )
        # filter the neighbors to the maximum number of neighbors
        mask = (index1_rank < knn_pad_size) & (index2_rank < knn_pad_size)
        index1 = index1[mask]
        index2 = index2[mask]
        index1_rank = index1_rank[mask]
        index2_rank = index2_rank[mask]
        disp = disp[mask]
        env = env[mask]

    # determine the number of atoms in the batch
    if max_atoms is None:
        max_atoms = num_atoms

    # assertions to ensure no index-out-of-range errors
    assert (index1 >= 0).all(), "Index1 contains negative indices."
    assert (index2 >= 0).all(), "Index2 contains negative indices."
    assert (index1 < max_atoms).all(), "Index1 contains indices larger than max_atoms."
    assert (index2 < max_atoms).all(), "Index2 contains indices larger than max_atoms."
    assert (index1_rank >= 0).all(), "Index1_rank contains negative indices."
    assert (index2_rank >= 0).all(), "Index2_rank contains negative indices."
    assert (
        index1_rank < knn_pad_size
    ).all(), "Index1_rank contains indices larger than max_neighbors."
    assert (
        index2_rank < knn_pad_size
    ).all(), "Index2_rank contains indices larger than max_neighbors."

    # initialize the padded tensors
    padded_index = (
        torch.arange(max_atoms, device=device)
        .view(-1, 1)
        .expand(max_atoms, knn_pad_size)
    )
    padded_rank = (
        torch.arange(knn_pad_size, device=device)
        .view(1, -1)
        .expand(max_atoms, knn_pad_size)
    )
    padded_disp = torch.zeros((max_atoms, knn_pad_size, 3), device=device)
    src_env = torch.full((max_atoms, knn_pad_size), torch.inf, device=device)
    dst_env = torch.full((max_atoms, knn_pad_size), torch.inf, device=device)
    edge_index = torch.stack([padded_index, padded_index], dim=0)
    src_index = torch.stack([padded_index, padded_rank], dim=0)
    dst_index = torch.stack([padded_index, padded_rank], dim=0)
    # fill the padded tensors
    padded_disp[index1, index1_rank] = disp
    src_env[index1, index1_rank] = env
    dst_env[index2, index2_rank] = env
    edge_index[0, index1, index1_rank] = index1
    edge_index[1, index1, index1_rank] = index2
    # the flipping index for switching between in source neighbors layout and
    # destination neighbors layout. Since index1 is the source atom's index and index2
    # is the destination atom's index, the edge that were placed at
    # [index1, index1_rank] in source neighbors layout should be placed at
    # [index2, index2_rank]
    src_index[0, index1, index1_rank] = index2
    src_index[1, index1, index1_rank] = index2_rank
    dst_index[0, index2, index2_rank] = index1
    dst_index[1, index2, index2_rank] = index1_rank

    # removing infinite envelopes for the padded atoms
    if num_atoms < max_atoms:
        src_env[num_atoms:] = 0
        dst_env[num_atoms:] = 0

    return (
        dist_pairwise,
        padded_disp,
        src_env,
        dst_env,
        src_index,
        dst_index,
        edge_index,
    )


def biknn_radius_graph(
    data: AtomicData,
    cutoff: float,
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    knn_use_low_mem: bool,
    knn_pad_size: int | None,
    device: torch.device,
    max_atoms: int | None = None,
    compute_dist_pairwise: bool = True,
    use_chunked: bool = False,
    chunk_size: int = 512,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    function to construct the biknn radius graph for the batch of systems. This function
    calculates the number of images to be included in the PBC and constructs the
    image identifier list and call the batched_radius_graph function to perform the
    construction.
    Args:
        data: the `torch_geometric.data.Data` object containing the atomic information
        cutoff: the cutoff distance in Angstrom
        knn_params: the parameters for the knn algorithm
        device: the device on which the tensors are allocated
        max_atoms: the size to pad to for the number of atoms
        max_neighbors: the size to pad to for the number of neighbors
        compute_dist_pairwise: whether to compute and return dist_pairwise (NxN matrix)
        use_chunked: whether to use chunked graph construction
        chunk_size: size of chunks for chunked graph construction
    Returns:
        dist_pairwise: the pairwise distance matrix (None if compute_dist_pairwise=False)
        padded_disp: the padded displacement tensor
        src_env: the source envelope tensor
        dst_env: the destination envelope tensor
        src_index: the destination layout to source layout index tensor
        dst_index: the source layout to destination layout index tensor
        edge_index: the edge index tensor
    """
    # pad the data if max_atoms is not None
    slices, cumsum, cat_dims, natoms_list = data.get_batch_stats()
    if slices is None or cumsum is None or cat_dims is None or natoms_list is None:
        # Unbatched single-system input (e.g., from ASE calculator): synthesize stats.
        n = data.pos.shape[0]
        natoms_list = [n]
        slices = {"pos": torch.tensor([0, n], device=data.pos.device)}

    pos_list: list[torch.Tensor] = list(torch.split(data.pos, natoms_list, dim=0))
    num_graphs = len(natoms_list)

    # Check if PBC is used at all - if not, skip expensive PBC calculations
    pbc_any = data.pbc.any().item()

    if pbc_any:
        # if PBC is used, construct the image identifier list by including all images within
        # the cutoff distance. Adopted from FairChem repository
        cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
        cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

        inv_min_dist_a1 = safe_norm(cross_a2a3 / cell_vol, dim=-1)
        rep_a1 = torch.ceil(cutoff * inv_min_dist_a1)

        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = safe_norm(cross_a3a1 / cell_vol, dim=-1)
        rep_a2 = torch.ceil(cutoff * inv_min_dist_a2)

        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = safe_norm(cross_a1a2 / cell_vol, dim=-1)
        rep_a3 = torch.ceil(cutoff * inv_min_dist_a3)

        rep_a1 = rep_a1.masked_fill(data.pbc[:, 0] == 0, 0).tolist()
        rep_a2 = rep_a2.masked_fill(data.pbc[:, 1] == 0, 0).tolist()
        rep_a3 = rep_a3.masked_fill(data.pbc[:, 2] == 0, 0).tolist()

        image_id_list: list[torch.Tensor] = [
            torch.cartesian_prod(
                *[
                    torch.arange(
                        -rep,
                        rep + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    )
                    for rep in reps
                ]
            )
            for reps in zip(rep_a1, rep_a2, rep_a3)
        ]
    else:
        # No PBC - use identity image only (no replications needed)
        identity_image = torch.zeros(
            (1, 3), device=device, dtype=torch.get_default_dtype()
        )
        image_id_list = [identity_image for _ in range(num_graphs)]

    cell_list: list[torch.Tensor] = list(data.cell)

    # call to the batched_radius_graph function to perform per-system biknn radius
    # graph construction.
    return batched_radius_graph(
        pos_list,
        cell_list,
        image_id_list,
        data.num_nodes,
        max_atoms,
        slices["pos"],
        knn_k,
        knn_soft,
        knn_sigmoid_scale,
        knn_lse_scale,
        knn_use_low_mem,
        knn_pad_size,
        cutoff,
        device,
        compute_dist_pairwise,
        use_chunked,
        chunk_size,
    )
