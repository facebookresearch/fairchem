"""
Modified from MinDScAIP: Minimally biased Differentiable Scaled Attention Interatomic Potential
Credit: Ryan Liu
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes the norm of a tensor with a small epsilon to avoid division by zero.
    Args:
        x: The input tensor.
        dim: The dimension to reduce.
        keepdim: Whether to keep the reduced dimension.
        eps: The epsilon value.
    Returns:
        The norm of the input tensor.
    """
    vec_norm_sq = x.square().sum(dim=dim, keepdim=keepdim)
    vec_norm = vec_norm_sq.clamp_min(eps).sqrt()
    vec_norm = torch.where(vec_norm_sq <= eps, torch.zeros_like(vec_norm), vec_norm)
    return vec_norm


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes the normalized vector with a small epsilon to avoid division by zero.
    Args:
        x: The input tensor.
        dim: The dimension to reduce.
        eps: The epsilon value.
    Returns:
        The L2 normalized tensor.
    """
    vec_norm_sq = x.square().sum(dim=dim, keepdim=True)
    vec_norm = vec_norm_sq.clamp_min(eps).sqrt()
    norm_vec = torch.where(vec_norm_sq <= eps, torch.zeros_like(x), x / vec_norm)
    return norm_vec


def envelope_fn(
    x: torch.Tensor,
    envelope: bool = True,
) -> torch.Tensor:
    """
    Computes the envelope function in log space that smoothly vanishes to -inf at x = 1.
    Args:
        x: The input tensor.
        envelope: Whether to use the envelope function. Default: True
    Returns:
        The envelope function in log space.
    """
    if envelope:
        env = -x.pow(2) / (1 - x.pow(2))
    else:
        env = torch.zeros_like(x)
    return torch.where(x < 1, env, -torch.inf)


def shifted_sine(x: torch.Tensor) -> torch.Tensor:
    """
    Shifted sine function for the low memory soft knn. Designed such that the behavior
    matches sigmoid for small x and the step function for large x.
    Args:
        x: the input tensor
    Returns:
        y: the shifted sine function value
    """
    return (
        0.5 * torch.where(x.abs() < torch.pi, torch.sin(0.5 * x), torch.sign(x)) + 0.5
    )


def soft_rank(
    dist: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    calculate the soft rankings for the soft knn
    Args:
        dist: the pairwise distance tensor
        scale: the scale factor for the sigmoid function (Å).
    Returns:
        ranks: the soft rankings
    """
    ranks = torch.sigmoid((dist[:, :, None] - dist[:, None, :]) / scale).sum(dim=-1)
    return ranks


def hard_rank(
    dist: torch.Tensor,
) -> torch.Tensor:
    """
    calculate the hard rankings for the hard knn
    Args:
        dist: the pairwise distance tensor
    Returns:
        ranks: the hard rankings
    """
    ranks = torch.empty_like(dist)
    ranks[
        torch.arange(dist.size(0), device=dist.device)[:, None],
        torch.argsort(dist, dim=-1),
    ] = torch.arange(dist.size(-1), device=dist.device, dtype=dist.dtype)
    return ranks


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
    ranks_T = shifted_sine(
        (sorted_dist[:, : k + delta, None] - sorted_dist[:, None, : k + delta]) / scale
    ).sum(dim=-1)
    ranks = torch.full_like(dist, torch.inf)
    ranks[
        torch.arange(dist.size(0), device=dist.device)[:, None],
        indicies[:, : k + delta],
    ] = ranks_T
    return ranks


@torch.jit.script
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
    # compute the rankings, depending on the soft or hard knn
    if soft:
        # calculate the rankings in a soft manner
        if use_low_mem:
            # use low memory soft knn
            src_ranks = soft_rank_low_mem(dist.view(N, N * M), k, sigmoid_scale).view(
                N, N, M
            )
            dst_ranks = (
                soft_rank_low_mem(dist_T.view(N, N * M), k, sigmoid_scale)
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
    index1_rank = torch.empty((N, N, M), device=device, dtype=torch.long)
    src_argsort = torch.argsort(env.view(N, N * M), dim=1)
    index1_rank[index, src_argsort // M, src_argsort % M] = ranks
    index2_rank = torch.empty((N, N, M), device=device, dtype=torch.long)
    dst_argsort = torch.argsort(env.transpose(0, 1).reshape(N, N * M), dim=1)
    index2_rank[dst_argsort // M, index, dst_argsort % M] = ranks
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
    )


def batched_radius_graph(
    pos_list: list[torch.Tensor],
    cell_list: list[torch.Tensor],
    image_id_list: list[torch.Tensor],
    N: int,
    natoms: torch.Tensor,
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    knn_use_low_mem: bool,
    knn_pad_size: int,
    cutoff: float,
    device: torch.device,
) -> tuple[
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
        N: the total number of atoms in the batch
        natoms: the number of atoms in each system
        knn_params: the parameters for the knn algorithm
        cutoff: the cutoff distance in Angstrom
        device: the device on which the tensors are allocated
    Returns:
        padded_disp: the padded displacement tensor
        src_env: the source envelope tensor
        dst_env: the destination envelope tensor
        src_index: the destination layout to source layout index tensor
        dst_index: the source layout to destination layout index tensor
        edge_index: the edge index tensor
    """
    # calculate the starting index of each system
    start_idxs = torch.cumsum(natoms, dim=0) - natoms
    # build the biknn radius graph for each system
    index1, index2, index1_rank, index2_rank, disp, env = map(
        torch.cat,
        zip(
            *[
                build_radius_graph(
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
                )
                for pos, cell, image_id, start_idx in zip(
                    pos_list, cell_list, image_id_list, start_idxs
                )
            ]
        ),
    )
    # if soft knn, pad the tensors with the maximum number of neighbors.
    padsize = knn_pad_size
    # initialize the padded tensors
    padded_disp = torch.zeros((N, padsize, 3), device=device)
    src_env = torch.full((N, padsize), torch.inf, device=device)
    dst_env = torch.full((N, padsize), torch.inf, device=device)
    edge_index = torch.zeros((2, N, padsize), device=device, dtype=torch.long)
    src_index = torch.zeros((2, N, padsize), device=device, dtype=torch.long)
    dst_index = torch.zeros((2, N, padsize), device=device, dtype=torch.long)
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

    return (
        padded_disp,
        src_env,
        dst_env,
        src_index,
        dst_index,
        edge_index,
    )


@torch.compiler.disable(recursive=True)
def biknn_radius_graph(
    data: AtomicData,
    cutoff: float,
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    knn_use_low_mem: bool,
    knn_pad_size: int,
    use_pbc: bool,
    device: torch.device,
) -> tuple[
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
        use_pbc: the flag for the periodic boundary condition
        device: the device on which the tensors are allocated
    Returns:
        padded_disp: the padded displacement tensor
        src_env: the source envelope tensor
        dst_env: the destination envelope tensor
        src_index: the destination layout to source layout index tensor
        dst_index: the source layout to destination layout index tensor
        edge_index: the edge index tensor
    """
    # if PBC is used, construct the image identifier list by including all images within
    # the cutoff distance. Adopted from FairChem repository
    if use_pbc:
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
        rep_a1, rep_a2, rep_a3 = rep_a1.tolist(), rep_a2.tolist(), rep_a3.tolist()
    else:
        rep_a1 = rep_a2 = rep_a3 = [0] * data.natoms.size(0)
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
    pos_list: list[torch.Tensor] = list(torch.split(data.pos, data.natoms.tolist()))
    cell_list: list[torch.Tensor] = list(data.cell)

    # call to the batched_radius_graph function to perform per-system biknn radius
    # graph construction.
    return batched_radius_graph(
        pos_list,
        cell_list,
        image_id_list,
        data.pos.size(0),
        data.natoms,
        knn_k,
        knn_soft,
        knn_sigmoid_scale,
        knn_lse_scale,
        knn_use_low_mem,
        knn_pad_size,
        cutoff,
        device,
    )
