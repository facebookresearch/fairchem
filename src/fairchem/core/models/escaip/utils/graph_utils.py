from __future__ import annotations

import math
import numpy as np 
import torch
import torch.nn.functional as F

from ..custom_types import GraphAttentionData

def pad_batch(
    max_atoms,
    max_batch_size,
    atomic_numbers,
    charge,
    spin,
    edge_direction,
    edge_distance,
    neighbor_index,
    node_batch,
    num_graphs,
    src_mask,
    dst_mask,
    src_index,
    dst_index,
    dist_pairwise,
):
    """
    Pad the batch to have the same number of nodes in total.
    Needed for torch.compile

    Note: the sampler for multi-node training could sample batchs with different number of graphs.
    The number of sampled graphs could be smaller or larger than the batch size.
    This would cause the model to recompile or core dump.
    Temporarily, setting the max number of graphs to be twice the batch size to mitigate this issue.
    TODO: look into a better way to handle this.
    """
    device = atomic_numbers.device
    _, num_nodes, _ = neighbor_index.shape
    pad_size = max_atoms - num_nodes
    assert (
        pad_size >= 0
    ), "Number of nodes exceeds the maximum number of nodes per batch"
    assert (
        max_batch_size >= num_graphs
    ), "Number of graphs exceeds the maximum batch size"

    # pad the features
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    edge_distance = F.pad(edge_distance, (0, 0, 0, pad_size), value=0)
    neighbor_index = F.pad(neighbor_index, (0, 0, 0, pad_size), value=-1) 
    node_batch = F.pad(node_batch, (0, pad_size), value=-1)
    src_mask = F.pad(src_mask, (0, 0, 0, pad_size), value=-torch.inf) # (num_nodes, num_nodes + pad_size)
    dst_mask = F.pad(dst_mask, (0, 0, 0, pad_size), value=-torch.inf) # (num_nodes, num_nodes + pad_size)
    src_index = F.pad(src_index, (0, 0, 0, pad_size), value=-1) # (num_edges,)
    dst_index = F.pad(dst_index, (0, 0, 0, pad_size), value=-1) # (num_edges,)
    dist_pairwise = F.pad(dist_pairwise, (0, pad_size, 0, pad_size), value=0)
    if charge is not None:
        charge = F.pad(charge, (0, max_batch_size - num_graphs), value=0)
    else:
        charge = torch.zeros(max_batch_size, dtype=torch.float, device=device)
    if spin is not None:
        spin = F.pad(spin, (0, max_batch_size - num_graphs), value=0)
    else:
        spin = torch.zeros(max_batch_size, dtype=torch.float, device=device)

    return (
        atomic_numbers,
        charge,
        spin,
        edge_direction,
        edge_distance,
        neighbor_index,
        src_mask,
        dst_mask,
        src_index,
        dst_index,
        dist_pairwise,
        node_batch,
    )


def unpad_result(results: dict, data: GraphAttentionData):
    """
    Unpad the results to remove the padding.
    """
    unpad_results = {}
    for key in results:
        if results[key].shape[0] == data.max_num_nodes:
            # Node-level results
            unpad_results[key] = results[key][: data.num_nodes]
        elif results[key].shape[0] == data.max_batch_size:
            # Graph-level results 
            unpad_results[key] = results[key][: data.num_graphs]
        elif (
            results[key].shape[0] == data.num_nodes
            or results[key].shape[0] == data.num_graphs
        ):
            # Results already unpadded
            unpad_results[key] = results[key]
        else:
            raise ValueError(
                f"Unknown padding mask shape for key '{key}': "
                f"result shape {results[key].shape}, "
                f"data shape {data.num_nodes}, {data.num_graphs}"
            )
    return unpad_results


def compilable_scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    torch_scatter scatter function with compile support.
    Modified from torch_geometric.utils.scatter_.
    """

    def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
        dim = ref.dim() + dim if dim < 0 else dim
        size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
        return src.view(size).expand_as(ref)

    dim = src.dim() + dim if dim < 0 else dim
    size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]

    if reduce == "sum" or reduce == "add":
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == "mean":
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    raise ValueError(f"Invalid reduce option '{reduce}'.")


def get_displacement_and_cell(data, regress_stress, regress_forces, direct_forces):
    """
    Get the displacement and cell from the data.
    For gradient-based forces/stress
    ref: https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/escn_md.py#L298
    """
    displacement = None
    orig_cell = None
    if regress_stress and not direct_forces:
        displacement = torch.zeros(
            (3, 3),
            dtype=data["pos"].dtype,
            device=data["pos"].device,
        )
        num_batch = len(data["natoms"])
        displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
        displacement.requires_grad = True
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        if data["pos"].requires_grad is False:
            data["pos"].requires_grad = True
        data["pos_original"] = data["pos"]
        data["pos"] = data["pos"] + torch.bmm(
            data["pos"].unsqueeze(-2),
            torch.index_select(symmetric_displacement, 0, data["batch"]),
        ).squeeze(-2)

        orig_cell = data["cell"]
        data["cell"] = data["cell"] + torch.bmm(data["cell"], symmetric_displacement)
    if (
        not regress_stress
        and regress_forces
        and not direct_forces
        and data["pos"].requires_grad is False
    ):
        data["pos"].requires_grad = True
    return displacement, orig_cell


def coulomb_energy_from_src_index(
    q: torch.Tensor,  # shape: (N,)
    src_index: torch.Tensor,  # shape: (2, N, max_neighbors)
    dist_pairwise: torch.Tensor,  # shape: (N, N)
    eps: float = 1e-8,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    twopi: float = 2.0 * np.pi,
    use_convergence: bool = False,
) -> torch.Tensor:
    """
    Compute Coulomb energy efficiently using src_index and dist_pairwise,
    with optional convergence function for smooth SR/LR transition.
    Args:
        q: charges, shape (N,)
        src_index: (2, N, max_neighbors), src_index[0] is source, src_index[1] is neighbor index
        dist_pairwise: (N, N) pairwise distance matrix
        eps: small value to avoid division by zero
        sigma: width parameter for the convergence function
        epsilon: shift for denominator to avoid singularity
        twopi: scaling factor for correct Coulomb prefactor
        use_convergence: whether to apply the convergence function (default: True)
    Returns:
        scalar Coulomb energy
    """
    # Get source and neighbor indices
    src, nbr = src_index  # (N, max_neighbors)
    # Get pairwise distances for each edge
    rij = dist_pairwise[src, nbr]  # (N, max_neighbors)
    # Get charges for each pair
    qi = q[src]  # (N, max_neighbors)
    qj = q[nbr]  # (N, max_neighbors)
    # remove last axis of qi and qj if needed
    if qi.dim() == 3:
        qi = qi.squeeze(-1)
    if qj.dim() == 3:
        qj = qj.squeeze(-1)

    # Mask out self-interactions (where src == nbr or rij == 0)
    mask = (src != nbr) & (rij > eps)
    
    # Convergence function (error function, as in Ewald)
    if use_convergence:
        convergence_func = torch.special.erf(rij / (sigma * (2.0 ** 0.5)))
    else:
        convergence_func = torch.ones_like(rij)

    # Compute pairwise Coulomb energy with epsilon shift and correct prefactor
    e_ij = torch.zeros_like(rij)
    # (qi * qj) / (rij + epsilon) / twopi / 2.0 * convergence_func
    e_ij[mask] = (
        qi[mask] * qj[mask] / (rij[mask] + epsilon) / twopi / 2.0 * convergence_func[mask]
    )

    # To avoid double-counting, sum only upper triangle or divide by 2
    energy = e_ij.sum(axis=1) * 90.0474
    return energy


def heisenberg_energy_from_src_index(
    q: torch.Tensor,  # shape: (N, 2) 
    src_index: torch.Tensor,  # shape: (2, N, max_neighbors)
    j_coupling_nn: torch.Tensor,  # shape: (N, )
    dist_pairwise: torch.Tensor,  # shape: (N, N)
    eps: float = 1e-8, 
) -> torch.Tensor:
    """
    Compute Heisenberg exchange energy efficiently using src_index.
    Args:
        s: spins, shape (N, 1) for Ising or (N, 3) for Heisenberg model
        src_index: (2, N, max_neighbors), src_index[0] is source, src_index[1] is neighbor index
        j_coupling_tensor: (N, ) coupling constant for each atom
    """
    # Get source and neighbor indices
    src, nbr = src_index  # (N, max_neighbors)
    # Get pairwise distances for each edge
    rij = dist_pairwise[src, nbr]  # (N, max_neighbors)
    
    # for batching NN on all rij pairs
    dims_rij = rij.shape
    rij_flatten = rij.view(-1).unsqueeze(1)
    j_coupling_vals = j_coupling_nn(rij_flatten)
    j_coupling_vals = j_coupling_vals.reshape(dims_rij)
    
    # Get charges for each pair
    qi = q[src]  # (N, max_neighbors, 2)
    qj = q[nbr]  # (N, max_neighbors, 2)
    qi_alpha = qi[:, :, 0]  # (N, max_neighbors)
    qi_beta = qi[:, :, 1]   # (N, max_neighbors)
    qj_alpha = qj[:, :, 0]  # (N, max_neighbors)
    qj_beta = qj[:, :, 1]   # (N, max_neighbors)
    # Mask out self-interactions (where src == nbr or rij == 0)
    mask = (src != nbr) & (rij > eps)

    # compute pairwise Heisenberg energy
    e_ij = torch.zeros_like(rij)

    # J_ij * (S_i . S_j = S_i^alpha * S_j^beta + S_j^alpha * S_i^beta)
    # e_ij[mask] = j_coupling * (qi_alpha[mask] * qj_beta[mask])
    e_ij[mask] = (qi_alpha[mask] * qj_beta[mask] + qj_alpha[mask] * qi_beta[mask]) * j_coupling_vals[mask]
    energy = e_ij.sum(axis=1)
    #print("Heisenberg energy shape: ", energy.shape)

    return energy


def charge_spin_renormalization(): 
    """
    Placeholder for future charge and spin renormalization functions.
    """
    pass

def charge_renormalization(q: torch.Tensor, emb: dict[str, torch.Tensor], eps: float = 1e-8,) -> torch.Tensor: 
    """
    Placeholder for future charge renormalization functions.
    """
    # Extract necessary data from emb
    num_nodes = emb["data"].num_nodes
    num_graphs = emb["data"].num_graphs
    node_batch = emb["data"].node_batch
    
    # Rescale charges to match target total charge per graph
    flattened_charges_raw = q.squeeze(-1) 
    valid_charges = flattened_charges_raw[:num_nodes]
    valid_node_batch = node_batch[:num_nodes]
    target_charges = emb["data"].charge[:num_graphs]

    global_charges = compilable_scatter(
        valid_charges, 
        index=valid_node_batch,
        dim_size=emb["data"].num_graphs,
        dim=0,
        reduce="sum"
    )
    # Ensure proper device placement and indexing
    target_charges = emb["data"].charge[:num_graphs]

    # Add small epsilon only where needed to avoid division by zero
    rescale_factor = torch.where(
        torch.abs(global_charges) < eps,
        torch.ones_like(global_charges),
        target_charges / global_charges
    )

    flattened_charges_raw[:num_nodes] *= rescale_factor[valid_node_batch]

    return flattened_charges_raw

