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
    #print("results dict: ", results)
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


def compilable_scatter_on_dictionary(
    src: dict[str, torch.Tensor],
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> dict[str, torch.Tensor]:
    """
    Scatter function for dictionary of tensors with compile support.
    """
    out = {}
    for key in src:
        out[key] = compilable_scatter(
            src[key], index, dim_size, dim=dim, reduce=reduce
        )
    return out


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

    q = q.squeeze(-1) if q.dim() > 1 else q


    # Get source and neighbor indices
    src, nbr = src_index[0], src_index[1]  # Avoid tuple unpacking

    # Get pairwise distances for each edge
    rij = dist_pairwise[src, nbr]  # (N, max_neighbors)
    # Get charges for each pair
    qi = q[src]  # (N, max_neighbors)
    qj = q[nbr]  # (N, max_neighbors)

    # Create mask using element-wise operations (more compile-friendly)
    mask = (src != nbr) & (rij > eps)
    
    # Pre-allocate output tensor
    e_ij = torch.zeros_like(rij)
    
    
    # Convergence function (error function, as in Ewald)
    if use_convergence:
        convergence_func = torch.special.erf(rij / (sigma * 1.4142135623730951))
    else:
        convergence_func = torch.ones_like(rij)

    # Compute pairwise Coulomb energy with epsilon shift and correct prefactor
    coulomb_term = (qi * qj) / (rij + epsilon) / twopi / 2.0 * convergence_func
    e_ij = torch.where(mask, coulomb_term, torch.zeros_like(coulomb_term))

    
    # To avoid double-counting, sum only upper triangle or divide by 2
    energy = e_ij.sum(dim=-1) * 90.0474  
    return energy


def heisenberg_energy_from_src_index(
    q: torch.Tensor,  # shape: (N, 2) 
    src_index: torch.Tensor,  # shape: (2, N, max_neighbors)
    j_coupling_nn: torch.nn.Module,  
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
    src, nbr = src_index[0], src_index[1]  # Avoid tuple unpacking

    # Get pairwise distances for each edge
    rij = dist_pairwise[src, nbr]  # (N, max_neighbors)
    
    # for batching NN on all rij pairs
    N, max_neighbors = rij.shape
    rij_flat = rij.contiguous().view(N * max_neighbors, 1)
    #rij_flatten = rij.view(-1).unsqueeze(1)
    j_coupling_vals = j_coupling_nn(rij_flat)
    j_coupling_vals = j_coupling_vals.reshape(N, max_neighbors)
    
    # Get charges for each pair
    qi = q[src]  # (N, max_neighbors, 2)
    qj = q[nbr]  # (N, max_neighbors, 2)
    
    # Extract alpha and beta components
    qi_alpha = qi[:, :, 0]  # (N, max_neighbors)
    qi_beta = qi[:, :, 1]   # (N, max_neighbors)
    qj_alpha = qj[:, :, 0]  # (N, max_neighbors)
    qj_beta = qj[:, :, 1]   # (N, max_neighbors)
    
    # Mask out self-interactions (where src == nbr or rij == 0)
    mask = (src != nbr) & (rij > eps)

    # compute pairwise Heisenberg energy
    e_ij = torch.zeros_like(rij)

    # J_ij * (S_i . S_j = S_i^alpha * S_j^beta + S_j^alpha * S_i^beta)
    spin_interaction = (qi_alpha * qj_beta + qj_alpha * qi_beta) * j_coupling_vals
    e_ij = torch.where(mask, spin_interaction, torch.zeros_like(spin_interaction))

    #e_ij[mask] = (qi_alpha[mask] * qj_beta[mask] + qj_alpha[mask] * qi_beta[mask]) * j_coupling_vals[mask]
    
    energy = e_ij.sum(axis=1)
    #print("Heisenberg energy shape: ", energy.shape)

    return energy


def charge_spin_renormalization(
        q: torch.Tensor, 
        emb: dict[str, torch.Tensor], 
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor: 
    """
    Rescale the predicted charges to match the target total charge per graph.
        Args:
        q: predicted charges, shape (N, 2)
        emb: dictionary containing graph data with keys:
            - "data": a GraphAttentionData object with attributes:
                - num_nodes: total number of nodes (N)
                - num_graphs: total number of graphs in the batch
                - node_batch: tensor of shape (N,) indicating graph membership of each node
                - charge: tensor of shape (num_graphs,) indicating target total charge per graph
        eps: small value to avoid division by zero
        weights: tensor of shape (N, 2) indicating weights for alpha and beta components
    Returns:
        rescaled charges, shape (N, 2)
    """
    # Extract necessary data from emb
    num_nodes = emb["data"].num_nodes
    num_graphs = emb["data"].num_graphs
    node_batch = emb["data"].node_batch
    
    # Rescale charges to match target total charge per graph
    results_tensor = torch.zeros_like(q)

    valid_charges = q[:num_nodes]
    valid_node_batch = node_batch[:num_nodes]
    target_charges = emb["data"].charge[:num_graphs]
    target_spins = emb["data"].spin[:num_graphs]
   
    alpha = valid_charges[:, 0]
    beta = valid_charges[:, 1]

    
    if weights is None:
        weights = torch.ones_like(valid_charges)
    
    w_alpha = weights[:, 0]
    w_beta  = weights[:, 1]
    ones_arr = torch.ones_like(w_alpha)

    
    scatter_dict = compilable_scatter_on_dictionary(
        {
            "alpha": alpha, 
            "beta": beta, 
            "w_alpha": w_alpha, 
            "w_beta": w_beta,
            "ones": ones_arr
        },
        index=valid_node_batch,
        dim_size=emb["data"].num_graphs,
        dim=0,
        reduce="sum"
    )

    q_sum = scatter_dict["alpha"] + scatter_dict["beta"]        # total charge
    s_sum = scatter_dict["alpha"] - scatter_dict["beta"]        # total spin

    dq = target_charges - q_sum
    ds = target_spins - s_sum
    
    # residuals / batch for each constraint also normalized by weights
    delta_alpha = (0.5 * (dq + ds))
    delta_beta  = (0.5 * (dq - ds))
    #print("delta_alpha: ", delta_alpha)
    #print("delta_beta: ", delta_beta)

    # expand out to nodes
    delta_alpha_expanded = delta_alpha[valid_node_batch] 
    delta_beta_expanded  = delta_beta[valid_node_batch]
    # divide by items per graph and weights
    delta_alpha_expanded = delta_alpha_expanded / (scatter_dict["w_alpha"][valid_node_batch] + 1e-8)
    delta_beta_expanded  = delta_beta_expanded  / (scatter_dict["w_beta"][valid_node_batch] + 1e-8)
    #print("shapes: ", delta_alpha.shape, delta_alpha_expanded.shape, valid_node_batch.shape)

    alpha_corr = alpha + delta_alpha_expanded
    beta_corr  = beta + delta_beta_expanded
    
    results_tensor[:num_nodes, 0] = alpha_corr.squeeze(-1)
    results_tensor[:num_nodes, 1] = beta_corr.squeeze(-1)
    
    return results_tensor


def charge_renormalization(q: torch.Tensor, emb: dict[str, torch.Tensor], eps: float = 1e-8,) -> torch.Tensor: 
    """
    Rescale the predicted charges to match the target total charge per graph.
    Args:
        q: predicted charges, shape (N, 1)
        emb: dictionary containing graph data with keys:
            - "data": a GraphAttentionData object with attributes:
                - num_nodes: total number of nodes (N)
                - num_graphs: total number of graphs in the batch
                - node_batch: tensor of shape (N,) indicating graph membership of each node
                - charge: tensor of shape (num_graphs,) indicating target total charge per graph
        eps: small value to avoid division by zero
    Returns:
        rescaled charges, shape (N, 1)
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

    # Add small epsilon only where needed to avoid division by zero
    rescale_factor = torch.where(
        torch.abs(global_charges) < eps,
        torch.ones_like(global_charges),
        target_charges / global_charges
    )

    flattened_charges_raw[:num_nodes] *= rescale_factor[valid_node_batch]

    return flattened_charges_raw

