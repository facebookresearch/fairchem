from __future__ import annotations

import numpy as np
import torch
from torch_scatter import scatter, scatter_add
from fairchem.core.models.les.util import grad

def potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    epsilon_factor_les: float = 1.0,  # \epsilon_infty
    twopi: float = 2.0 * np.pi,
    return_bec: bool = False,
    batch: torch.Tensor | None = None,
    conv_function_tf: bool = False,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 1)
        radius_lr: cutoff radius for long-range interactions
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
        twopi: 2 * pi
        max_num_neighbors: maximum number of neighbors for each atom
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    # yields list of interactions [source, target]
    results = {}
    n, d = pos.shape
    assert d == 3, 'r dimension error'
    assert n == q.size(0), 'q dimension error'
    
    if batch is None:
        batch = torch.zeros(n, dtype=torch.int64, device=pos.device)

    unique_batches = torch.unique(batch)  # Get unique batch indices
    

    if return_bec:
        # Ensure pos has requires_grad=True
        if not pos.requires_grad:
            pos.requires_grad_(True)
        
        if not q.requires_grad:
            q.requires_grad_(True)

        normalization_factor = epsilon_factor_les ** 0.5

        all_P = []
        all_phases = [] 
        unique_batches = torch.unique(batch)  # Get unique batch indices
        
        for i in unique_batches:    
            #print("batch: ", batch)
            mask = batch == i  # Create a mask for the i-th configuration
            
            r_now, q_now = pos[mask], q[mask].reshape(-1, 1)  # [n_atoms, 1]
            #print("mask shape: ", mask.shape)
            #print("q_now shape: ", q_now.shape)

            q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)
            polarization = torch.sum(q_now * r_now, dim=0)
            phase = torch.ones_like(r_now, dtype=torch.complex64)

            all_P.append(polarization * normalization_factor)
            all_phases.append(phase)

        P = torch.stack(all_P, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # Ensure P has requires_grad=True
        if not P.requires_grad:
            P.requires_grad_(True)
        
        bec_complex = grad(y=P, x=pos)
        
        # dephase
        result = bec_complex * phases.unsqueeze(1).conj()
        result_bec = result.real
        results["bec"] = result_bec
    
    
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    # red to [n_interactions, 1]
    edge_dist = distance_vec.norm(dim=-1)
    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    
    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    #print("q_source shape: ", q_source.shape, " q_target shape: ", q_target.shape)
    pairwise_potential = q_source * q_target * edge_dist_transformed
    
    if conv_function_tf:
        convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
        pairwise_potential *= convergence_func#.unsqueeze(2)

    
    # remove diagonal elements 
    pairwise_potential = pairwise_potential * (i != j).float()
    
    # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
    # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
    norm_factor = 90.0474
    #print("pairwise_potential shape: ", pairwise_potential.shape)
    results["potential"] = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    ) * norm_factor
    return results


def heisenberg_potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    nn: torch.nn.Module,
    sigma: float = 1.0,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 2)
        nn: neural network to calculate the coupling term
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1).reshape(-1, 1)
    edge_dist.requires_grad_(True)

    convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5)).reshape(-1, 1)
    coupling = nn(edge_dist)

    q_source = q[i]
    q_target = q[j]
    pairwise_potential = q_source * q_target * coupling * convergence_func

    results = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    ).sum(dim=1)

    return results


def batch_spin_charge_renormalization(
    charges_raw: torch.Tensor, # [n_atoms, 2],
    q_total: torch.Tensor, # [n_batches]
    s_total: torch.Tensor, # [n_batches]
    epsilon: float = 1e-12,
    batch: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
):
    """
    Renormalizes the charges and spins for each batch.
    Args:
        charges_raw: raw charges of shape (n_atoms, 2)
        q_total: total charge for each batch of shape (n_batches,)
        s_total: total spin for each batch of shape (n_batches,)
        epsilon: small value to avoid division by zero
    Returns:
        charges_renormalized: renormalized charges of shape (n_atoms, 2)
    """
    B, N = charges_raw.shape

    device = charges_raw.device
    num_batches = q_total.shape[0]    

    alpha = charges_raw[:, 0]
    beta = charges_raw[:, 1]
    # Default weights: uniform per channel
    if weights is None:
        weights = torch.ones_like(charges_raw)
    w_alpha = weights[:, 0]
    w_beta  = weights[:, 1]

    # Compute sums per batch for predicted α+β and α−β
    alpha_sum = torch.zeros(num_batches, device=device).scatter_add_(0, batch, alpha)
    beta_sum  = torch.zeros(num_batches, device=device).scatter_add_(0, batch, beta)

    q_sum = alpha_sum + beta_sum        # total charge
    s_sum = alpha_sum - beta_sum        # total spin

    # Compute per-batch residuals
    dq = q_total - q_sum
    ds = s_total - s_sum

    # Normalize weights per batch
    w_alpha_sum = torch.zeros(num_batches, device=device).scatter_add_(0, batch, w_alpha)
    w_beta_sum  = torch.zeros(num_batches, device=device).scatter_add_(0, batch, w_beta)
    w_alpha_norm = w_alpha / (w_alpha_sum[batch] + epsilon)
    w_beta_norm  = w_beta  / (w_beta_sum[batch] + epsilon)

    # Residuals per batch for each constraint
    dq_atom = dq[batch]
    ds_atom = ds[batch]

    delta_alpha = 0.5 * (dq_atom * w_alpha_norm + ds_atom * w_alpha_norm)
    delta_beta  = 0.5 * (dq_atom * w_beta_norm - ds_atom * w_beta_norm)

    alpha_corr = alpha + delta_alpha
    beta_corr  = beta  + delta_beta

    return torch.stack([alpha_corr, beta_corr], dim=-1)