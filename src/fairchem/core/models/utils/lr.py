from __future__ import annotations

import numpy as np
import torch
from torch_scatter import scatter
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
        batch = torch.zeros(n, dtype=torch.int64, device=r.device)

    unique_batches = torch.unique(batch)  # Get unique batch indices
    
    """
    batch_indices = batch.unsqueeze(-1) == unique_batches.unsqueeze(0)  # Broadcasting mask
    pos_now = pos.unsqueeze(1) * batch_indices  # Shape: [n_atoms, n_batches, 3]
    q_now = q.unsqueeze(1) * batch_indices  # Shape: [n_atoms, n_batches, 1]
    q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)  # Center charges per batch

    j, i = edge_index
    distance_vec = pos_now[j] - pos_now[i]
    # red to [n_interactions, 1]
    edge_dist = distance_vec.norm(dim=-1).unsqueeze(-1)
    edge_dist.requires_grad_(True)  
    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    q_source = q_now[i].view(-1)
    q_target = q_now[j].view(-1)
    pairwise_potential = q_source.unsqueeze(0) * q_target.unsqueeze(1)
    pairwise_potential = pairwise_potential * edge_dist_transformed#.unsqueeze(2)
    
    if conv_function_tf:
        convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
        pairwise_potential *= convergence_func
    
    # remove diagonal elements
    pairwise_potential = pairwise_potential * (i != j).float()
    # add back self-interaction
    self_interaction = torch.sum(q_now ** 2, dim=0) / (sigma * twopi**(3./2.))
    pairwise_potential += self_interaction
    norm_factor=90.0474
    results["potential"] = torch.stack(pairwise_potential, dim=0).sum(dim=1) * norm_factor
    """
    #results = scatter(
    #    pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    #).sum(dim=1)

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
        '''        
        batch_indices = batch.unsqueeze(-1) == unique_batches.unsqueeze(0)  # Broadcasting mask
        r_now = pos.unsqueeze(1) * batch_indices  # Shape: [n_atoms, n_batches, 3]
        q_now = q.unsqueeze(1) * batch_indices  # Shape: [n_atoms, n_batches, 1]
        q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)  # Center charges per batch
        polarization = torch.sum(q_now * r_now, dim=0)  # Sum over atoms per batch
        phase = torch.ones_like(r_now, dtype=torch.complex64, requires_grad=True)
        '''
        
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
    
    #self_interaction = torch.sum(q ** 2) / (sigma * twopi**(3./2.))
    #pairwise_potential += self_interaction
    
    # add back self-interaction
    #self_interaction = torch.sum(q ** 2) / (sigma * twopi**(3./2.))
    #pairwise_potential += self_interaction
    
    #results["potential"] = scatter(
    #    pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    #)
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
