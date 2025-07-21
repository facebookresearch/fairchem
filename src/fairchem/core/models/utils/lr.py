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
    
    if return_bec:
        # Ensure pos has requires_grad=True
        if not pos.requires_grad:
            pos.requires_grad_(True)
        
        if not q.requires_grad:
            q.requires_grad_(True)

        normalization_factor = epsilon_factor_les ** 0.5
        n, d = pos.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
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
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now = pos[mask], q[mask].reshape(-1, 1)  # [n_atoms, 1]
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

        # take the gradient of the polarization w.r.t. the positions to get the complex BEC
        #print("grad shapes: ", P.shape, pos.shape)
        #print("grad requires grad: ", P.requires_grad, pos.requires_grad, q.requires_grad)
        # Compute gradient conditionally based on training mode
        #print("P requires grad: ", P.requires_grad)
        #print("P requires grad: ", P.grad_fn)
        #print("q requires grad: ", q.grad_fn)
        
        bec_complex = grad(y=P, x=pos)
        
        # dephase
        result = bec_complex * phases.unsqueeze(1).conj()
        result_bec = result.real

    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    # red to [n_interactions, 1]
    edge_dist = distance_vec.norm(dim=-1)
    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    
    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    pairwise_potential = q_source * q_target * edge_dist_transformed 
    
    if conv_function_tf:
        convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
        pairwise_potential *= convergence_func


    results = {}
    if return_bec:
        results["bec"] = result_bec
    
    # remove diagonal elements 
    
    pairwise_potential = pairwise_potential * (i != j).float()
    
    # add back self-interaction
    self_interaction = torch.sum(q_source * q_target) / (sigma * twopi**(3./2.))
    pairwise_potential += self_interaction
    
    results["potential"] = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    )
    
    
    #results["potential"] = scatter(pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum")

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
