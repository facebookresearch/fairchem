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
        batch: batch vector of shape (n_atoms,)
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
        n, d = pos.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        all_P = []
        all_phases = [] 
        unique_batches = torch.unique(batch)  # Get unique batch indices
        
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
        
        bec_complex = grad(y=P, x=pos)
        
        # dephase
        result = bec_complex * phases.unsqueeze(1).conj()
        result_bec = result.real
        results["bec"] = result_bec
    
    
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1)
    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    
    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    pairwise_potential = q_source * q_target * edge_dist_transformed
    
    if conv_function_tf:
        convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
        pairwise_potential *= convergence_func#.unsqueeze(2)
    
    # remove diagonal elements 
    pairwise_potential = pairwise_potential * (i != j).float()
    norm_factor = 90.0474
    results["potential"] = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    ) * norm_factor
    
    return results


            
def potential_full_ewald_batched(
    pos: torch.Tensor,
    q: torch.Tensor,
    cell: torch.Tensor,
    dl: float = 2.0,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    return_bec: bool = False,
    batch: torch.Tensor | None = None,
):
    """
    Get the potential energy for each atom in the batch using Ewald summation.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        q: charge vector of shape (n_atoms, 1)
        cell: cell matrix of shape (batch_size, 3, 3)
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
        dl: grid resolution
        k_sq_max: maximum k^2 value
        twopi: 2 * pi
        max_num_neighbors: maximum number of neighbors for each atom
        batch: batch vector of shape (n_atoms,)
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """
    
    device = pos.device
    sigma_sq_half = sigma ** 2 / 2.0
    k_sq_max = (np.pi / dl) ** 2
    norm_factor = 90.0474
    
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=device)
    
    # Compute reciprocal lattice vectors for each batch
    cell_inv = torch.linalg.inv(cell)  # [B, 3, 3]
    G = 2 * torch.pi * cell_inv.transpose(-2, -1)  # [B, 3, 3]

    # Determine maximum Nk for each axis in each batch
    norms = torch.norm(cell, dim=2)  # [B, 3]
    Nk = torch.clamp(torch.floor(norms / dl).int(), min=1)  # [B, 3]
    
    # Pre-allocate maximum grid size to reuse memory
    max_Nk = Nk.max()
    max_grid_size = (2 * max_Nk + 1) ** 3
    
    # Pre-allocate reusable tensors
    nvec_buffer = torch.empty((max_grid_size, 3), device=device, dtype=G.dtype)
    kvec_buffer = torch.empty((max_grid_size, 3), device=device, dtype=G.dtype)
    k_sq_buffer = torch.empty(max_grid_size, device=device, dtype=G.dtype)
    
    # Process each batch separately to minimize memory usage
    unique_batches = torch.unique(batch)
    result_potentials = torch.zeros(pos.shape[0], device=device)
    
    for b_idx in unique_batches:
        # Get atoms and cell for this batch
        atom_mask = batch == b_idx
        pos_b = pos[atom_mask]  # [n_atoms_b, 3]
        q_b = q[atom_mask]  # [n_atoms_b, 1] or [n_atoms_b]
        
        if q_b.dim() == 3:
            q_b = q_b.squeeze(-1)
        
        # Generate k-vectors only for this batch
        G_b = G[b_idx]  # [3, 3]
        Nk_b = Nk[b_idx]  # [3]
        
        # Calculate actual grid size for this batch
        grid_size = (2 * Nk_b[0] + 1) * (2 * Nk_b[1] + 1) * (2 * Nk_b[2] + 1)
        
        # Use views of pre-allocated buffers instead of creating new tensors
        nvec = nvec_buffer[:grid_size]
        kvec = kvec_buffer[:grid_size]
        k_sq = k_sq_buffer[:grid_size]
        
        # Generate grid indices efficiently using broadcasting
        n1_range = torch.arange(-Nk_b[0], Nk_b[0] + 1, device=device, dtype=G_b.dtype)
        n2_range = torch.arange(-Nk_b[1], Nk_b[1] + 1, device=device, dtype=G_b.dtype)
        n3_range = torch.arange(-Nk_b[2], Nk_b[2] + 1, device=device, dtype=G_b.dtype)
        
        # Use meshgrid but reshape directly into nvec buffer
        n1_grid, n2_grid, n3_grid = torch.meshgrid(n1_range, n2_range, n3_range, indexing="ij")
        nvec[:, 0] = n1_grid.flatten()
        nvec[:, 1] = n2_grid.flatten()
        nvec[:, 2] = n3_grid.flatten()
        
        # Compute k vectors in-place using matrix multiplication
        torch.mm(nvec, G_b, out=kvec)
        
        # Compute k_sq in-place
        torch.sum(kvec ** 2, dim=1, out=k_sq)
        
        # Apply filters using boolean indexing (creates views, not copies)
        valid_mask = (k_sq > 0) & (k_sq <= k_sq_max)
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        
        if valid_indices.numel() == 0:
            continue
            
        # Use advanced indexing to get valid vectors (still creates copies, but smaller)
        nvec_valid = nvec[valid_indices]
        kvec_valid = kvec[valid_indices]
        k_sq_valid = k_sq[valid_indices]
        
        # Hemisphere masking - use existing logic but optimize
        non_zero_mask = nvec_valid != 0
        has_non_zero = non_zero_mask.any(dim=1)
        first_non_zero_idx = torch.argmax(non_zero_mask.float(), dim=1)
        
        # Use gather efficiently
        sign = torch.gather(nvec_valid, 1, first_non_zero_idx.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ~has_non_zero
        
        # Final filtering
        final_indices = torch.nonzero(hemisphere_mask, as_tuple=True)[0]
        if final_indices.numel() == 0:
            continue
            
        kvec_final = kvec_valid[final_indices]
        k_sq_final = k_sq_valid[final_indices]
        nvec_final = nvec_valid[final_indices]
        
        # Symmetry factors - compute in-place
        is_origin = (nvec_final == 0).all(dim=1)
        factors = torch.where(is_origin, 1.0, 2.0)
        
        # Compute kfac in-place
        kfac = torch.exp(-sigma_sq_half * k_sq_final)
        kfac.div_(k_sq_final + epsilon)
        
        # Structure factor computation - reuse intermediate results
        k_dot_r = torch.mm(pos_b, kvec_final.T)
        
        # Compute S_k components without creating intermediate tensors
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        
        # Multiply q_b in-place and sum
        cos_k_dot_r *= q_b#.unsqueeze(1)
        sin_k_dot_r *= q_b#.unsqueeze(1)
        
        S_k_real = torch.sum(cos_k_dot_r, dim=0)
        S_k_imag = torch.sum(sin_k_dot_r, dim=0)
        
        # Compute S_k_sq in-place
        S_k_real.pow_(2)
        S_k_imag.pow_(2)
        S_k_sq = S_k_real + S_k_imag
        
        # Compute potential for this batch
        volume = torch.det(cell[b_idx])
        
        # Combine factors, kfac, and S_k_sq efficiently
        factors *= kfac
        factors *= S_k_sq
        pot_b = torch.sum(factors) / volume
        
        # Remove self-interaction
        pot_b -= torch.sum(q_b**2) / (sigma * (2 * torch.pi)**1.5)
        
        # Assign to result
        result_potentials[atom_mask] = pot_b * norm_factor
    
    results = {"potential": result_potentials}
    return results


def heisenberg_potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor, 
    q: torch.Tensor,
    nn: torch.nn.Module,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 2)
        nn: neural network to calculate the coupling term
        sigma: sigma parameter for the error function
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1).reshape(-1, 1)
    edge_dist.requires_grad_(True)

    coupling = nn(edge_dist)
    q_source_alpha = q[i][:, 0]
    q_source_beta  = q[i][:, 1]
    q_target_alpha = q[j][:, 0]
    q_target_beta  = q[j][:, 1]

    #q_source = q[i]
    #q_target = q[j]
    pairwise_potential = (
        q_source_beta * q_target_alpha + q_source_alpha * q_target_beta
    ) * coupling
    
    #pairwise_potential = q_source * q_target * coupling

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

