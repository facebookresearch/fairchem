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

    q_source = q[i]
    q_target = q[j]
    pairwise_potential = q_source * q_target * coupling

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


def fmm_coulomb_potentials_batched(
    pos: torch.Tensor,
    q: torch.Tensor,
    batch: torch.Tensor,
    max_particles_per_cell: int = 8,
    min_width: float = 1e-6,
    theta: float = 0.5,
    norm_factor: float = 90.0474,
    kernel_fn=None,
):
    """
    Batched version of FMM for multiple molecular systems.
    
    Args:
        pos: [n_atoms, 3] atom positions
        q: [n_atoms,] or [n_atoms, k] charges/properties
        batch: [n_atoms] batch assignments for atoms
        max_particles_per_cell: max particles before splitting cell
        min_width: minimum cell width
        theta: opening angle for multipole acceptance
        norm_factor: normalization factor for energy
        kernel_fn: optional custom interaction kernel function(x, y, q_y) -> float
                   where x is target position, y is source positions, q_y is source charges
                   
    Returns:
        potentials: [n_atoms] potential at each atom
    """
    device = pos.device
    dtype = pos.dtype
    potentials = torch.zeros(pos.shape[0], dtype=dtype, device=device)
    
    # Process each batch separately
    unique_batches = torch.unique(batch)
    for b in unique_batches:
        mask = batch == b
        if mask.sum() > 0:  # Skip empty batches
            pos_b = pos[mask]
            q_b = q[mask] if q.dim() == 1 else q[mask, :]
            
            # Build octree for this batch
            tree = build_octree_ts(
                pos_b, 
                max_particles_per_cell=max_particles_per_cell, 
                min_width=min_width
            )
            
            # Compute multipoles
            multipoles = compute_multipoles_ts(tree, q_b)
            
            # Compute potentials using either custom kernel or default coulomb
            if kernel_fn is not None:
                pot_b = traverse_and_accumulate_kernel_ts(
                    tree, multipoles, pos_b, q_b, kernel_fn, theta=theta
                )
            else:
                pot_b = traverse_and_accumulate_coulomb_ts(
                    tree, multipoles, pos_b, q_b, theta=theta
                )
                
            potentials[mask] = pot_b * norm_factor
    
    return potentials


def traverse_and_accumulate_kernel_ts(
    tree, 
    multipoles, 
    pos, 
    q, 
    kernel_fn,
    theta: float = 0.5, 
    eps: float = 1e-8
):
    """
    Generic traversal function that uses a custom kernel for interactions.
    
    Args:
        tree: output of build_octree_ts
        multipoles: multipole expansions from compute_multipoles_ts
        pos: [n_atoms, 3] positions
        q: [n_atoms] or [n_atoms, k] charges/properties
        kernel_fn: function(x, y, q_y) -> float that computes interaction
                   between target x and sources y with properties q_y
        theta: acceptance parameter for multipole approximation
        eps: small value to avoid division by zero
        
    Returns:
        potentials: [n_atoms] potential at each atom
    """
    centers = tree["centers"]
    half_widths = tree["half_widths"]
    children = tree["children"]
    node_particles = tree["node_particles"]

    num_nodes = len(centers)
    n = pos.size(0)
    device = pos.device
    potentials = torch.zeros((n,), dtype=pos.dtype, device=device)

    # Pre-stack centers and half_widths for efficiency
    centers_stack = torch.stack(centers) if num_nodes > 0 else torch.empty((0, 3), device=device)
    half_stack = torch.stack([hw.view(()) for hw in half_widths]) if num_nodes > 0 else torch.empty((0,), device=device)

    for ti in range(n):
        x = pos[ti]
        pot = 0.0
        stack = [0]  # Start at root
        
        while len(stack) > 0:
            node_idx = stack.pop()
            if node_idx < 0:
                continue
                
            center = centers_stack[node_idx]
            hw = half_stack[node_idx]
            dx = center - x
            d = torch.norm(dx)
            s = hw * 2.0
            
            if (d <= 0.0) or (node_particles[node_idx].numel() == 0 and (children[node_idx] == -1).all()):
                continue
                
            if (node_particles[node_idx].numel() > 0) and ((node_particles[node_idx] == ti).any()):
                # Same leaf or contains target: direct interactions with other particles
                idxs = node_particles[node_idx]
                mask = idxs != ti
                sel = idxs[mask]
                if sel.numel() > 0:
                    # Use provided kernel for particle-particle interactions
                    pot = pot + kernel_fn(x, pos.index_select(0, sel), q.index_select(0, sel))
            else:
                # Decide acceptance using MAC criterion
                if (d > 0.0) and (s / d < theta):
                    # Accept multipole approximation
                    # Use kernel for particle-multipole interaction
                    pot = pot + kernel_fn(x, center.unsqueeze(0), multipoles[node_idx].unsqueeze(0))
                else:
                    # Recurse to children
                    child_ids = children[node_idx]
                    for k in range(8):
                        cid = int(child_ids[k].item()) if isinstance(child_ids[k], torch.Tensor) else int(child_ids[k])
                        if cid >= 0:
                            stack.append(cid)
                            
        potentials[ti] = pot
        
    return potentials


def heisenberg_kernel(x, y, q_y, nn):
    """
    Kernel function for Heisenberg interactions.
    
    Args:
        x: [3] target position
        y: [n, 3] source positions
        q_y: [n, 2] source properties (alpha, beta)
        nn: neural network for coupling term
        
    Returns:
        potential: scalar potential at position x
    """
    distance_vec = y - x.unsqueeze(0)
    distances = torch.norm(distance_vec, dim=1).reshape(-1, 1)
    distances.requires_grad_(True)
    
    coupling = nn(distances)
    potential = (q_y * coupling).sum()
    
    return potential


def coulomb_kernel(x, y, q_y, eps=1e-8):
    """
    Standard Coulomb interaction kernel.
    
    Args:
        x: [3] target position
        y: [n, 3] source positions
        q_y: [n] source charges
        eps: small value to avoid division by zero
        
    Returns:
        potential: scalar potential at position x
    """
    r = torch.norm(x.unsqueeze(0) - y, dim=1) + eps
    return (q_y.view(-1) / r).sum()


def potential_full_from_edge_inds_fmm(
    pos: torch.Tensor,
    q: torch.Tensor,
    batch: torch.Tensor = None,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    epsilon_factor_les: float = 1.0,
    twopi: float = 2.0 * np.pi,
    norm_factor: float = 90.0474,
    max_particles_per_cell: int = 8,
    theta: float = 0.5,
    conv_function_tf: bool = False,
    return_bec: bool = False,
):
    """
    FMM-based implementation of electrostatic potential energy.
    
    Args:
        pos: [n_atoms, 3] positions
        q: [n_atoms, 1] charges
        batch: [n_atoms] batch assignments
        sigma, epsilon: parameters for convergence function
        norm_factor: energy scaling factor
        max_particles_per_cell: FMM parameter
        theta: FMM acceptance parameter
        conv_function_tf: whether to use convergence function
        return_bec: whether to compute Born effective charges
        
    Returns:
        results: dictionary with "potential" [n_atoms]
    """
    results = {}
    
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)
        
    # Define kernel with convergence function if requested
    def kernel_fn(x, y, q_y):
        r = torch.norm(x.unsqueeze(0) - y, dim=1) + epsilon
        pot = q_y.view(-1) / r
        
        if conv_function_tf:
            conv = torch.special.erf(r / sigma / (2.0**0.5))
            pot = pot * conv
            
        return pot.sum()
    
    # Compute potential using FMM
    potentials = fmm_coulomb_potentials_batched(
        pos=pos,
        q=q.view(-1),
        batch=batch,
        max_particles_per_cell=max_particles_per_cell,
        theta=theta,
        norm_factor=1.0,  # Apply normalization later
        kernel_fn=kernel_fn
    )
    
    # Store in results
    results["potential"] = potentials * norm_factor
    
    # Handle Born effective charges if requested
    if return_bec:
        if not pos.requires_grad:
            pos.requires_grad_(True)
        
        if not q.requires_grad:
            q.requires_grad_(True)
            
        normalization_factor = epsilon_factor_les ** 0.5
        all_P = []
        all_phases = []
        
        unique_batches = torch.unique(batch)
        for i in unique_batches:
            mask = batch == i
            r_now, q_now = pos[mask], q[mask].reshape(-1, 1)
            q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)
            polarization = torch.sum(q_now * r_now, dim=0)
            phase = torch.ones_like(r_now, dtype=torch.complex64)
            
            all_P.append(polarization * normalization_factor)
            all_phases.append(phase)
            
        P = torch.stack(all_P, dim=0)
        phases = torch.cat(all_phases, dim=0)
        
        bec_complex = grad(y=P, x=pos)
        result = bec_complex * phases.unsqueeze(1).conj()
        results["bec"] = result.real
        
    return results


def heisenberg_potential_full_from_edge_inds_fmm(
    pos: torch.Tensor,
    q: torch.Tensor,
    nn: torch.nn.Module,
    batch: torch.Tensor = None,
    sigma: float = 1.0,
    max_particles_per_cell: int = 8,
    theta: float = 0.5,
):
    """
    FMM-based implementation of Heisenberg interactions.
    
    Args:
        pos: [n_atoms, 3] positions
        q: [n_atoms, 2] alpha/beta properties
        nn: neural network for coupling term
        batch: [n_atoms] batch assignments
        sigma: parameter for convergence function
        max_particles_per_cell: FMM parameter
        theta: FMM acceptance parameter
        
    Returns:
        results: [n_atoms] potential energy per atom
    """
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)
        
    # Define Heisenberg kernel with the neural network
    def kernel_fn(x, y, q_y):
        return heisenberg_kernel(x, y, q_y, nn)
        
    # Compute potential using FMM
    potentials = fmm_coulomb_potentials_batched(
        pos=pos,
        q=q,  # Pass full q tensor with shape [n_atoms, 2]
        batch=batch,
        max_particles_per_cell=max_particles_per_cell,
        theta=theta,
        kernel_fn=kernel_fn
    )
    
    return potentials