import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional
import numpy as np

__all__ = ['Ewald']

class Ewald(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.0474,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)  # Get unique batch indices

        results = []
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]
            
            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            else:
                # the box is periodic, we use the reciprocal sum
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            results.append(pot)

        return torch.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q):
        # Compute pairwise distances (norm of vector differences)
        # Add epsilon for safe Hessian compute
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)
 
        # Error function scaling for long-range interactions
        
        # Compute inverse distance
        r_p_ij = 1.0 / (r_ij_norm)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        #n_node, n_q = q.shape
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        pot = q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)

        #Exclude diagonal terms from energy
        mask = ~torch.eye(pot.shape[0], device=pot.device).to(torch.bool).unsqueeze(-1)
        mask = torch.vstack([mask.transpose(0,-1)]*pot.shape[-1]).transpose(0,-1)
        pot = pot[mask].sum().view(-1) / self.twopi / 2.0

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
    
        return pot * self.norm_factor
 
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T
        #print('G', G.type())

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(G.dtype)
        kvec = nvec @ G  # [N_total, 3]
        #-print("kvec: ", kvec.shape)

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        S_k = torch.sum(q * exp_ikr, dim=0)  # [M]

         #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        #print("qshape: ", q.shape)
        #print("cos_k_dot_r: ", cos_k_dot_r.shape)
        #print("sin_k_dot_r: ", sin_k_dot_r.shape)
        #print("q.shape: ", q.shape)
        S_k_real = torch.sum(q * cos_k_dot_r, dim=0)  # [M]
        S_k_imag = torch.sum(q * sin_k_dot_r, dim=0)  # [M]
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        pot = (factors * kfac * S_k_sq).sum() / volume
        

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)

        return pot.unsqueeze(0) * self.norm_factor

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"