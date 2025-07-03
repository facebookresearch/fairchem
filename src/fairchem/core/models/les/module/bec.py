import torch
import torch.nn as nn
from typing import Optional

from ..util import grad

__all__ = ['BEC']

class BEC(nn.Module):
    def __init__(self,
                 remove_mean: bool = True,
                 epsilon_factor: float = 1., # \epsilon_infty
                 ):
        super().__init__()
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                output_index: Optional[int] = None, # 0, 1, 2 to select only one component
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

        # compute the polarization for each batch
        all_P = []
        all_phases = [] 
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now = r[mask], q[mask]
            if self.remove_mean:
                q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)
    
            if cell is not None:
                box_now = cell[i]  # Get the box for the i-th configuration

            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                polarization = torch.sum(q_now * r_now, dim=0)
                phase = torch.ones_like(r_now, dtype=torch.complex64)
            else:
                polarization, phase = self.compute_pol_pbc(r_now, q_now, box_now)
            if output_index is not None:
                polarization = polarization[output_index]
                phase = phase[:, output_index]

            all_P.append(polarization * self.normalization_factor)
            all_phases.append(phase)
        P = torch.stack(all_P, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # take the gradient of the polarization w.r.t. the positions to get the complex BEC
        bec_complex = grad(y=P, x=r)
   
        # dephase
        result = bec_complex * phases.unsqueeze(1).conj()
        return result.real
 
    def compute_pol_pbc(self, r_now, q_now, box_now):
        r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        phase = torch.exp(1j * 2.* torch.pi * r_frac)
        S = torch.sum(q_now * phase, dim=0)
        polarization = torch.matmul(box_now.to(S.dtype), 
                                    S.unsqueeze(1)) / (1j * 2.* torch.pi)
        return polarization.reshape(-1), phase

    def __repr__(self):
        return f'BEC(remove_mean={self.remove_mean}, epsilon_factor={self.epsilon_factor})'