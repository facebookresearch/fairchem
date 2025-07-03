import torch
from torch import nn
from typing import Dict, Any, Union, Optional

from .module import (
    Atomwise,
    Ewald,
    BEC
)

__all__ = ['Les']

class Les(nn.Module):

    def __init__(self, 
            n_in=None,  # input dimension of representation
            n_layers: int = 3,
            n_hidden: Union[int, list] = [32, 16],
            add_linear_nn: bool = True,
            output_scaling_factor: float = 0.1,
            sigma: float = 1.0,
            dl: float = 2.0,
            remove_mean: bool = True,
            epsilon_factor: float = 1.,
            use_atomwise: bool = True,
            les_arguments: Optional[Dict[str, Any]] = None
        ):
        """
        LES model for long-range interations
        """
        super().__init__()

        if les_arguments is not None:
            self._parse_arguments(les_arguments)
        else:
            self.n_in = n_in
            self.les_arguments = {}
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.add_linear_nn = add_linear_nn
            self.output_scaling_factor = output_scaling_factor
            self.sigma = sigma
            self.dl = dl
            self.remove_mean = remove_mean
            self.epsilon_factor = epsilon_factor
            self.use_atomwise = use_atomwise
        
        
        self.atomwise: nn.Module = (
            Atomwise(
                n_in=self.n_in,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor, 
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )
        # FLAG
        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl
            )

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )
        
    
    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.n_in = les_arguments.get('n_in', None)  # input dimension of representation    
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_atomwise = les_arguments.get('use_atomwise', True)

    def forward(self, 
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        latent_charges: torch.Tensor
        One can also directly input the latent charges. Shape: (n_atoms, )
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)


        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError("desc must be provided and use_atomwise must be True if latent_charges is not provided")
            # compute the latent charges
            assert desc.shape[0] == positions.shape[0]
            
            # error is here - desc/batch
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        # compute the long-range interactions
        if compute_energy:
            E_lr = self.ewald(q=latent_charges,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              )
        else:
            E_lr = None

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
                           r=positions,
                           cell=cell,
                           batch=batch,
                           output_index=bec_output_index,
		           )
        else:
            bec = None

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'BEC': bec,
            }
        return output 

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")