from typing import Dict
import torch
from torch import nn
from typing import List, Optional

def grad(y: torch.Tensor, x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """
    a wrapper for the gradient calculation
    alow multiple dimensional and/or complex y
    y: [n_graphs, ] or [n_graphs, dim_y]
    x: [n_nodes, :]
    """
    if y.is_complex():
        get_imag = True
    else:
        get_imag = False

    if len(y.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
        gradient_real = torch.autograd.grad(
            outputs=[y],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=(training or get_imag),  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
        assert gradient_real is not None, "Gradient real is None"
        if get_imag:
            gradient_imag = torch.autograd.grad(
                outputs=[y/1j],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=training,  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]
            assert gradient_imag is not None, "Gradient imag is None"
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    else:
        dim_y = y.shape[1] 
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y[:,0])]
        grad_list_real = []
        for i in range(dim_y):
            g = torch.autograd.grad(
                outputs=[y[:, i]],  # [n_graphs, ]
                inputs=[x],         # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < dim_y - 1) or get_imag),
                create_graph=training, # Create graph for second derivative
                allow_unused=True, # For complete dissociation turn to true
            )[0]
            assert g is not None, f"Gradient real for channel {i} is None"
            grad_list_real.append(g)
        gradient_real = torch.stack(grad_list_real, dim=2)  # [n_nodes, 3, dim_y]
        # if y is complex, we need to calculate the imaginary part
        if get_imag:
            grad_list_imag = []
            for i in range(dim_y):
                g = torch.autograd.grad(
                    outputs=[y[:, i]/1j], # [n_graphs, ]
                    inputs=[x], # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=(training or (i < dim_y - 1)), # Make sure the graph is not destroyed during training
                    create_graph=training, # Create graph for second derivative
                    allow_unused=True, # For complete dissociation turn to true
                )[0]
                assert g is not None, f"Gradient imag for channel {i} is None"
                grad_list_imag.append(g)
            gradient_imag = torch.stack(grad_list_imag, dim=2)  # [n_nodes, 3, dim_y]
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    if get_imag:
        return gradient_real + 1j * gradient_imag
    else:
        return gradient_real