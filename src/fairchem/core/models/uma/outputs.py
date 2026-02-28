"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fairchem.core.common import gp_utils

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.uma.escn_md import GradRegressConfig


def get_displacement_and_cell(
    data: AtomicData,
    regress_config: GradRegressConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Prepare displacement tensor and cell for gradient-based stress computation.

    This function sets up the displacement tensor and modifies the input data
    to enable gradient-based computation of forces and stress. When stress
    regression is enabled with gradient-based forces, it:
    - Creates a symmetric displacement tensor with gradients enabled
    - Applies the displacement to atomic positions
    - Applies the displacement to the unit cell
    - Stores original positions and cell for later use

    Args:
        data: Atomic data containing positions, cell, and batch information.
        regress_config: Configuration for gradient-based computation of forces and stress.

    Returns:
        A tuple of (displacement, orig_cell) where:
        - displacement: Symmetric strain displacement tensor of shape [num_systems, 3, 3],
          or None if stress regression is disabled.
        - orig_cell: Original unit cell before displacement, shape [num_systems, 3, 3],
          or None if stress regression is disabled.

    Note:
        This function modifies the input data dict in place:
        - Sets data["pos_original"] to original positions
        - Modifies data["pos"] to include displacement
        - Modifies data["cell"] to include displacement
        - Enables gradients on data["pos"] if needed
    """
    displacement = None
    orig_cell = None

    # Set up displacement for stress computation
    if regress_config.stress and not regress_config.direct_forces:
        displacement = torch.zeros(
            (3, 3),
            dtype=data["pos"].dtype,
            device=data["pos"].device,
        )
        num_batch = len(data["natoms"])
        displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
        displacement.requires_grad = True

        # Create symmetric displacement tensor
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))

        # Enable gradients on positions if needed
        if data["pos"].requires_grad is False:
            data["pos"].requires_grad = True

        # Store original positions and apply displacement
        data["pos_original"] = data["pos"]
        data["pos"] = data["pos"] + torch.bmm(
            data["pos"].unsqueeze(-2),
            torch.index_select(symmetric_displacement, 0, data["batch"]),
        ).squeeze(-2)

        # Store original cell and apply displacement
        orig_cell = data["cell"]
        data["cell"] = data["cell"] + torch.bmm(data["cell"], symmetric_displacement)

    # Enable gradients for force-only computation
    if (
        not regress_config.stress
        and regress_config.forces
        and not regress_config.direct_forces
        and data["pos"].requires_grad is False
    ):
        data["pos"].requires_grad = True

    return displacement, orig_cell


def get_l_component(x: torch.Tensor, l: int) -> torch.Tensor:
    """Extract the (2L+1) spherical harmonic components for a specific L from node embeddings.

    The node embeddings are assumed to be organized as [N, (lmax+1)^2, C] where the
    second dimension contains spherical harmonic coefficients ordered by L:
    - L=0: index 0 (1 component)
    - L=1: indices 1-3 (3 components)
    - L=2: indices 4-8 (5 components)
    - etc.

    Args:
        x: Node embeddings tensor of shape [N, (lmax+1)^2, C].
        l: The angular momentum quantum number (0, 1, 2, ...).

    Returns:
        Tensor of shape [N, 2L+1, C] containing the L-th spherical harmonic components.
    """
    start_idx = l * l  # Sum of (2k+1) for k=0 to l-1 equals l^2
    num_components = 2 * l + 1
    return x.narrow(1, start_idx, num_components)


def reduce_node_to_system(
    node_values: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce node-level values to system-level by summing over nodes in each system.

    Handles graph-parallel (GP) reduction when GP is initialized.

    Args:
        node_values: Node-level values of shape [N, ...] where N is the number of nodes.
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.

    Returns:
        A tuple of (reduced, unreduced) where:
        - reduced: System-level values after GP reduction (if applicable), shape [num_systems, ...].
        - unreduced: System-level values before GP reduction, useful for autograd, shape [num_systems, ...].
    """
    output_shape = (num_systems,) + node_values.shape[1:]
    system_values = torch.zeros(
        output_shape,
        device=node_values.device,
        dtype=node_values.dtype,
    )

    if node_values.dim() == 1:
        system_values.index_add_(0, batch, node_values)
    else:
        # For multi-dimensional tensors, we need to handle each trailing dimension
        flat_node = node_values.view(node_values.shape[0], -1)
        flat_system = system_values.view(num_systems, -1)
        flat_system.index_add_(0, batch, flat_node)
        system_values = flat_system.view(output_shape)

    if gp_utils.initialized():
        reduced = gp_utils.reduce_from_model_parallel_region(system_values)
    else:
        reduced = system_values

    return reduced, system_values


def compute_energy(
    node_energy: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute system-level energy from node-level energy predictions.

    Args:
        node_energy: Per-node energy predictions, shape [N] or [N, 1].
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.

    Returns:
        A tuple of (energy, energy_part) where:
        - energy: System-level energy after GP reduction, shape [num_systems].
        - energy_part: System-level energy before GP reduction (for autograd), shape [num_systems].
    """
    # Flatten to 1D if needed
    node_energy_flat = node_energy.view(-1)
    energy, energy_part = reduce_node_to_system(node_energy_flat, batch, num_systems)
    return energy, energy_part


def compute_forces(
    energy_part: torch.Tensor,
    pos: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute forces as negative gradient of energy with respect to positions.

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        training: Whether to create graph for higher-order gradients.

    Returns:
        Forces tensor of shape [N, 3].
    """
    (grad,) = torch.autograd.grad(
        energy_part.sum(),
        pos,
        create_graph=training,
    )
    forces = torch.neg(grad)

    if gp_utils.initialized():
        forces = gp_utils.reduce_from_model_parallel_region(forces)

    return forces


def compute_forces_and_stress(
    energy_part: torch.Tensor,
    pos_original: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forces and stress from energy using autograd.

    Forces are computed as the negative gradient of energy with respect to positions.
    Stress is computed from the virial (gradient with respect to strain/displacement)
    divided by the cell volume.

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos_original: Original atomic positions (before strain applied), shape [N, 3].
        displacement: Strain displacement tensor, shape [num_systems, 3, 3].
        cell: Unit cell vectors, shape [num_systems, 3, 3].
        training: Whether to create graph for higher-order gradients.

    Returns:
        A tuple of (forces, stress) where:
        - forces: Shape [N, 3].
        - stress: Shape [num_systems, 9] (flattened 3x3 tensor).
    """
    grads = torch.autograd.grad(
        [energy_part.sum()],
        [pos_original, displacement],
        create_graph=training,
    )

    if gp_utils.initialized():
        grads = (
            gp_utils.reduce_from_model_parallel_region(grads[0]),
            gp_utils.reduce_from_model_parallel_region(grads[1]),
        )

    forces = torch.neg(grads[0])
    virial = grads[1].view(-1, 3, 3)
    volume = torch.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = stress.view(-1, 9)

    return forces, stress


def compute_hessian_vmap(
    forces_flat: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Compute Hessian using vectorized mapping (vmap).

    Uses torch.vmap to compute all Hessian components in parallel by taking
    gradients of each force component with respect to all positions.

    Args:
        forces_flat: Flattened forces tensor, shape [N*3].
        pos: Atomic positions, shape [N, 3].
        create_graph: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [N*3, N*3].
    """

    def compute_grad_component(vec):
        """Compute gradient of forces w.r.t. positions for a single component."""
        return torch.autograd.grad(
            -1 * forces_flat,
            pos,
            grad_outputs=vec,
            retain_graph=True,
            create_graph=create_graph,
        )[0]

    # Use vmap to compute all components in parallel
    hessian = torch.vmap(compute_grad_component)(
        torch.eye(forces_flat.numel(), device=forces_flat.device)
    )

    return hessian


def compute_hessian_loop(
    forces_flat: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Compute Hessian using a loop over force components.

    Iteratively computes gradients of each force component with respect to
    positions. This is a fallback when vmap is not desired or unavailable.

    Args:
        forces_flat: Flattened forces tensor, shape [N*3].
        pos: Atomic positions, shape [N, 3].
        create_graph: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [N*3, N*3].
    """
    n_forces = len(forces_flat)
    hessian = torch.zeros(
        (n_forces, n_forces),
        device=forces_flat.device,
        dtype=forces_flat.dtype,
        requires_grad=False,
    )

    for i in range(n_forces):
        hessian[:, i] = torch.autograd.grad(
            -forces_flat[i],
            pos,
            retain_graph=i < n_forces - 1,
            create_graph=create_graph,
        )[0].flatten()

    return hessian


def compute_hessian(
    forces: torch.Tensor,
    pos: torch.Tensor,
    vmap: bool = True,
    training: bool = False,
) -> torch.Tensor:
    """Compute Hessian matrix as second derivative of energy w.r.t. positions.

    The Hessian is computed as the negative gradient of forces with respect to
    positions: H = -∇_pos(forces) = ∇²_pos(energy).

    Args:
        forces: Force tensor, shape [N, 3].
        pos: Atomic positions, shape [N, 3].
        vmap: Whether to use vectorized mapping (faster but higher memory).
        training: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [N*3, N*3].

    Note:
        Graph parallel (GP) mode is not fully supported. The Hessian should
        be computed after forces have been reduced across GP ranks.
    """
    if gp_utils.initialized():
        raise NotImplementedError(
            "Hessian computation is not currently supported with graph parallel mode. "
            "Please compute Hessian on a single rank after force reduction."
        )

    forces_flat = forces.flatten()

    if vmap:
        hessian = compute_hessian_vmap(forces_flat, pos, create_graph=training)
    else:
        hessian = compute_hessian_loop(forces_flat, pos, create_graph=training)

    return hessian
