"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


def get_l_component_range(x: torch.Tensor, l_min: int, l_max: int) -> torch.Tensor:
    """Extract spherical harmonic components for L in [l_min, l_max] from node embeddings.

    The node embeddings are assumed to be organized as [N, (lmax+1)^2, C] where the
    second dimension contains spherical harmonic coefficients ordered by L:
    - L=0: index 0 (1 component)
    - L=1: indices 1-3 (3 components)
    - L=2: indices 4-8 (5 components)
    - etc.

    Args:
        x: Node embeddings tensor of shape [N, (lmax+1)^2, C].
        l_min: Lowest angular momentum quantum number to include (0, 1, 2, ...).
        l_max: Highest angular momentum quantum number to include (>= l_min).

    Returns:
        Tensor of shape [N, (l_max+1)^2 - l_min^2, C] containing the concatenated
        spherical harmonic components for all L in [l_min, l_max].
    """
    start_idx = l_min * l_min
    num_components = (l_max + 1) ** 2 - l_min**2
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
    emb: dict[str, torch.Tensor],
    energy_block: torch.nn.Module,
    batch: torch.Tensor,
    num_systems: int,
    natoms: torch.Tensor | None = None,
    reduce: Literal["sum", "mean"] = "sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute system-level energy from node embeddings and an energy block.

    Extracts the L=0 (scalar) component from node embeddings, applies the energy
    block to produce per-node energies, reduces to system-level energies, and
    optionally normalizes by the number of atoms per system.

    Args:
        emb: Embedding dictionary containing "node_embedding" of shape [N, (lmax+1)^2, C].
        energy_block: Module that maps scalar node features [N, C] to per-node energies [N, 1].
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.
        natoms: Number of atoms per system, shape [num_systems]. Required when reduce="mean".
        reduce: How to aggregate node energies into system energies. "sum" returns the total
            energy; "mean" divides by natoms to return the average energy per atom.

    Returns:
        A tuple of (energy, energy_part) where:
        - energy: System-level energy after GP reduction and reduce, shape [num_systems].
        - energy_part: System-level energy before GP reduction (for autograd), shape [num_systems].
    """
    scalar_embedding = get_l_component_range(
        emb["node_embedding"], l_min=0, l_max=0
    ).squeeze(1)
    node_energy = energy_block(scalar_embedding)
    node_energy_flat = node_energy.view(-1)
    energy, energy_part = reduce_node_to_system(node_energy_flat, batch, num_systems)

    if reduce == "sum":
        pass
    elif reduce == "mean":
        if natoms is None:
            raise ValueError("natoms must be provided when reduce='mean'")
        energy = energy / natoms
    else:
        raise ValueError(f"reduce can only be sum or mean, got: {reduce}")

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
    pos: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forces and stress from energy using autograd.

    Forces are computed as the negative gradient of energy with respect to positions.
    Stress is computed by reconstructing the virial tensor from the position and cell
    gradients, equivalent to the strain-derivative approach.

    The virial is:
        V = (g_r^T r + r^T g_r) / 2 + (cell^T g_h + g_h^T cell) / 2

    where g_r = dE/dpos and g_h = dE/dcell. This matches dE/dε for a symmetric
    strain ε applied as r' = r(I + ε), h' = h(I + ε).

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        cell: Unit cell vectors, shape [num_systems, 3, 3].
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.
        training: Whether to create graph for higher-order gradients.

    Returns:
        A tuple of (forces, stress) where:
        - forces: Shape [N, 3].
        - stress: Shape [num_systems, 9] (flattened 3x3 tensor).
    """
    grads = torch.autograd.grad(
        [energy_part.sum()],
        [pos, cell],
        create_graph=training,
    )

    if gp_utils.initialized():
        grads = (
            gp_utils.reduce_from_model_parallel_region(grads[0]),
            gp_utils.reduce_from_model_parallel_region(grads[1]),
        )

    forces = torch.neg(grads[0])

    pos_virial_per_atom = grads[0].unsqueeze(2) * pos.unsqueeze(1)  # [N, 3, 3]
    pos_virial, _ = reduce_node_to_system(pos_virial_per_atom, batch, num_systems)

    cell_virial = cell.mT @ grads[1]  # [B, 3, 3]

    virial = (pos_virial + pos_virial.mT + cell_virial + cell_virial.mT) / 2
    volume = torch.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = stress.view(-1, 9)

    return forces, stress
