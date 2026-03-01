"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest
import torch
from ase.build import bulk, fcc111

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.models.uma.escn_md import GradRegressConfig


def apply_strain(atoms: Atoms, strain: np.ndarray) -> Atoms:
    """
    Apply a strain matrix to an atomic structure.

    Args:
        atoms: ASE Atoms object to strain.
        strain: 3x3 strain matrix where strain[i,j] represents the strain
                in direction i for cell vector j. Diagonal elements are
                normal strains (e.g., 0.02 for 2% tensile strain),
                off-diagonal elements are shear strains.

    Returns:
        The strained Atoms object (modified in place).
    """
    cell = atoms.get_cell()
    strained_cell = cell @ (np.eye(3) + strain)
    atoms.set_cell(strained_cell, scale_atoms=True)
    return atoms


@pytest.fixture()
def slab_atoms() -> Atoms:
    atoms = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    atoms.pbc = True
    return atoms


@pytest.fixture()
def bulk_atoms() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2))
    return atoms


@pytest.fixture()
def uma_predict_unit(request):
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0], device="cuda")


def get_displacement_and_cell(
    data: AtomicData,
    regress_config: GradRegressConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Prepare displacement tensor and cell for OLD stress computation method.

    This function is used for testing purposes to compare the old displacement-gradient
    method with the new virial reconstruction method. It sets up the displacement tensor
    and modifies the input data to enable gradient-based computation of forces and stress
    using the old implementation.

    When stress regression is enabled with gradient-based forces, it:
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

    Example usage in tests:
        ```python
        from fairchem.core.datasets.atomic_data import AtomicData
        from fairchem.core.models.uma.outputs import compute_energy
        from tests.core.testing_utils import (
            get_displacement_and_cell,
            compute_stress_old_method,
        )

        # Prepare atomic data
        atoms = bulk("Cu").repeat(2)
        atomic_data = AtomicData.from_ase(atoms, task_name="omat")

        # Get model components
        model = predict_unit.model.module
        backbone = model.backbone
        head_name = "energyandforcehead"
        heads = model.heads
        energy_block = heads[head_name].energy_block
        regress_config = heads[head_name].regress_config

        # Set up for old method
        displacement, orig_cell = get_displacement_and_cell(atomic_data, regress_config)

        # Forward pass with gradients
        with torch.set_grad_enabled(True):
            emb = backbone(atomic_data)
            energy, energy_part = compute_energy(
                emb, energy_block, atomic_data.batch, len(atomic_data.natoms)
            )

            # Compute stress with old method
            forces, stress = compute_stress_old_method(
                energy_part,
                atomic_data["pos_original"],
                displacement,
                orig_cell,
                training=False,
            )
        ```
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


def compute_stress_from_cell_displacement(
    energy_part: torch.Tensor,
    pos_original: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OLD stress computation method using displacement gradient.

    This is the previous implementation from the MLP-head-refactor branch, preserved
    for testing purposes to verify that the new stress implementation produces
    identical results.

    The old method takes gradients with respect to a displacement tensor that is
    applied as a symmetric strain to both positions and the unit cell. The virial
    is obtained directly from the gradient with respect to the displacement tensor.

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos_original: Original atomic positions before displacement, shape [N, 3].
        displacement: Symmetric strain displacement tensor, shape [num_systems, 3, 3].
        cell: Unit cell vectors after displacement, shape [num_systems, 3, 3].
        training: Whether to create graph for higher-order gradients.

    Returns:
        A tuple of (forces, stress) where:
        - forces: Shape [N, 3].
        - stress: Shape [num_systems, 9] (flattened 3x3 tensor).

    Example usage in tests comparing old vs new implementation:
        ```python
        import numpy.testing as npt
        from fairchem.core.datasets.atomic_data import AtomicData
        from fairchem.core.models.uma.outputs import (
            compute_energy,
            compute_forces_and_stress,  # NEW method
        )
        from tests.core.testing_utils import (
            get_displacement_and_cell,
            compute_stress_old_method,  # OLD method
        )

        # Test with old method
        atoms = bulk("Cu").repeat(2)
        atomic_data_old = AtomicData.from_ase(atoms, task_name="omat")
        displacement, orig_cell = get_displacement_and_cell(atomic_data_old, regress_config)

        with torch.set_grad_enabled(True):
            emb_old = backbone(atomic_data_old)
            energy_old, energy_part_old = compute_energy(
                emb_old, energy_block, atomic_data_old.batch, num_systems
            )
            forces_old, stress_old = compute_stress_old_method(
                energy_part_old,
                atomic_data_old["pos_original"],
                displacement,
                orig_cell,
                training=False,
            )

        # Test with new method
        atomic_data_new = AtomicData.from_ase(atoms, task_name="omat")
        atomic_data_new["pos"].requires_grad = True
        atomic_data_new["cell"].requires_grad = True

        with torch.set_grad_enabled(True):
            emb_new = backbone(atomic_data_new)
            energy_new, energy_part_new = compute_energy(
                emb_new, energy_block, atomic_data_new.batch, num_systems
            )
            forces_new, stress_new = compute_forces_and_stress(
                energy_part_new,
                atomic_data_new["pos"],
                atomic_data_new["cell"],
                atomic_data_new.batch,
                num_systems,
                training=False,
            )

        # Compare results
        npt.assert_allclose(
            forces_old.detach().cpu().numpy(),
            forces_new.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        npt.assert_allclose(
            stress_old.detach().cpu().numpy(),
            stress_new.detach().cpu().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        ```
    """
    grads = torch.autograd.grad(
        [energy_part.sum()],
        [displacement],
        create_graph=training,
    )

    if gp_utils.initialized():
        grads = gp_utils.reduce_from_model_parallel_region(grads[0])

    virial = grads[0].view(-1, 3, 3)
    volume = torch.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = stress.view(-1, 9)

    return stress


@pytest.mark.parametrize("atoms_fixture", ["bulk_atoms", "slab_atoms"])
def test_stress_old_vs_new_single_system(request, atoms_fixture, uma_predict_unit):
    """
    Test that old and new stress implementations give identical results for single systems.

    Uses FAIRChemCalculator to predict stress on single atomic structures and compares
    the old displacement-gradient method with the new virial reconstruction method.
    """

    atoms = request.getfixturevalue(atoms_fixture)

    # Apply biaxial strain: 2% tensile in x, 3% compressive in y
    strain = np.diag([0.02, -0.03, 0.0])
    apply_strain(atoms, strain)

    # Get the model
    model = uma_predict_unit.model.module
    backbone = model.backbone
    efs_head = model.output_heads["energyandforcehead"].head

    # Get regress config
    regress_config = efs_head.regress_config

    # Compute energy
    energy_block = efs_head.energy_block
    num_systems = 1

    task_name = "omat"
    # Test with gradient computation
    # First, set up for old method
    atomic_data_old = AtomicData.from_ase(atoms, task_name=task_name)
    displacement_old, orig_cell_old = get_displacement_and_cell(
        atomic_data_old, regress_config
    )

    with torch.set_grad_enabled(True):
        emb_old = backbone(atomic_data_old)
        from fairchem.core.models.uma.outputs import compute_energy

        energy_old, energy_part_old = compute_energy(
            emb_old, energy_block, atomic_data_old.batch, num_systems
        )

        # Compute stress with old method
        stress_old = compute_stress_from_cell_displacement(
            energy_part_old,
            atomic_data_old["pos_original"],
            displacement_old,
            orig_cell_old,
            training=False,
        )

    # Now test new method
    atomic_data_new = AtomicData.from_ase(atoms, task_name=task_name)
    atomic_data_new["pos"].requires_grad = True
    atomic_data_new["cell"].requires_grad = True

    with torch.set_grad_enabled(True):
        emb_new = backbone(atomic_data_new)
        energy_new, energy_part_new = compute_energy(
            emb_new, energy_block, atomic_data_new.batch, num_systems
        )

        # Compute stress with new method
        from fairchem.core.models.uma.outputs import compute_forces_and_stress

        _, stress_new = compute_forces_and_stress(
            energy_part_new,
            atomic_data_new["pos"],
            atomic_data_new["cell"],
            atomic_data_new.batch,
            num_systems,
            training=False,
        )
        # preds = uma_predict_unit.predict(atomic_data_new)
        # stress_new = preds["stress"]

    npt.assert_allclose(
        stress_old.detach().cpu().numpy(),
        stress_new.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Stress differs between old and new implementations",
    )


def test_stress_old_vs_new_batch_prediction(bulk_atoms, slab_atoms, uma_predict_unit):
    """
    Test that old and new stress implementations give identical results for batch predictions.

    Creates a batch containing multiple atomic structures with different strains and compares
    stress predictions from the old displacement-gradient method with the new virial
    reconstruction method.
    """
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    from fairchem.core.models.uma.outputs import (
        compute_energy,
        compute_forces_and_stress,
    )

    # Prepare batch data with multiple systems, each with different strains
    task_name = "omat"

    # Create copies and apply different strains to each
    bulk_1 = bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2))
    apply_strain(bulk_1, np.diag([0.03, -0.02, 0.01]))  # anisotropic strain

    bulk_2 = bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
    apply_strain(bulk_2, np.diag([0.05, 0.05, 0.05]))  # isotropic tensile strain

    slab_1 = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    slab_1.pbc = True
    apply_strain(slab_1, np.diag([0.02, -0.03, 0.0]))  # biaxial strain

    bulk_3 = bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2))
    # Apply shear strain
    shear_strain = np.array([[0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    apply_strain(bulk_3, shear_strain)

    slab_2 = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    slab_2.pbc = True
    apply_strain(slab_2, np.diag([-0.01, -0.01, 0.0]))  # biaxial compression

    slab_3 = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    slab_3.pbc = True
    apply_strain(slab_3, np.diag([0.04, 0.0, 0.0]))  # uniaxial strain

    atoms_list = [bulk_1, bulk_2, slab_1, bulk_3, slab_2, slab_3]

    # Get the model
    model = uma_predict_unit.model.module
    backbone = model.backbone
    efs_head = model.output_heads["energyandforcehead"].head
    energy_block = efs_head.energy_block
    regress_config = efs_head.regress_config

    # Test with old method
    atomic_data_list_old = [
        AtomicData.from_ase(atoms, task_name=task_name) for atoms in atoms_list
    ]
    batched_data_old = atomicdata_list_to_batch(atomic_data_list_old)
    num_systems = len(batched_data_old.natoms)

    displacement_old, orig_cell_old = get_displacement_and_cell(
        batched_data_old, regress_config
    )

    with torch.set_grad_enabled(True):
        emb_old = backbone(batched_data_old)
        energy_old, energy_part_old = compute_energy(
            emb_old, energy_block, batched_data_old.batch, num_systems
        )

        # Compute stress with old method
        stress_old = compute_stress_from_cell_displacement(
            energy_part_old,
            batched_data_old["pos_original"],
            displacement_old,
            orig_cell_old,
            training=False,
        )

    # Test with new method
    atomic_data_list_new = [
        AtomicData.from_ase(atoms, task_name=task_name) for atoms in atoms_list
    ]
    batched_data_new = atomicdata_list_to_batch(atomic_data_list_new)
    batched_data_new["pos"].requires_grad = True
    batched_data_new["cell"].requires_grad = True

    with torch.set_grad_enabled(True):
        emb_new = backbone(batched_data_new)
        energy_new, energy_part_new = compute_energy(
            emb_new, energy_block, batched_data_new.batch, num_systems
        )

        # Compute stress with new method
        _, stress_new = compute_forces_and_stress(
            energy_part_new,
            batched_data_new["pos"],
            batched_data_new["cell"],
            batched_data_new.batch,
            num_systems,
            training=False,
        )

    # Compare results
    npt.assert_allclose(
        stress_old.detach().cpu().numpy(),
        stress_new.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Stress differs between old and new implementations for batch prediction",
    )
