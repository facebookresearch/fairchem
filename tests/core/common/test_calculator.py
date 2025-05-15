"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.build import add_adsorbate, bulk, fcc111, molecule
from ase.optimize import BFGS
from huggingface_hub import hf_hub_download

from fairchem.core import FAIRChemCalculator
from fairchem.core.common.calculator import (
    AllZeroUnitCellError,
    MixedPBCError,
)
from fairchem.core.units.mlip_unit.api.inference import inference_settings_turbo

if TYPE_CHECKING:
    from ase import Atoms

# List of Hugging Face Hub checkpoints for testing - requires gated model access via hf_hub cli login or HF_TOKEN
HF_HUB_CHECKPOINTS = [
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "uma_sm_130525.pt",
        "task_name": "omat",
        "charge_spin": False,
    },
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "uma_sm_130525.pt",
        "task_name": "omol",
        "charge_spin": True,
    },
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "uma_sm_130525.pt",
        "task_name": "omc",
        "charge_spin": False,
    },
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "uma_sm_130525.pt",
        "task_name": "odac",
        "charge_spin": False,
    },
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "uma_sm_130525.pt",
        "task_name": "oc20",
        "charge_spin": False,
    },
    {
        "repo_id": "facebook/uma-prerelease-checkpoints",
        "filename": "omol_test_checkpoint_130525.pt",
        "task_name": None,
        "charge_spin": True,
    },
]


@pytest.fixture()
def slab_atoms() -> Atoms:
    atoms = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    add_adsorbate(atoms, "O", height=1.2, position="fcc")
    atoms.pbc = True
    return atoms


@pytest.fixture()
def bulk_atoms() -> Atoms:
    return bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2))


@pytest.fixture()
def aperiodic_atoms() -> Atoms:
    return molecule("H2O")


@pytest.fixture()
def periodic_h2o_atoms() -> Atoms:
    """Create a periodic box of H2O molecules."""
    atoms = molecule("H2O")
    atoms.set_cell([100.0, 100.0, 100.0])  # Define a cubic cell
    atoms.set_pbc(True)  # Enable periodic boundary conditions
    atoms = atoms.repeat((2, 2, 2))  # Create a 2x2x2 periodic box
    return atoms


@pytest.fixture()
def large_bulk_atoms() -> Atoms:
    """Create a bulk system with approximately 1000 atoms."""
    return bulk("Fe", "bcc", a=2.87).repeat((10, 10, 10))  # 10x10x10 unit cell


@pytest.mark.gpu()
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_calculator_setup(checkpoint):
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )

    assert "energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties
    # assert "stress" in calc.implemented_properties

    # all conservative UMA checkpoints should support E/F/S!
    if not calc.predictor.direct_forces and calc.task_name is not None:
        for key in ["energy", "forces", "stress"]:
            assert key in calc.calc_property_to_model_key_mapping
    else:
        for key in ["energy", "forces"]:
            assert key in calc.calc_property_to_model_key_mapping


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "atoms_fixture",
    [
        "slab_atoms",
        "bulk_atoms",
        "aperiodic_atoms",
        "periodic_h2o_atoms",
        "large_bulk_atoms",
    ],
)
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_energy_calculation(request, atoms_fixture, checkpoint):
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )
    atoms = request.getfixturevalue(atoms_fixture)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)


@pytest.mark.gpu()
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_relaxation_final_energy(slab_atoms, checkpoint):
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )

    slab_atoms.calc = calc
    opt = BFGS(slab_atoms)
    opt.run(fmax=0.05, steps=100)
    final_energy = slab_atoms.get_potential_energy()
    assert isinstance(final_energy, float)


# TODO: the wigner matrices should be dependent on the RNG, but the energies
# are not actually different using the above seed setting code. To be dug into in the future.
# def test_random_seed_final_energy(bulk_atoms, tmp_path):
#     seeds = [100, 200, 300, 200]
#     results_by_seed = {}
#     for seed in seeds:
#         calc = FAIRChemCalculator(
#             hf_hub_repo_id=HF_HUB_REPO_ID,
#             hf_hub_filename=HF_HUB_FILENAMES[0],
#             device="cuda",
#             task_name="omat",
#             seed=seed,
#         )
#         bulk_atoms.calc = calc
#         energy = bulk_atoms.get_potential_energy()
#         if seed in results_by_seed:
#             assert results_by_seed[seed] == energy
#         else:
#             results_by_seed[seed] = energy
#     for seed_a in set(seeds):
#          for seed_b in set(seeds) - {seed_a}:
#              assert results_by_seed[seed_a] != results_by_seed[seed_b]


@pytest.mark.gpu()
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_calculator_configurations_turbo(slab_atoms, checkpoint):
    # turbo mode requires compilation and needs to reset here
    torch.compiler.reset()
    device = "cuda"
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        device=device,
        inference_settings=inference_settings_turbo(),
        task_name=checkpoint["task_name"],
    )
    slab_atoms.calc = calc

    # Test energy calculation
    energy = slab_atoms.get_potential_energy()
    assert isinstance(energy, float)

    forces = slab_atoms.get_forces()
    assert isinstance(forces, np.ndarray)

    if "stress" in calc.calc_property_to_model_key_mapping is not None:
        stress = slab_atoms.get_stress()
        assert isinstance(stress, np.ndarray)


@pytest.mark.gpu()
@pytest.mark.parametrize("hf_hub", [True, False])
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_calculator_checkpoint_download(slab_atoms, hf_hub, checkpoint):
    """Test downloading a checkpoint from Hugging Face Hub and using checkpoint_path directly."""

    if hf_hub:
        calc = FAIRChemCalculator(
            hf_hub_repo_id=checkpoint["repo_id"],
            hf_hub_filename=checkpoint["filename"],
            device="cuda",
            task_name=checkpoint["task_name"],
        )
    else:
        checkpoint_path = hf_hub_download(
            repo_id=checkpoint["repo_id"], filename=checkpoint["filename"]
        )
        calc = FAIRChemCalculator(
            checkpoint_path=checkpoint_path,
            device="cuda",
            task_name=checkpoint["task_name"],
        )

    slab_atoms.calc = calc

    # Test energy calculation
    energy = slab_atoms.get_potential_energy()
    assert isinstance(energy, float)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "checkpoint",
    [checkpoint for checkpoint in HF_HUB_CHECKPOINTS if checkpoint["charge_spin"]],
)
def test_omol_missing_spin_charge_logs_warning(periodic_h2o_atoms, caplog, checkpoint):
    """Test that missing spin/charge in atoms.info logs a warning when task_name='omol'."""
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )
    periodic_h2o_atoms.calc = calc

    with caplog.at_level(logging.WARNING):
        _ = periodic_h2o_atoms.get_potential_energy()

    assert "charge is not set in atoms.info" in caplog.text
    assert "spin multiplicity is not set in atoms.info" in caplog.text


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "checkpoint",
    [checkpoint for checkpoint in HF_HUB_CHECKPOINTS if checkpoint["charge_spin"]],
)
def test_omol_energy_diff_for_charge_and_spin(aperiodic_atoms, checkpoint):
    """Test that energy differs for H2O molecule with different charge and spin_multiplicity."""
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )

    # Test all combinations of charge and spin
    charges = [0, 1, -1]
    spins = [0, 1, 2]
    energy_results = {}

    for charge in charges:
        for spin in spins:
            aperiodic_atoms.info["charge"] = charge
            aperiodic_atoms.info["spin"] = spin
            aperiodic_atoms.calc = calc
            energy = aperiodic_atoms.get_potential_energy()
            energy_results[(charge, spin)] = energy

    # Ensure all combinations produce unique energies
    energy_values = list(energy_results.values())
    assert len(energy_values) == len(
        set(energy_values)
    ), "Energy values are not unique for different charge/spin combinations"


@pytest.mark.gpu()
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_large_bulk_system(large_bulk_atoms, checkpoint):
    """Test a bulk system with 1000 atoms using the small model."""
    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )
    large_bulk_atoms.calc = calc

    # Test energy calculation
    energy = large_bulk_atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = large_bulk_atoms.get_forces()
    assert isinstance(forces, np.ndarray)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "pbc",
    [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, False),
    ],
)
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_mixed_pbc_behavior(aperiodic_atoms, pbc, checkpoint):
    """Test guess_pbc behavior"""
    pbc = np.array(pbc)
    aperiodic_atoms.pbc = pbc
    if np.all(pbc):
        aperiodic_atoms.cell = [100.0, 100.0, 100.0]

    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name=checkpoint["task_name"],
        device="cuda",
    )

    if np.any(aperiodic_atoms.pbc) and not np.all(aperiodic_atoms.pbc):
        with pytest.raises(MixedPBCError):
            aperiodic_atoms.calc = calc
            aperiodic_atoms.get_potential_energy()
    else:
        aperiodic_atoms.calc = calc
        energy = aperiodic_atoms.get_potential_energy()
        assert isinstance(energy, float)


@pytest.mark.gpu()
@pytest.mark.parametrize("checkpoint", HF_HUB_CHECKPOINTS)
def test_error_for_pbc_with_zero_cell(aperiodic_atoms, checkpoint):
    """Test error raised when pbc=True but atoms.cell is zero."""
    aperiodic_atoms.pbc = True  # Set PBC to True

    calc = FAIRChemCalculator(
        hf_hub_repo_id=checkpoint["repo_id"],
        hf_hub_filename=checkpoint["filename"],
        task_name="omol",
        device="cuda",
    )

    with pytest.raises(AllZeroUnitCellError):
        aperiodic_atoms.calc = calc
        aperiodic_atoms.get_potential_energy()
