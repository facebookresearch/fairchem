from __future__ import annotations

import os
import random
import subprocess
import tempfile

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write
from sklearn.model_selection import train_test_split

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.common.utils import get_timestamp_uid
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.mlip_unit import UNIT_INFERENCE_CHECKPOINT


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_random_bulk_structure():
    """Generate a random bulk structure with various crystal systems and elements."""

    # Common elements for bulk structures
    elements = ["Al", "Cu", "Fe", "Ni", "Ti", "Mg", "Zn", "Cr", "Mn", "Co"]

    # Crystal structures
    crystal_structures = ["fcc", "bcc", "hcp", "diamond", "sc"]

    # Randomly select element and structure
    element = random.choice(elements)
    structure = random.choice(crystal_structures)

    # Generate base structure with random lattice parameter
    base_a = random.uniform(3.0, 4.0)  # Lattice parameter in Angstroms

    try:
        atoms = bulk(element, structure, a=base_a, cubic=True)
    except:  # noqa: E722
        # Fallback to fcc if structure doesn't work for the element
        atoms = bulk(element, "fcc", a=base_a, cubic=True)

    atoms = atoms.repeat((2, 2, 2))

    # Add random displacement to atoms (thermal motion simulation)
    displacement_magnitude = random.uniform(0.05, 0.2)
    positions = atoms.get_positions()
    displacements = np.random.normal(0, displacement_magnitude, positions.shape)
    atoms.set_positions(positions + displacements)

    # Add random strain to the cell
    strain_magnitude = random.uniform(0.01, 0.05)
    cell = atoms.get_cell()
    strain_tensor = np.eye(3) + np.random.normal(0, strain_magnitude, (3, 3))
    atoms.set_cell(np.dot(cell, strain_tensor), scale_atoms=True)

    return atoms


def generate_random_molecule(n_atoms=5, elements=("C", "H", "O", "N"), box_size=10.0):
    """Generate a random molecule for unit testing."""
    symbols = np.random.choice(elements, n_atoms)
    positions = np.random.uniform(-box_size / 2, box_size / 2, (n_atoms, 3))
    return Atoms(symbols=symbols, positions=positions)


def generate_fake_energy(atoms):
    """Generate fake energy based on number of atoms and some random component."""
    n_atoms = len(atoms)

    # Base energy per atom (roughly based on cohesive energies)
    base_energy_per_atom = random.uniform(-4.0, -2.0)  # eV per atom

    # Add some random variation
    energy_variation = random.uniform(-0.5, 0.5)

    total_energy = n_atoms * base_energy_per_atom + energy_variation
    return total_energy


def generate_fake_forces(atoms):
    """Generate fake forces for all atoms."""
    n_atoms = len(atoms)

    # Generate random forces with realistic magnitudes
    force_magnitude = random.uniform(0.1, 2.0)  # eV/Angstrom
    forces = np.random.normal(0, force_magnitude, (n_atoms, 3))

    # Ensure forces sum to zero (Newton's third law)
    forces -= np.mean(forces, axis=0)

    return forces


def create_dataset(
    type,
    n_structures=1000,
    train_ratio=0.8,
    output_dir="bulk_structures",
    random_state=42,
):
    """
    Create a dataset of random bulk structures with train/validation split.

    Parameters:
    - type: str either molecule or bulk
    - n_structures: Total number of structures to generate
    - train_ratio: Fraction of data for training (default 0.8 for 80/20 split)
    - output_dir: Directory to save the structures
    """

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"Generating {n_structures} random bulk structures...")

    # Generate all structures
    structures = []
    for i in range(n_structures):
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_structures} structures")

        # Generate random structure
        if type == "bulk":
            atoms = generate_random_bulk_structure()
        elif type == "molecule":
            atoms = generate_random_molecule()
        else:
            raise AssertionError("invalid type!")

        # Add fake energy and forces
        energy = generate_fake_energy(atoms)
        forces = generate_fake_forces(atoms)

        # Store energy and forces properly for extxyz format
        atoms.info["energy"] = energy
        atoms.info["config_type"] = "bulk_structure"
        atoms.info["n_atoms"] = len(atoms)

        # Store forces in arrays - ensure they're the right shape and type
        atoms.arrays["forces"] = forces.astype(np.float64)

        # Create a simple calculator to hold the results
        from ase.calculators.singlepoint import SinglePointCalculator

        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc

        structures.append((atoms, i))

    # Split into train and validation sets
    train_structures, val_structures = train_test_split(
        structures, train_size=train_ratio, random_state=random_state
    )

    print(f"Saving {len(train_structures)} training structures...")
    # Save training structures to individual .traj files
    for atoms, original_idx in train_structures:
        filename = os.path.join(train_dir, f"structure_{original_idx:04d}.traj")
        write(filename, atoms)

    # Also save all training structures to a single trajectory file
    train_traj_file = os.path.join(output_dir, "train_all.traj")
    train_atoms_only = [atoms for atoms, _ in train_structures]
    write(train_traj_file, train_atoms_only)

    print(f"Saving {len(val_structures)} validation structures...")
    # Save validation structures to individual .traj files
    for atoms, original_idx in val_structures:
        filename = os.path.join(val_dir, f"structure_{original_idx:04d}.traj")
        write(filename, atoms)

    # Also save all validation structures to a single trajectory file
    val_traj_file = os.path.join(output_dir, "val_all.traj")
    val_atoms_only = [atoms for atoms, _ in val_structures]
    write(val_traj_file, val_atoms_only)

    print("\nDataset creation complete!")
    print(f"Total structures: {n_structures}")
    print(f"Training structures: {len(train_structures)} (saved in {train_dir})")
    print(f"Validation structures: {len(val_structures)} (saved in {val_dir})")
    print("Combined trajectories: train_all.traj, val_all.traj")


@pytest.mark.parametrize(
    "type, random_state",
    [
        ("bulk", 42),
        ("bulk", 49),
        ("molecule", 999),
    ],
)
def test_create_finetune_dataset(type, random_state):
    set_seeds(random_state)
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_dataset(
            type=type,
            n_structures=100,
            train_ratio=0.8,
            output_dir=tmpdirname,
            random_state=random_state,
        )
        create_dataset_command = [
            "python",
            "src/fairchem/core/scripts/create_uma_finetune_dataset.py",
            "--train-dir",
            f"{tmpdirname}/train",
            "--val-dir",
            f"{tmpdirname}/val",
            "--output-dir",
            os.path.join(tmpdirname, "dataset"),
            "--uma-task",
            "omol",
            "--regression-tasks",
            "ef",
        ]
        subprocess.run(create_dataset_command, check=True)
        assert os.path.exists(
            os.path.join(tmpdirname, "dataset", "train", "data.0000.aselmdb")
        )
        assert os.path.exists(
            os.path.join(tmpdirname, "dataset", "val", "data.0000.aselmdb")
        )


def assert_efs_valid(energy, forces, stress):
    """Assert that the energy, forces, and stress are valid."""
    assert energy != 0
    assert not np.isnan(energy)
    # the single atom bulks tend to get zero forces
    # assert np.count_nonzero(forces) > 0
    assert np.count_nonzero(np.isnan(forces)) == 0
    assert np.count_nonzero(stress) > 0
    assert np.count_nonzero(np.isnan(stress)) == 0


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "reg_task,type",
    [
        ("e", "bulk"),
        ("e", "molecule"),
        ("ef", "bulk"),
        ("ef", "molecule"),
    ],
)
def test_e2e_finetuning_bulks(reg_task, type):
    set_seeds(42)
    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.cuda.empty_cache()
        # create a bulks dataset
        create_dataset(
            type=type, n_structures=100, train_ratio=0.8, output_dir=tmpdirname
        )
        # create the ase dataset and yaml
        generated_dataset_dir = os.path.join(tmpdirname, "dataset")
        create_dataset_command = [
            "python",
            "src/fairchem/core/scripts/create_uma_finetune_dataset.py",
            "--train-dir",
            f"{tmpdirname}/train",
            "--val-dir",
            f"{tmpdirname}/val",
            "--output-dir",
            generated_dataset_dir,
            "--uma-task",
            "omat",
            "--regression-tasks",
            reg_task,
        ]
        subprocess.run(create_dataset_command, check=True)
        # finetune for 1 epoch
        job_dir_id = get_timestamp_uid()
        run_dir = os.path.join(tmpdirname, "run_dir")
        train_cmd = [
            "fairchem",
            "-c",
            f"{generated_dataset_dir}/uma_sm_finetune_template.yaml",
            f"job.run_dir={run_dir}",
            f"+job.timestamp_id={job_dir_id}",
            "batch_size=1",
            "max_neighbors=50",
        ]
        subprocess.run(train_cmd, check=True)
        checkpoint_path = os.path.join(
            run_dir, job_dir_id, "checkpoints", "final", UNIT_INFERENCE_CHECKPOINT
        )
        assert os.path.exists(checkpoint_path)
        # try loading this checkpoint and run inference
        predictor = load_predict_unit(checkpoint_path)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        if type == "bulk":
            atoms = bulk("Fe")
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            assert_efs_valid(energy, forces, stress)
        elif type == "molecule":
            atoms = molecule("H2O")
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            assert_efs_valid(energy, forces, stress)
        else:
            raise AssertionError("type unknown!")
