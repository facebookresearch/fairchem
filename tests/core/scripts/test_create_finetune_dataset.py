from __future__ import annotations

import os
import random
import subprocess
import tempfile

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from sklearn.model_selection import train_test_split

from fairchem.core.common.utils import get_timestamp_uid
from fairchem.core.units.mlip_unit.mlip_unit import UNIT_INFERENCE_CHECKPOINT


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

    # Create supercell with random size (2x2x2 to 4x4x4)
    nx, ny, nz = random.randint(2, 4), random.randint(2, 4), random.randint(2, 4)
    atoms = atoms.repeat((nx, ny, nz))

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


def create_dataset(n_structures=1000, train_ratio=0.8, output_dir="bulk_structures"):
    """
    Create a dataset of random bulk structures with train/validation split.

    Parameters:
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
        atoms = generate_random_bulk_structure()

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
        structures, train_size=train_ratio, random_state=42
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

    # Create summary file
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Dataset Summary\n")
        f.write("================\n")
        f.write(f"Total structures: {n_structures}\n")
        f.write(f"Training structures: {len(train_structures)}\n")
        f.write(f"Validation structures: {len(val_structures)}\n")
        f.write(f"Train/Val ratio: {train_ratio:.1f}/{1-train_ratio:.1f}\n")
        f.write("\nFiles created:\n")
        f.write("- Individual structure files: .traj format\n")
        f.write("- Combined trajectory files: train_all.traj, val_all.traj\n")
        f.write("\nStructure properties:\n")
        f.write("- Random elements from: Al, Cu, Fe, Ni, Ti, Mg, Zn, Cr, Mn, Co\n")
        f.write("- Crystal structures: fcc, bcc, hcp, diamond, sc\n")
        f.write("- Supercell sizes: 2x2x2 to 4x4x4\n")
        f.write("- Random atomic displacements and cell strain applied\n")
        f.write("- Fake energies and forces generated\n")

    print("\nDataset creation complete!")
    print(f"Total structures: {n_structures}")
    print(f"Training structures: {len(train_structures)} (saved in {train_dir})")
    print(f"Validation structures: {len(val_structures)} (saved in {val_dir})")
    print("Combined trajectories: train_all.traj, val_all.traj")
    print(f"Summary saved to: {summary_file}")


def test_create_finetune_dataset():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_dataset(n_structures=100, train_ratio=0.8, output_dir=tmpdirname)
        create_dataset_command = [
            "python",
            "src/fairchem/core/scripts/create_uma_finetune_dataset.py",
            "--train-dir",
            f"{tmpdirname}/train",
            "--val-dir",
            f"{tmpdirname}/val",
            "--output-dir",
            os.path.join(tmpdirname, "dataset"),
            "--task-to-finetune",
            "omol",
        ]
        subprocess.run(create_dataset_command, check=True)
        assert os.path.exists(
            os.path.join(tmpdirname, "dataset", "train", "data.0000.aselmdb")
        )
        assert os.path.exists(
            os.path.join(tmpdirname, "dataset", "val", "data.0000.aselmdb")
        )


@pytest.mark.gpu()
def test_e2e_finetuning_bulks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a bulks dataset
        create_dataset(n_structures=100, train_ratio=0.8, output_dir=tmpdirname)
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
            "--task-to-finetune",
            "omol",
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
        ]
        subprocess.run(train_cmd, check=True)
        checkpoint_dir = os.path.join(run_dir, job_dir_id, "checkpoints", "final")
        assert os.path.exists(os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT))
        # try loading this checkpoint and run inference
