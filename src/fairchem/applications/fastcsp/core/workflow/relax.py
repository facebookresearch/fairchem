"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Relaxation Module

This module provides functionality for relaxing crystal structures using machine learning
potentials from the FAIRChem toolkit. It supports structure optimization with various
constraints and optimizers, parallel processing via SLURM, and comprehensive analysis
of relaxation outcomes.

Key Features:
- ML-based structure relaxation using Universal Model for Atoms (UMA)
- Support for different optimization algorithms (BFGS, L-BFGS, FIRE)
- Optional symmetry preservation and cell relaxation
- Parallel processing for large datasets via submitit
- Comprehensive validation of relaxation quality
- Energy analysis and structural change detection

The relaxation workflow:
1. Load predicted crystal structures from Parquet files
2. Convert structures to ASE Atoms format
3. Apply ML calculator (UMA variants for different tasks)
4. Perform constrained optimization with selected algorithm
5. Validate structural integrity and analyze changes
6. Save relaxed structures with comprehensive metadata

Requires:
- FAIRChem core with UMA model checkpoints
- ASE for atomic structure manipulation
- PyMatGen for crystallographic analysis
- GPU resources for efficient ML potential evaluation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import submitit
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from fairchem.applications.fastcsp.core.utils.structure import (
    check_no_changes_in_covalent_matrix,
    check_no_changes_in_Z,
)
from p_tqdm import p_map
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from fairchem.core import FAIRChemCalculator, pretrained_mlip

# Recommended UMA models w/ different tasks
CHECKPOINTS = {
    "uma_sm_1p1_omc": {  # RECOMMENDED: UMA w/ OMC task
        "path": "",
        "type": "uma-s-1.1",
        "task_name": "omc",
    },
    "uma_sm_1p1_omol": {  # UMA w/ OMol task
        "path": "",
        "type": "uma-s-1.1",
        "task_name": "omol",
    },
}


def create_calculator(args):
    """
    Create a FAIRChem ML potential calculator for structure relaxation.

    Initializes the Universal Model for Atoms (UMA) predictor with the specified
    task name in a FAIRChemCalculator for use with ASE.

    Args:
        args: Arguments object containing:
            - calculator: Name of the UMA model variant to use
              (e.g., "uma_sm_1p1_omc", "uma_sm_1p1_omol")

    Returns:
        FAIRChemCalculator: Configured calculator ready for structure optimization
    """
    predictor = pretrained_mlip.get_predict_unit("uma-s-1.1", device="cuda")
    calc = FAIRChemCalculator(
        predictor, task_name=CHECKPOINTS[args.calculator]["task_name"]
    )
    return calc


def relax_atoms(atoms, args, calc):
    """
    Perform structure relaxation on a single structure.

    Optimizes the atomic positions (and optionally unit cell parameters) using
    the specified ML potential and optimization algorithm.

    Args:
        atoms: ASE Atoms object containing the initial structure
        args: Arguments object containing relaxation parameters:
            - fix_symmetry: Whether to preserve crystal symmetry during optimization
            - relax_cell: Whether to optimize unit cell parameters
            - optimizer: Optimization algorithm ("bfgs", "lbfgs", "fire")
            - fmax: Force convergence criterion (eV/Å)
            - max_steps: Maximum optimization steps
        calc: FAIRChemCalculator for energy and force evaluation

    Returns:
        ASE Atoms: Relaxed structure with additional info:
            - converged: Boolean indicating whether optimization converged
            - energy: Final potential energy of the relaxed structure

    Raises:
        ValueError: If unsupported optimizer is specified
    """
    # Apply symmetry constraint if requested
    if args.fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    atoms.calc = calc

    # Enable cell optimization if requested
    if args.relax_cell:
        atoms = FrechetCellFilter(atoms)

    # Configure optimization algorithm
    if args.optimizer == "bfgs":
        optimizer = BFGS(atoms)
    elif args.optimizer == "lbfgs":
        optimizer = LBFGS(atoms)
    elif args.optimizer == "fire":
        optimizer = FIRE(atoms)
    else:
        raise ValueError(
            f"Unsupported optimizer: {args.optimizer}. (L)BFGS and FIRE are recommended.)"
        )

    # Perform optimization
    converged = optimizer.run(fmax=args.fmax, steps=args.max_steps)
    print(f"Converged: {converged}, Energy: {atoms.get_potential_energy()}")

    # Store relaxation metadata
    atoms.info["converged"] = converged  # Store convergence status
    atoms.info["energy"] = atoms.get_potential_energy()  # Store relaxed energy
    return atoms


def relax_structures(args, input_files, output_path, column_name="cif"):
    """
    Relax multiple crystal structures from Parquet files using ML potentials.

    Processes batches of crystal structures by loading them from Parquet format,
    converting to ASE Atoms, performing ML-based relaxation, and saving results.

    Args:
        args: Arguments object containing relaxation parameters (optimizer, constraints, etc.)
        input_files: List of Parquet file paths containing structures to relax
        output_path: Directory path for saving relaxed structure files
        column_name: Name of column containing CIF strings in the input files

    Workflow:
        1. Load structures from Parquet files
        2. Convert CIF strings to ASE Atoms objects
        3. Apply special handling for molecular tasks (spin/charge)
        4. Perform parallel relaxation of all structures
        5. Convert back to PyMatGen structures and extract properties
        6. Validate structural integrity and chemical consistency
        7. Save comprehensive results with relaxation metadata

    Output:
        Parquet files containing:
        - Original structure data
        - Relaxed CIF structures
        - Lattice parameters and volume/density
        - Energies and convergence status
        - Structural validation flags (Z preservation, connectivity)
    """
    calc = create_calculator(args)

    for input_file in tqdm(input_files):
        # Generate output file path maintaining directory structure
        output_file = output_path / input_file.relative_to(args.root)
        if output_file.exists():
            print(f"Skipping {input_file} because {output_file} exists")
            continue

        print(f"Relaxing {input_file}")

        # Load structures from Parquet file
        structures_df = pd.read_parquet(input_file)
        atoms_list = (
            structures_df[column_name]
            .apply(
                lambda x: AseAtomsAdaptor.get_atoms(Structure.from_str(x, fmt="cif"))
            )
            .to_numpy()
        )

        # Special handling for OMol tasks - setting spin and charge
        if args.calculator == "uma_sm_1p1_omol":
            for atoms in atoms_list:
                atoms.info.update({"spin": 1, "charge": 0})

        # Perform relaxation for all structures
        atoms_relaxed = [relax_atoms(atoms, args, calc) for atoms in tqdm(atoms_list)]

        # Convert relaxed structures back to pymatgen format
        structures_relaxed = [
            AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_relaxed
        ]

        # Extract properties
        structures_df["relaxed_cif"] = [
            structure.to(fmt="cif") for structure in structures_relaxed
        ]
        structures_df["volume"] = [structure.volume for structure in structures_relaxed]
        structures_df["density"] = [
            structure.density for structure in structures_relaxed
        ]
        structures_df["lattice_a"] = [
            structure.lattice.a for structure in structures_relaxed
        ]
        structures_df["lattice_b"] = [
            structure.lattice.b for structure in structures_relaxed
        ]
        structures_df["lattice_c"] = [
            structure.lattice.c for structure in structures_relaxed
        ]
        structures_df["lattice_alpha"] = [
            structure.lattice.alpha for structure in structures_relaxed
        ]
        structures_df["lattice_beta"] = [
            structure.lattice.beta for structure in structures_relaxed
        ]
        structures_df["lattice_gamma"] = [
            structure.lattice.gamma for structure in structures_relaxed
        ]
        structures_df["energy_relaxed"] = [
            atoms.get_potential_energy() for atoms in atoms_relaxed
        ]
        structures_df["energy_relaxed_per_molecule"] = (
            structures_df["energy_relaxed"] / structures_df["z"]
        )
        structures_df["converged"] = [
            atoms.info["converged"] for atoms in atoms_relaxed
        ]

        # Validate structural integrity after relaxation
        structures_df["Z_unchanged"] = [
            check_no_changes_in_Z(atoms_initial, atoms_relaxed)
            for atoms_initial, atoms_relaxed in zip(atoms_list, atoms_relaxed)
        ]
        structures_df["connectivity_unchanged"] = [
            check_no_changes_in_covalent_matrix(atoms_initial, atoms_relaxed)
            for atoms_initial, atoms_relaxed in zip(atoms_list, atoms_relaxed)
        ]
        structures_df["connectivity_unchanged"] = p_map(
            check_no_changes_in_covalent_matrix,
            atoms_list,
            atoms_relaxed,
            num_cpus=120,  # Parallel processing for connectivity validation
        )
        # Save results to Parquet
        output_file.parent.mkdir(parents=True, exist_ok=True)
        structures_df.to_parquet(output_file, compression="zstd")
        print(f"Wrote {structures_df.shape[0]} relaxed structures to {output_file}")


def run_relax_jobs(
    input_path, output_path, relax_config, slurm_config, column_name="cif"
):
    """
    Orchestrate parallel structure relaxation jobs using SLURM submission.

    Main function that coordinates the parallel relaxation of crystal structures
    across multiple compute nodes. Distributes work and manages job submission
    to SLURM scheduler.

    Args:
        args: Arguments object containing relaxation parameters and paths
        config: Master configuration dictionary containing:
            - root: Base directory path
            - relax: Relaxation-specific configuration including:
                - slurm: SLURM job parameters (time, resources, partition)
                - num_ranks: Number of parallel jobs to create
        input_path: Directory containing input Parquet files with structures
        output_path: Directory for saving relaxed structure results
        column_name: Name of column containing CIF strings in input files

    Returns:
        list[submitit.Job]: List of submitted relaxation jobs for monitoring

    Workflow:
        1. Set up SLURM executor with GPU and memory requirements
        2. Discover all input Parquet files to process
        3. Filter out already-processed files to avoid duplication
        4. Distribute files across parallel ranks for load balancing
        5. Submit batch jobs to SLURM scheduler
        6. Return job objects for progress monitoring

    Note:
        Requires GPU-enabled compute nodes for efficient UMA model inference.
    """

    # Set up SLURM executor with GPU requirements
    executor = submitit.AutoExecutor(folder=output_path.parent / "slurm")
    executor.update_parameters(
        slurm_job_name=slurm_config.get("job-name", "relax"),
        timeout_min=slurm_config.get("time", 1000),
        gpus_per_node=slurm_config.get("gpus_per_node", 1),
        cpus_per_task=slurm_config.get("cpus_per_task", 10),
        mem_gb=slurm_config.get("mem_gb", 50),
    )

    # Discover all input files to process
    input_files = list(input_path.glob("**/*.parquet"))
    print(f"Total number of input files: {len(input_files)}")

    # Filter out files that have already been relaxed to avoid recomputation
    input_files = [
        file
        for file in input_files
        if not (output_path / file.relative_to(args.root)).exists()
    ]
    print(f"Number of input files to relax: {len(input_files)}")

    # Distribute work across parallel ranks and submit jobs
    jobs = []
    num_ranks = relax_config.get("num_ranks", 1000)
    with executor.batch():
        for rank in range(min(num_ranks, len(input_files))):
            input_files_rank = input_files[rank::num_ranks]
            job = executor.submit(
                relax_structures, args, input_files_rank, output_path, column_name
            )
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id}")
    return jobs


if __name__ == "__main__":
    """
    Example usage for structure relaxation.

    This example demonstrates how to run ML-based structure relaxation
    using UMA models on a dataset of generated structures.
    """
    import argparse

    import yaml

    # Set up argument parser for standalone execution
    parser = argparse.ArgumentParser(
        description="Crystal Structure Relaxations with UMA"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--input_path", required=True, help="Directory with structures to relax"
    )
    parser.add_argument(
        "--output_path", required=True, help="Directory for relaxed structures"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    # Execute structure relaxation
    jobs = run_relax_jobs(args, config, Path(args.input_path), Path(args.output_path))

    print(f"Started {len(jobs)} relaxation jobs")
    print("Use job.wait() or SLURM commands to monitor progress")
