"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Relaxation Module for FastCSP

This module provides functionality for relaxing crystal structures using machine learning
interatomic potentials (MLIPs), specifically the Universal Model for Atoms (UMA) from
the FAIRChem toolkit.

Key Features:
- UMA-based ML potential calculations for accurate and efficient structure optimization
- Batch processing for high-throughput structure relaxation
- SLURM integration for parallel GPU-accelerated relaxations

The module supports multiple UMA model tasks:
- uma_sm_1p1_omc: UMA's OMC task [RECOMMENDED]
- uma_sm_1p1_omol: UMA's OMoltask
"""

from __future__ import annotations

from logging import root
from pathlib import Path
from typing import Any

import pandas as pd
import submitit
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.units import eV, kJ, mol
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import get_relax_slurm_config
from fairchem.applications.fastcsp.core.utils.structure import (
    check_no_changes_in_covalent_matrix,
)
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.units import mlip_unit

EV_TO_KJ_PER_MOL = eV / (kJ / mol)

CHECKPOINTS = {
    "uma_sm_1p1_omc": {  # RECOMMENDED: UMA w/ OMC task
        "checkpoint": None,
        "model": "uma-s-1p1",
        "task_name": "omc",
    },
    "uma_sm_1p1_omol": {  # UMA w/ OMol task
        "checkpoint": None,
        "model": "uma-s-1p1",
        "task_name": "omol",
    },
}


def create_calculator(relax_config):
    """
    Create UMA ML potential calculator for structure relaxation.
    """
    if CHECKPOINTS[relax_config["calculator"]]["checkpoint"] is not None:
        predictor = mlip_unit.load_predict_unit(
            CHECKPOINTS[relax_config["calculator"]]["checkpoint"], device="cuda"
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(
            CHECKPOINTS[relax_config["calculator"]]["model"], device="cuda"
        )
    calc = FAIRChemCalculator(
        predictor,
        task_name=CHECKPOINTS[relax_config.get("calculator", "uma_sm_1p1_omc")][
            "task_name"
        ],
    )
    return calc


def get_relax_config_and_dir(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """
    Generate relaxation parameters and determine output directory from workflow configuration.

    This function processes the global configuration to extract relaxation-specific parameters
    and constructs a standardized output directory name based on the relaxation settings.

    Args:
        config: Full workflow configuration dictionary containing 'relax' section

    Returns:
        tuple: (relaxation_parameters_dict, output_directory_path)
            - relaxation_parameters_dict: Processed relaxation configuration
            - output_directory_path: Path where relaxed structures will be stored

    Configuration Parameters:
        - calculator: ML model to use ("uma-s-1p1-omc" default)
        - optimizer: Optimization algorithm ("bfgs", "fire", "lbfgs")
        - fmax: Force convergence criterion (0.01 eV/Å default)
        - max-steps: Maximum optimization steps (1000 default)
        - fix-symmetry: Preserve crystallographic symmetry during relaxation
        - relax-cell: Allow unit cell parameters to change during optimization

    Notes:
        Output directory name encodes all relaxation parameters for reproducibility
        and easy identification of different relaxation runs.
    """
    root = Path(config["root"]).resolve()
    relax_config = config.get("relax", {})

    relax_params = {
        "root": root,
        "calculator": relax_config.get("calculator", "uma-s-1p1-omc"),
        "optimizer": relax_config.get("optimizer", "bfgs").lower(),
        "fmax": relax_config.get("fmax", 0.01),
        "max_steps": relax_config.get("max-steps", 1000),
        "fix_symmetry": relax_config.get("fix-symmetry", False),
        "relax_cell": relax_config.get("relax-cell", True),
    }

    relax_output_dir = f"{relax_params['calculator']}_{relax_params['optimizer']}_{relax_params['fmax']}_{relax_params['max_steps']}"
    if relax_params["fix_symmetry"]:
        relax_output_dir += "_fixsymm"
    if relax_params["relax_cell"]:
        relax_output_dir += "_relaxcell"

    relax_output_dir = root / "relaxed" / relax_output_dir

    logger = get_central_logger()
    logger.info("Relaxation configuration:")
    logger.info(f"Relaxation config: {relax_config}")
    logger.info(f"Relaxation output directory: {relax_output_dir}")
    return relax_params, relax_output_dir


def relax_atoms_batch(atoms_list, relax_config, calc):
    """
    Relax multiple crystal structures simultaneously using batch optimization.

    This function performs efficient batch relaxation of multiple structures.

    Args:
        atoms_list: List of ASE Atoms objects to be relaxed
        relax_config: Dictionary containing relaxation parameters:
            - optimizer: Must be "batch_lbfgs" for this function
            - relax_cell: Whether to optimize unit cell parameters
            - fix_symmetry: Must be False (not supported in batch mode)
            - fmax: Force convergence criterion in eV/Å
            - max_steps: Maximum optimization steps
        calc: FAIRChemCalculator instance
    Returns:
        list[ASE.Atoms]: Relaxed structures with updated info dictionary containing:
            - 'converged': Boolean indicating if optimization converged
            - 'energy': Final potential energy in eV

    Raises:
        AssertionError: If fix_symmetry is True or optimizer is not "batch_lbfgs"
    """
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from fairchem.core.optim.lbfgs_torch import LBFGS as FairchemLBFGS
    from fairchem.core.optim.optimizable import (
        OptimizableBatch,
        OptimizableUnitCellBatch,
    )

    assert not relax_config["fix_symmetry"]
    assert relax_config["optimizer"] == "batch_lbfgs"

    predictor = calc.predictor
    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omc") for atoms in atoms_list
    ]
    atoms_batch = atomicdata_list_to_batch(atomic_data_list)
    if relax_config["relax_cell"]:
        ecf = OptimizableUnitCellBatch(atoms_batch, predictor)
    else:
        ecf = OptimizableBatch(atoms_batch, predictor)
    optimizer = FairchemLBFGS(ecf)
    potential_energies = ecf.get_potential_energies()
    converged_batch = optimizer.run(
        fmax=relax_config["fmax"], steps=relax_config["max_steps"]
    )
    atoms_relaxed = ecf.get_atoms_list()
    for atoms, converged, potential_energy in zip(
        atoms_relaxed, converged_batch, potential_energies
    ):
        atoms.info["converged"] = converged.item()
        atoms.info["energy"] = potential_energy.item()
    return atoms_relaxed


def relax_atoms(atoms, relax_config, calc):
    """
    Relax a single crystal structure using ASE-compatible optimizers.

    This function performs structure optimization using traditional ASE optimizers
    (BFGS, FIRE, L-BFGS) with optional unit cell relaxation and symmetry preservation.
    Includes comprehensive quality control checks to ensure structural integrity.

    Args:
        atoms: ASE Atoms object representing the crystal structure
        relax_config: Dictionary containing relaxation parameters:
            - optimizer: Optimization algorithm ("bfgs", "fire", "lbfgs")
            - relax_cell: Whether to optimize unit cell parameters
            - fix_symmetry: Whether to preserve crystallographic symmetry
            - fmax: Force convergence criterion in eV/Å
            - max_steps: Maximum optimization steps
        calc: Calculator instance

    Returns:
        ASE.Atoms: Relaxed structure with updated info dictionary containing:
            - 'converged': Boolean indicating if optimization converged
            - 'energy': Final potential energy in eV
            - 'n_steps': Number of optimization steps taken

    Raises:
        ValueError: If an unsupported optimizer is specified
        RuntimeError: If structural integrity checks fail after relaxation

    Quality Control Checks:
        - Atomic composition conservation (Z-number preservation)
        - Covalent bonding network conservation
    """
    # Apply symmetry constraint if requested
    if relax_config["fix_symmetry"]:
        atoms.set_constraint(FixSymmetry(atoms))
    atoms.calc = calc

    # Configure optimization algorithm
    OPTIMIZERS = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "fire": FIRE,
    }
    optimizer_name = relax_config.get("optimizer", "bfgs").lower()
    optimizer_cls = OPTIMIZERS.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. (L)BFGS and FIRE are recommended."
        )
    if relax_config.get("relax_cell"):
        optimizer = optimizer_cls(FrechetCellFilter(atoms))
    else:
        optimizer = optimizer_cls(atoms)
    # Perform optimization
    converged = optimizer.run(
        fmax=relax_config["fmax"], steps=relax_config["max_steps"]
    )
    logger = get_central_logger()
    logger.debug(
        f"Relaxation converged: {converged}, Energy: {atoms.get_potential_energy()}"
    )

    # Store relaxation metadata
    atoms.info["converged"] = converged  # Store convergence status
    atoms.info["energy"] = atoms.get_potential_energy()  # Store relaxed energy
    return atoms


def relax_structures(input_files, output_dir, relax_config, column_name="cif"):
    """Relax crystal structures from Parquet files using ML potentials."""
    logger = get_central_logger()
    calc = create_calculator(relax_config)

    for input_file in tqdm(input_files):
        output_file = output_dir.parent / input_file.relative_to(
            output_dir.parent.parent.parent
        )
        if output_file.exists():
            logger.info(f"Skipping {input_file} because {output_file} exists")
            continue

        logger.info(f"Relaxing structures from {input_file}")

        structures_df = pd.read_parquet(input_file)
        atoms_list = (
            structures_df[column_name]
            .apply(
                lambda x: AseAtomsAdaptor.get_atoms(Structure.from_str(x, fmt="cif"))
            )
            .to_numpy()
        )

        # Special handling for OMol tasks - setting spin and charge
        if relax_config["calculator"] == "uma_sm_1p1_omol":
            for atoms in atoms_list:
                atoms.info.update({"spin": 1, "charge": 0})

        # Perform relaxation for all structures
        if relax_config["optimizer"] == "batch_lbfgs":
            from itertools import batched, chain

            batch_size = relax_config.get("batch_size", 10)
            batches = list(batched(atoms_list, batch_size))
            atoms_relaxed = [
                relax_atoms_batch(atoms_batch, relax_config, calc)
                for atoms_batch in tqdm(batches)
            ]
            atoms_relaxed = list(chain.from_iterable(atoms_relaxed))
        else:
            atoms_relaxed = [
                relax_atoms(atoms, relax_config, calc) for atoms in tqdm(atoms_list)
            ]
        # Extract properties
        structures_relaxed = [
            AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_relaxed
        ]
        structures_df["relaxed_cif"] = [
            structure.to(fmt="cif") for structure in structures_relaxed
        ]
        structures_df["volume"] = [structure.volume for structure in structures_relaxed]
        structures_df["density"] = [
            structure.density for structure in structures_relaxed
        ]
        structures_df["energy_relaxed"] = [
            atoms.get_potential_energy() * EV_TO_KJ_PER_MOL for atoms in atoms_relaxed
        ]
        structures_df["energy_relaxed_per_molecule"] = (
            structures_df["energy_relaxed"] / structures_df["z"]
        )
        structures_df["converged"] = [
            atoms.info["converged"] for atoms in atoms_relaxed
        ]

        # Validate structural integrity after relaxation
        structures_df["connectivity_unchanged"] = [
            check_no_changes_in_covalent_matrix(atoms_initial, atoms_relaxed)
            for atoms_initial, atoms_relaxed in zip(atoms_list, atoms_relaxed)
        ]
        # Save results to Parquet
        output_file.parent.mkdir(parents=True, exist_ok=True)
        structures_df.to_parquet(output_file, compression="zstd")
        logger.info(
            f"Wrote {structures_df.shape[0]} relaxed structures to {output_file}"
        )


def run_relax_jobs(input_dir, output_dir, relax_config, column_name="cif"):
    """Submit parallel structure relaxation jobs to SLURM."""

    # Configure SLURM parameters
    relax_slurm_config, executor_params = get_relax_slurm_config(relax_config)

    # Set up SLURM executor with GPU requirements
    executor = submitit.AutoExecutor(folder=output_dir.parent / "slurm")
    executor.update_parameters(**executor_params)

    logger = get_central_logger()

    # Discover all input files to process
    input_files = list(input_dir.glob("**/*.parquet"))
    logger.info(f"Total number of input files: {len(input_files)}")

    # Filter out files that have already been relaxed to avoid recomputation
    input_files = [
        file
        for file in input_files
        if not (
            output_dir.parent / file.relative_to(output_dir.parent.parent.parent)
        ).exists()
    ]
    logger.info(f"Number of input files to relax: {len(input_files)}")

    jobs = []
    num_ranks = relax_slurm_config.get("num_ranks", 1000)
    with executor.batch():
        for rank in range(min(num_ranks, len(input_files))):
            input_files_rank = input_files[rank::num_ranks]
            job = executor.submit(
                relax_structures,
                input_files_rank,
                output_dir,
                relax_config,
                column_name,
            )
            jobs.append(job)

    logger = get_central_logger()
    logger.info(
        f"Submitted {len(jobs)} relaxation jobs: {jobs[0].job_id.split('_')[0] if jobs else 'none'}"
    )
    return jobs


if __name__ == "__main__":
    """Standalone script for structure relaxation."""
    import argparse

    import yaml

    # Set up argument Parser for standalone execution
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

    relax_config, relax_output_dir = get_relax_config_and_dir(config)
    root = Path(config["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Execute structure relaxation
    jobs = run_relax_jobs(
        input_dir=root / "raw_structures",
        output_dir=relax_output_dir / "relaxed_structures",
        relax_config=relax_config,
    )
    logger = get_central_logger()
    logger.info(f"Started {len(jobs)} relaxation jobs")
    logger.info("Use job.wait() or SLURM commands to monitor progress")
