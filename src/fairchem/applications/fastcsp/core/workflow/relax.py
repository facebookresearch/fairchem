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

import warnings
from logging import root
from pathlib import Path
from typing import Any

import pandas as pd
import submitit
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS
from ase.units import eV, kJ, mol
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import get_relax_slurm_config
from fairchem.applications.fastcsp.core.utils.structure import (
    check_connectivity_unchanged,
    check_correct_z,
    check_molecule_matches_reference,
    load_reference_graph,
)
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.units import mlip_unit

# Suppress scipy logm numerical precision warnings during structure relaxation
# These warnings occur in matrix operations but the errors (~1e-13) are negligible
warnings.filterwarnings(
    "ignore",
    message="logm result may be inaccurate",
    category=RuntimeWarning,
)

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
    "uma_sm_1p2_omc": {  # UMA 1p2 w/ OMC task
        "checkpoint": None,
        "model": "uma-s-1p2",
        "task_name": "omc",
    },
    "uma_sm_1p2_omol": {  # UMA 1p2 w/ OMol task
        "checkpoint": None,
        "model": "uma-s-1p2",
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


def get_relax_config_and_dir(
    config: dict[str, Any], verbose=False
) -> tuple[dict[str, Any], Path]:
    """
    Extract relaxation parameters and construct output directory path.

    Args:
        config: Workflow configuration containing 'relax' section.
        verbose: Log configuration details if True.

    Returns:
        Tuple of (relaxation_params, output_directory_path).
    """
    root = Path(config["root"]).resolve()
    relax_config = config.get("relax", {})

    relax_params = {
        "root": root,
        "calculator": relax_config.get("calculator", "uma-s-1p1-omc"),
        "optimizer": relax_config.get("optimizer", "bfgs").lower(),
        "fmax": relax_config.get("fmax", 0.01),
        "max_steps": relax_config.get("max_steps", relax_config.get("max-steps", 1000)),
        "fix_symmetry": relax_config.get(
            "fix_symmetry", relax_config.get("fix-symmetry", False)
        ),
        "relax_cell": relax_config.get(
            "relax_cell", relax_config.get("relax-cell", True)
        ),
        "write_traj": relax_config.get(
            "write_traj", relax_config.get("write-traj", False)
        ),
        "traj_interval": relax_config.get(
            "traj_interval", relax_config.get("traj-interval", 1)
        ),
        "slurm": relax_config.get("slurm", {}),
    }

    relax_output_dir = f"{relax_params['calculator']}_{relax_params['optimizer']}_{relax_params['fmax']}_{relax_params['max_steps']}"
    if relax_params["fix_symmetry"]:
        relax_output_dir += "_fixsymm"
    if relax_params["relax_cell"]:
        relax_output_dir += "_relaxcell"

    relax_output_dir = root / "relaxed" / relax_output_dir

    if verbose:
        logger = get_central_logger()
        logger.info("Relaxation configuration:")
        logger.info(f"Relaxation config: {relax_config}")
        logger.info(f"Relaxation output directory: {relax_output_dir}")
    return relax_params, relax_output_dir


def relax_atoms_batch(atoms_list, relax_config, calc):
    """
    Relax multiple structures simultaneously using batch L-BFGS optimization.

    Args:
        atoms_list: List of ASE Atoms objects.
        relax_config: Relaxation parameters (must have optimizer="batch_lbfgs").
        calc: FAIRChemCalculator instance.

    Returns:
        List of relaxed Atoms with 'converged' and 'energy' in info dict.
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
    converged_batch = optimizer.run(
        fmax=relax_config["fmax"], steps=relax_config["max_steps"]
    )
    potential_energies = ecf.get_potential_energies()
    atoms_relaxed = ecf.get_atoms_list()
    for atoms, converged, potential_energy in zip(
        atoms_relaxed, converged_batch, potential_energies
    ):
        atoms.info["converged"] = converged.item()
        atoms.info["energy"] = potential_energy.item()
    return atoms_relaxed


def relax_atoms(atoms, relax_config, calc):
    """
    Relax a single structure using ASE optimizers (BFGS, FIRE, L-BFGS).

    Args:
        atoms: ASE Atoms object.
        relax_config: Dict with optimizer, fmax, max_steps, relax_cell, fix_symmetry.
        calc: Calculator instance.

    Returns:
        Relaxed Atoms with 'converged', 'energy', 'optimizer_steps' in info dict.
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
        optimizer = optimizer_cls(FrechetCellFilter(atoms), logfile=None)
    else:
        optimizer = optimizer_cls(atoms, logfile=None)

    # Perform optimization
    if relax_config.get("write_traj"):
        traj = Trajectory(atoms.info["traj_path"], "w", atoms)
        optimizer.attach(traj.write, interval=relax_config.get("traj_interval"))
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
    atoms.info["optimizer_steps"] = (
        optimizer.nsteps
    )  # Store number of optimization steps

    if relax_config.get("write_traj"):
        traj.close()
    return atoms


def relax_structures(
    input_files,
    output_dir,
    relax_config,
    column_name="cif_generated",
    generated_structures_dir=None,
):
    """Relax crystal structures from Parquet files using ML potentials."""
    logger = get_central_logger()
    calc = create_calculator(relax_config)

    for input_file in tqdm(input_files):
        output_file = output_dir.parent / input_file.relative_to(
            output_dir.parent.parent.parent
        )
        if output_file.exists():
            logger.debug(f"Skipping {input_file} because {output_file} exists")
            continue

        logger.debug(f"Relaxing structures from {input_file}")

        structures_df = pd.read_parquet(input_file)

        # Build the reference-molecule graph once per (mol, conf). The reference
        # XYZ lives at <generated_structures>/<mol>/<conf>/<conf>.xyz (.sdf, .mol
        # also accepted). When generated_structures_dir is not provided, fall
        # back to deriving it from input_file's location.
        mol_id = str(structures_df["mol_id"].iloc[0])
        conf_id = str(structures_df["conf_id"].iloc[0])
        if generated_structures_dir is None:
            # input_file: <root>/raw_structures/<mol>/<conf>/partition_id=*/*.parquet
            try:
                derived_root = input_file.parents[4]
                generated_conf_dir = (
                    derived_root / "generated_structures" / mol_id / conf_id
                )
            except IndexError:
                generated_conf_dir = None
        else:
            generated_conf_dir = Path(generated_structures_dir) / mol_id / conf_id
        reference_graph = load_reference_graph(generated_conf_dir, conf_id)

        atoms_list = (
            structures_df[column_name]
            .apply(
                lambda x: AseAtomsAdaptor.get_atoms(Structure.from_str(x, fmt="cif"))
            )
            .to_numpy()
        )
        # Deep-copy the pre-relax atoms so the post-relax bond-matrix
        # comparison (check_connectivity_unchanged) has the original
        # JmolNN adjacency on hand. relax_atoms() mutates atoms in place.
        atoms_list_original = [atoms.copy() for atoms in atoms_list]
        structure_ids_list = structures_df["structure_id"].to_numpy().astype(str)

        # Create traj folder
        if relax_config.get("write_traj"):
            traj_folder = (
                output_dir.parent
                / "relaxation_trajectories"
                / input_file.parent.relative_to(output_dir.parent.parent.parent)
            )
            traj_folder.mkdir(parents=True, exist_ok=True)
            for structure_id, atoms in zip(structure_ids_list, atoms_list):
                traj_path = traj_folder / f"{structure_id}.traj"
                atoms.info["traj_path"] = str(traj_path)

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
        structures_df["cif_relaxed"] = [
            structure.to(fmt="cif") for structure in structures_relaxed
        ]
        structures_df["volume_relaxed"] = [
            structure.volume for structure in structures_relaxed
        ]
        structures_df["density_relaxed"] = [
            structure.density for structure in structures_relaxed
        ]
        structures_df["energy_relaxed"] = [
            atoms.get_potential_energy() * EV_TO_KJ_PER_MOL for atoms in atoms_relaxed
        ]
        structures_df["energy_relaxed_per_molecule"] = (
            structures_df["energy_relaxed"] / structures_df["z"]
        )
        structures_df["optimizer_steps"] = [
            atoms.info["optimizer_steps"] for atoms in atoms_relaxed
        ]
        structures_df["optimizer_converged"] = [
            atoms.info["converged"] for atoms in atoms_relaxed
        ]

        # Validate structural integrity after relaxation using reference-anchored
        # checks on the relaxed structure: (a) Z (molecule count) matches the
        # canonical Z value, and (b) every fragment is isomorphic to the
        # reference molecular graph.
        structures_df["validity.crystal_relaxed.correct_z"] = [
            check_correct_z(structure, int(z))
            for structure, z in zip(structures_relaxed, structures_df["z"])
        ]
        structures_df["validity.crystal_relaxed.molecule_matches_reference"] = [
            check_molecule_matches_reference(structure, reference_graph)
            for structure in structures_relaxed
        ]
        # Strict init↔final bond-matrix equality (no site permutation).
        # Catches any bond broken / formed during relax, independent of the
        # reference-anchored checks above.
        structures_df["validity.crystal_relaxed.connectivity_unchanged"] = [
            check_connectivity_unchanged(atoms_initial, atoms_final)
            for atoms_initial, atoms_final in zip(atoms_list_original, atoms_relaxed)
        ]
        # Save results to Parquet
        output_file.parent.mkdir(parents=True, exist_ok=True)
        structures_df.to_parquet(output_file, compression="zstd")
        logger.debug(
            f"Wrote {structures_df.shape[0]} relaxed structures to {output_file}"
        )


def run_relax_jobs(
    input_dir,
    output_dir,
    relax_config,
    column_name="cif_generated",
    generated_structures_dir=None,
):
    """Submit parallel structure relaxation jobs to SLURM.

    Args:
        input_dir: Directory containing input parquet files (e.g., raw_structures/).
        output_dir: Destination for relaxed parquet output.
        relax_config: Relaxation parameters.
        column_name: Name of the input CIF column.
        generated_structures_dir: Optional path to the workspace's
            generated_structures/ directory; used by each worker to locate the
            per-conformer reference XYZ for the relaxed-side validity flags.
            If None, the worker derives it from input_file.parents[4].
    """

    logger = get_central_logger()

    # Configure SLURM parameters
    relax_slurm_config, executor_params = get_relax_slurm_config(relax_config)

    # Set up SLURM executor with GPU requirements
    executor = submitit.AutoExecutor(folder=output_dir.parent / "slurm")
    executor.update_parameters(**executor_params)

    # Discover all input files to process
    input_files = list(input_dir.glob("**/*.parquet"))
    logger.info(f"Total number of input parquet files found: {len(input_files)}")

    # Filter out files that have already been relaxed to avoid recomputation
    input_files = [
        file
        for file in input_files
        if not (
            output_dir.parent / file.relative_to(output_dir.parent.parent.parent)
        ).exists()
    ]
    logger.info(f"Number of input parquet files to relax: {len(input_files)}")

    jobs = []
    num_ranks = relax_slurm_config.get("num_ranks", 25000)
    with executor.batch():
        for rank in range(min(num_ranks, len(input_files))):
            input_files_rank = input_files[rank::num_ranks]
            job = executor.submit(
                relax_structures,
                input_files_rank,
                output_dir,
                relax_config,
                column_name,
                generated_structures_dir,
            )
            jobs.append(job)

    logger = get_central_logger()
    logger.info(
        f"Submitted {len(jobs)} relaxation array jobs with job-id: {jobs[0].job_id.split('_')[0] if jobs else ''}"
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
        column_name="cif_generated",
        generated_structures_dir=root / "generated_structures",
    )
    logger = get_central_logger()
    logger.info(f"Started {len(jobs)} relaxation jobs")
    logger.info("Use job.wait() or SLURM commands to monitor progress")
