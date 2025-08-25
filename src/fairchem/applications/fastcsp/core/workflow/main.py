"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP - Fast Crystal Structure Prediction Workflow

This module provides the main orchestration script for the FastCSP (Fast Crystal Structure
Prediction) workflow, which combines Genarris crystal structure generation
with machine learning-based structure relaxation with the Universal Model for Atoms (UMA)
and validation against experimental data.

The workflow consists of several stages:
1. Generate: Generate initial crystal structures for molecular compounds
2. Process Generated: Process and deduplicate generated structures
3. Relax: Perform ML-based structure relaxation using FAIRChem models
4. Rank: Energy-based filtering and duplicate removal of relaxed structures for final ranking
5. Evaluate: Compare predicted structures against experimental references
6. VASP: Create DFT input files and perform high-accuracy validation
7. Analysis: Collate and analyze final results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fairchem.applications.fastcsp.core.dft.vasp import (
    create_vasp_relaxation_jobs,
)
from fairchem.applications.fastcsp.core.dft.vasp_utils import collate_vasp_outputs
from fairchem.applications.fastcsp.core.utils.configuration import (
    reorder_stages_by_dependencies,
    validate_config,
)
from fairchem.applications.fastcsp.core.utils.slurm import wait_for_jobs
from fairchem.applications.fastcsp.core.workflow.eval import compute_structure_matches
from fairchem.applications.fastcsp.core.workflow.filter import (
    filter_and_deduplicate_structures,
)
from fairchem.applications.fastcsp.core.workflow.free_energy import (
    calculate_free_energies,
)
from fairchem.applications.fastcsp.core.workflow.generate import run_genarris_jobs
from fairchem.applications.fastcsp.core.workflow.process_generated import (
    process_genarris_outputs,
)
from fairchem.applications.fastcsp.core.workflow.relax import run_relax_jobs

if TYPE_CHECKING:
    import argparse


def load_config(args: argparse.Namespace):
    """
    Load and parse configuration file for the FastCSP workflow.

    Args:
        args: Command line arguments namespace containing config file path

    Returns:
        dict: Parsed configuration dictionary
    """
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    # Validate configuration completeness
    validate_config(config, args.stages)

    return config


def get_output_path(config):
    """
    Generate descriptive output directory path based on relaxation parameters.

    Creates a unique output directory name that encodes the key computational
    parameters used for structure relaxation, enabling easy identification
    and comparison of different parameter sets.

    Args:
        config: Configuration dictionary containing relaxation parameters

    Returns:
        tuple: (output_path, relax_params) where:
            - output_path: Complete path to output directory
            - relax_params: Dictionary of relaxation parameters for downstream use

    Example:
        For calculator='FAIRChem', optimizer='BFGS', fmax=0.01, max_steps=500:
        Returns: (/path/to/root/fastcsp_results/FAIRChem_BFGS_0.01_500_fixsymm_relaxcell, {...})
    """
    root = Path(config["root"]).resolve()
    relax_config = config["relax"]

    # Extract relaxation parameters from config
    relax_params = {
        "root": root,
        "calculator": relax_config["calculator"],
        "optimizer": relax_config["optimizer"].lower(),
        "fmax": relax_config["fmax"],
        "max_steps": relax_config["max-steps"],
        "fix_symmetry": relax_config["fix-symmetry"],
        "relax_cell": relax_config["relax-cell"],
    }

    # Build descriptive directory name from parameters
    output_name = f"{relax_params['calculator']}_{relax_params['optimizer']}_{relax_params['fmax']}_{relax_params['max_steps']}"
    if relax_params["fix_symmetry"]:
        output_name += "_fixsymm"
    if relax_params["relax_cell"]:
        output_name += "_relaxcell"

    output_path = root / "fastcsp_results" / output_name
    print("Output path:", output_path)
    return output_path, relax_params


def main(args: argparse.Namespace):
    """
    Main orchestration function for the FastCSP crystal structure prediction workflow.

    Executes the complete FastCSP workflow based on the specified stages.

    The workflow stages are:
    1. generate: Generate initial crystal structures using Genarris
    2. process_generated: Process and deduplicate generated structures
    3. relax: ML-based structure relaxation using fairchem models
    4. rank: Energy-based filtering and duplicate removal for final ranking
    5. evaluate: Compare against experimental references (requires CSD license)
    6. free_energy: Calculate free energies for improved ranking (TODO: in development)
    7. create_vasp_inputs: Generate DFT input files for validation
    8. submit_vasp: Submit VASP jobs (user-customizable for specific clusters)
    9. read_vasp_outputs: Process DFT results and compute final matches

    Args:
        args: Command line arguments containing:
            - config: Path to YAML configuration file
            - stages: List of pipeline stages to execute

    Workflow:
        Each stage depends on outputs from previous stages:
        genarris → process_genarris → relax → rank → evaluate → free_energy
                                              ↓
        read_vasp_outputs ← submit_vasp ← create_vasp_inputs_relaxed

    Note:
        The function automatically reorders stages to match the canonical workflow
        order but does not add missing dependency stages. Users must explicitly
        specify all required stages.
    """
    print("Starting FastCSP workflow...")
    print("Stages requested:", args.stages)

    # Reorder stages to match canonical workflow order
    ordered_stages = reorder_stages_by_dependencies(args.stages)
    args.stages = ordered_stages  # Update args with reordered stages
    print("Stages to execute (final order):", args.stages)

    # Load configuration and set up directory structure
    config = load_config(args)
    print(f"Running FastCSP workflow with {config} config")

    root = Path(config["root"]).resolve()
    output_path, relax_params = get_output_path(config)

    # 1. Generate putative structures using Genarris
    if "generate" in args.stages:
        jobs = run_genarris_jobs(output_dir=output_path / "genarris", config=config)
        wait_for_jobs(jobs)

    # 2. Read Genarris outputs, deduplicate and create Parquet files
    if "process_generated" in args.stages:
        jobs = process_genarris_outputs(
            root=output_path / "genarris",
            output_dir=output_path / "raw_structures",
            ltol=config["pre_relax_match_params"]["ltol"],
            stol=config["pre_relax_match_params"]["stol"],
            angle_tol=config["pre_relax_match_params"]["angle_tol"],
            npartitions=config.get("npartitions", 1000),
        )
        wait_for_jobs(jobs)

    # 3. Relax structures using UMA MLIP
    if "relax" in args.stages:
        jobs = run_relax_jobs(
            input_path=root / "raw_structures",
            output_path=output_path / "relaxed_structures",
            relax_config=relax_params,
            slurm_config=config["relax"].get("slurm", {}),
        )
        wait_for_jobs(jobs)

    # 4. Filter, deduplicate, and rank structures
    if "rank" in args.stages:
        jobs = filter_and_deduplicate_structures(
            root=output_path / "raw_structures",
            output_path=output_path / "filtered_structures",
            energy_cutoff_kj_per_mol=config["energy_cutoff"],
            # root_unrelaxed=Path(config["root"]) / "raw_structures",  # TODO: remove this
            root_unrelaxed=None,
            ltol=config["post_relax_match_params"]["ltol"],
            stol=config["post_relax_match_params"]["stol"],
            angle_tol=config["post_relax_match_params"]["angle_tol"],
            density_cutoff=config["density_cutoff"],
            outlier_removal_method=config.get("outlier_removal_method", None),
        )
        wait_for_jobs(jobs)

    # 5. Compute structure matches using either CSD API or pymatgen StructureMatcher
    # Compares predicted structures against experimental references
    if "evaluate" in args.stages:
        compute_structure_matches(
            output_path / "filtered_structures",
            output_path / "matched_structures",
            config["molecules"],
            config,  # Pass config to enable method selection
        )

    # 6. Calculate free energies for structures
    # TODO: Implementation in progress - will be available soon
    if "free_energy" in args.stages:
        print("Free energy calculations requested...")
        calculate_free_energies(
            output_path / "matched_structures",
            output_path / "free_energy_results",
            config,
        )

    # 7. (Optional) Create VASP input files for DFT validation
    # Generates VASP inputs for ML-relaxed structures
    if "create_vasp_inputs_relaxed" in args.stages:
        create_vasp_relaxation_jobs(
            output_path / "matched_structures",
            output_path / "vasp_inputs",
            sym_prec=config["vasp"].get("sym_prec", 1e-5),
        )

    # Generate VASP inputs for unrelaxed structures for comparison
    if "create_vasp_inputs_unrelaxed" in args.stages:
        create_vasp_relaxation_jobs(
            output_path / "matched_structures",
            output_path / "vasp_inputs_unrelaxed",
            root / "raw_structures",
            unrelaxed=True,
            sym_prec=config["vasp"]["sym_prec"],
        )

    # 8. Submit VASP jobs
    # Users should implement their own function for their specific setup
    if "submit_vasp" in args.stages:
        print("VASP job submission requested...")
        print("Please implement your own VASP job submission function.")
        print(
            "This stage is intentionally left for users to customize based on their cluster environment."
        )
        print(
            "You can use the VASP input files created in the 'vasp_inputs' directory."
        )
        print(
            "Example: modify submit_vasp_jobs() in fairchem.applications.fastcsp.core.dft.vasp"
        )
        print("or create your own submission script for your job scheduler.")

        # Uncomment and modify this line to use your VASP submission function:
        # submit_vasp_jobs(output_path / "vasp_inputs", output_path / "vasp_jobs.txt")

    # 9. Read VASP outputs and compute matches against experimental data
    if "read_vasp_outputs" in args.stages:
        # read VASP outputs
        jobs = collate_vasp_outputs(
            output_path / "matched_structures",
            output_path / "vasp_inputs",
            output_path / "vasp_structures",
        )
        wait_for_jobs(jobs)

        # Compute matches for DFT-relaxed structures
        compute_structure_matches(
            output_path / "vasp_structures",
            output_path / "vasp_matched_structures",
            config["molecules"],
            config,
        )
