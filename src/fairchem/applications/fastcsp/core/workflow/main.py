"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP - Fast Crystal Structure Prediction Workflow

This module provides the main orchestration script for the FastCSP (Fast Crystal Structure
Prediction) workflow, which combines Genarris crystal structure generation
with machine learning-based structure relaxation with the Universal Model for Atoms (UMA)
and validation against experimental data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fairchem.applications.fastcsp.core.dft.vasp import create_vasp_relaxation_jobs
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


def get_relax_path_and_parameters(config):
    """
    Generate descriptive output directory path based on relaxation parameters.

    Creates a unique output directory name that encodes the key computational
    parameters used for structure relaxation, enabling easy identification
    and comparison of different parameter sets.

    Args:
        config: Configuration dictionary containing relaxation parameters

    Returns:
        tuple: (output_dir, relax_params) where:
            - output_dir: Complete path to output directory
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
    relax_output_dir = f"{relax_params['calculator']}_{relax_params['optimizer']}_{relax_params['fmax']}_{relax_params['max_steps']}"
    if relax_params["fix_symmetry"]:
        relax_output_dir += "_fixsymm"
    if relax_params["relax_cell"]:
        relax_output_dir += "_relaxcell"

    relax_output_dir = root / "fastcsp_results" / relax_output_dir
    print("Output path:", relax_output_dir)
    return relax_output_dir, relax_params


def main(args: argparse.Namespace):
    """
    Main orchestration function for the FastCSP crystal structure prediction workflow.

    Executes the complete FastCSP workflow based on the specified stages.

    The workflow stages are:
    1. generate: Generate initial crystal structures using Genarris
    2. process_generated: Process and deduplicate generated structures
    3. relax: ML-based structure relaxation using fairchem models
    4. filter: Energy-based filtering and duplicate removal for initial ranking
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
        genarris → process_genarris → relax → filter → evaluate → free_energy
                                                ↓
        read_vasp_outputs ← submit_vasp ← create_vasp_inputs_relaxed
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
    print(f"Writing all outputs to {root}")

    # 1. Generate putative structures using Genarris
    if "generate" in args.stages:
        print("Starting Genarris generation...")
        jobs = run_genarris_jobs(output_dir=root / "genarris", config=config)
        wait_for_jobs(jobs)
        print(f"Finished Genarris generation with {len(jobs)} jobs.")

    # 2. Read Genarris outputs, deduplicate, and create Parquet files
    if "process_generated" in args.stages:
        print("Starting deduplication of Genarris structures...")
        pre_relax_match_params = config.get("pre_relax_match_params", {})
        jobs = process_genarris_outputs(
            input_dir=root / "genarris",
            output_dir=root / "raw_structures",
            ltol=pre_relax_match_params.get("ltol", 0.2),
            stol=pre_relax_match_params.get("stol", 0.3),
            angle_tol=pre_relax_match_params.get("angle_tol", 5),
            npartitions=pre_relax_match_params.get("npartitions", 1000),
        )
        wait_for_jobs(jobs)
        print(f"Finished deduplicating structures from Genarris with {len(jobs)} jobs.")

    # 3. Relax structures using UMA MLIP
    if "relax" in args.stages:
        print("Starting ML-relaxation of deduplicated structures...")
        relax_output_dir, relax_params = get_relax_path_and_parameters(config)
        jobs = run_relax_jobs(
            input_dir=root / "raw_structures",
            output_dir=relax_output_dir / "relaxed_structures",
            relax_config=relax_params,
            slurm_config=config["relax"].get("slurm", {}),
        )
        wait_for_jobs(jobs)
        print(f"Finished relaxing structures with {len(jobs)} jobs.")

    # 4. Filter, deduplicate, and rank structures
    if "filter" in args.stages:
        print("Starting filtering and deduplication of ML-relaxed structures...")
        jobs = filter_and_deduplicate_structures(
            input_dir=relax_output_dir / "raw_structures",
            output_dir=relax_output_dir / "filtered_structures",
            energy_cutoff_kj_per_mol=config["filter"]["energy_cutoff"],
            density_cutoff=config["filter"]["density_cutoff"],
            ltol=config["post_relax_match_params"]["ltol"],
            stol=config["post_relax_match_params"]["stol"],
            angle_tol=config["post_relax_match_params"]["angle_tol"],
        )
        wait_for_jobs(jobs)
        print(
            "Finished filtering and deduplication of ML-relaxed structures with {len(jobs)} jobs."
        )

    # 5. (Optional) Compare predicted structures to experimental
    # using either CSD API or pymatgen StructureMatcher
    if "evaluate" in args.stages:
        print("Starting evaluating for structure matches to experimental structures...")
        compute_structure_matches(
            input_dir=relax_output_dir / "filtered_structures",
            output_dir=relax_output_dir / "matched_structures",
            molecules_file=config["molecules"],
            method=config["evaluate"]["method"],
        )
        print("Finished evaluation against experimental structures.")

    # 6. (Optional) Calculate free energies for structures
    # TODO: Implementation in progress - will be available soon
    if "free_energy" in args.stages:
        print("Free energy calculations requested...")
        calculate_free_energies(
            relax_output_dir / "matched_structures",
            relax_output_dir / "free_energy_results",
            config,
        )
        print("Finished free energy calculations.")

    # 7. (Optional) Create VASP input files for DFT validation
    # Generates VASP inputs for ML-relaxed structures
    if "create_vasp_inputs_relaxed" in args.stages:
        print("Creating VASP inputs for ML-relaxed structures.")
        create_vasp_relaxation_jobs(
            relax_output_dir / "matched_structures",
            relax_output_dir / "vasp_inputs",
            sym_prec=config["vasp"].get("sym_prec", 1e-5),
        )
        print("Finished creating VASP inputs for ML-relaxed structures.")

    # Generate VASP inputs for unrelaxed structures for comparison
    if "create_vasp_inputs_unrelaxed" in args.stages:
        print("Creating VASP inputs for unrelaxed structures.")
        create_vasp_relaxation_jobs(
            relax_output_dir / "matched_structures",
            relax_output_dir / "vasp_inputs_unrelaxed",
            root / "raw_structures",
            unrelaxed=True,
            sym_prec=config["vasp"]["sym_prec"],
        )
        print("Finished creating VASP inputs for unrelaxed structures.")

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
        # submit_vasp_jobs(relax_output_dir / "vasp_inputs", relax_output_dir / "vasp_jobs.txt")

    # 9. Read VASP outputs and compute matches against experimental data
    if "read_vasp_outputs" in args.stages:
        # read VASP outputs
        jobs = collate_vasp_outputs(
            relax_output_dir / "matched_structures",
            relax_output_dir / "vasp_inputs",
            relax_output_dir / "vasp_structures",
        )
        wait_for_jobs(jobs)

        # Compute matches for DFT-relaxed structures
        compute_structure_matches(
            relax_output_dir / "vasp_structures",
            relax_output_dir / "vasp_matched_structures",
            config["molecules"],
            config,
        )
