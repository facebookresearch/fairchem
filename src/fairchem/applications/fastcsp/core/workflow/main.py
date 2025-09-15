"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP - Fast Crystal Structure Prediction Workflow

This module provides the main orchestration script for the FastCSP (Fast Crystal Structure
Prediction) workflow. It coordinates the execution of all workflow stages including structure
generation, relaxation, filtering, evaluation, and optional DFT validation.

Key Features:
- Stage-based workflow execution with dependency management
- Automatic restart capability with progress detection
- SLURM integration for high-performance computing

The workflow stages are:
1. generate: Generate crystal structures using Genarris
2. process_generated: Process and deduplicate raw structures
3. relax: ML-based structure relaxation using UMA
4. filter: Energy filtering and final deduplication
5. evaluate: Compare against experimental data (optional)
6. create_vasp_inputs_*: Generate DFT input files (optional)
7. submit_vasp: Submit DFT calculations (optional)
8. read_vasp_outputs: Process DFT results (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fairchem.applications.fastcsp.core.utils import logging
from fairchem.applications.fastcsp.core.utils.configuration import (
    reorder_stages_by_dependencies,
    validate_config,
)
from fairchem.applications.fastcsp.core.utils.slurm import wait_for_jobs

if TYPE_CHECKING:
    import argparse


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Load and validate FastCSP workflow configuration from YAML file.

    This function reads the configuration file specified in the command line arguments,
    loads it as a YAML document, and validates it against the required stages.

    Args:
        args: Command line arguments containing the config file path and stages

    Returns:
        dict: Validated configuration dictionary containing all workflow parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is not valid YAML
        ValueError: If the configuration is missing required parameters for requested stages
    """
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    validate_config(config, args.stages)
    return config


def detect_restart(root_dir: Path, log_file: str = "FastCSP.log") -> bool:
    """
    Detect if this is a workflow restart by checking for existing log file.

    This function determines whether the current execution is a fresh start or
    a restart of a previous workflow by checking if the log file already exists
    and contains data.

    Args:
        root_dir: Root directory where the log file would be located
        log_file: Name of the log file to check (default: "FastCSP.log")

    Returns:
        bool: True if this appears to be a restart (log file exists with content),
              False for a fresh start
    """
    log_path = root_dir / log_file
    return log_path.exists() and log_path.stat().st_size > 0


def main(args: argparse.Namespace) -> None:
    """
    Main orchestration function for the FastCSP crystal structure prediction workflow.

    This function coordinates the execution of all FastCSP workflow stages in the correct
    dependency order. It handles:
    - Configuration loading and validation
    - Workspace setup and logging initialization
    - Restart detection and state management
    - Stage-by-stage execution with job dependency handling
    - Progress monitoring and error reporting

    The workflow supports both complete pipeline execution and selective stage execution,
    with automatic dependency resolution and restart capability.

    Workflow Stages Executed:
    1. generate: Crystal structure generation using Genarris
    2. process_generated: Structure processing and initial deduplication
    3. relax: ML-based structure relaxation using Universal Model for Atoms
    4. filter: Energy filtering and final structure deduplication
    5. evaluate: Experimental structure comparison (optional)
    6. create_vasp_inputs_*: DFT input file generation (optional)
    7. submit_vasp: DFT job submission (optional, requires customization)
    8. read_vasp_outputs: DFT result processing and analysis (optional)

    Args:
        args: Command line arguments containing:
            - config: Path to YAML configuration file
            - stages: List of workflow stages to execute

    Raises:
        FileNotFoundError: If required input files are missing
        ValueError: If configuration validation fails
        RuntimeError: If any workflow stage fails to complete successfully

    Side Effects:
        - Creates workspace directory structure
        - Generates log files and progress tracking
        - Submits SLURM jobs for parallel processing
        - Creates intermediate and final result files
    """
    # Load configuration and set up workspace
    config = load_config(args)
    root = Path(config["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Reorder stages based on dependencies
    args.stages = reorder_stages_by_dependencies(args.stages)

    # Set up logging to FastCSP.log in root directory
    log_file = root / "FastCSP.log"
    is_restart = detect_restart(root)
    logging.setup_fastcsp_logger(log_file=log_file, append=True)
    logging.ensure_all_modules_use_central_logger()
    logger = logging.get_central_logger()

    if is_restart:
        logger.info("=" * 80)
        logger.info(f"🔄 FASTCSP RESTART DETECTED - {log_file}")
        logger.info(f"📋 Executing stages: {', '.join(args.stages)}")
        logger.info("=" * 80)
        logging.print_fastcsp_header(logger, is_restart=True, stages=args.stages)
    else:
        logging.print_fastcsp_header(logger, is_restart=False, stages=args.stages)
        logger.info("Starting FastCSP workflow...")

    logger.info(f"Stages requested: {args.stages}")
    logger.info(f"Stages to execute (final order): {args.stages}")
    logger.info("Configuration loaded successfully")
    logger.info(f"Workspace directory: {root}")
    logging.log_config_pretty(logger, config)

    # Execute workflow stages
    # 1. Generate putative structures using Genarris
    if "generate" in args.stages:
        logging.log_stage_start(logger, "Genarris generation")
        from fairchem.applications.fastcsp.core.workflow.generate import (
            get_genarris_config,
            run_genarris_jobs,
        )

        genarris_config = get_genarris_config(config)
        jobs = run_genarris_jobs(
            output_dir=root / "genarris",
            genarris_config=genarris_config,
            molecules_file=config["molecules"],  # validate this
        )
        wait_for_jobs(jobs)
        logging.log_stage_complete(logger, "Genarris generation", len(jobs))

    # 2. Read Genarris outputs, deduplicate, and create Parquet files
    if "process_generated" in args.stages:
        logging.log_stage_start(logger, "deduplication of Genarris structures")
        from fairchem.applications.fastcsp.core.workflow.process_generated import (
            get_pre_relax_filter_config,
            process_genarris_outputs,
        )

        pre_relax_config = get_pre_relax_filter_config(config)
        jobs = process_genarris_outputs(
            input_dir=root / "genarris",
            output_dir=root / "raw_structures",
            pre_relax_config=pre_relax_config,
            ltol=pre_relax_config["ltol"],
            stol=pre_relax_config["stol"],
            angle_tol=pre_relax_config["angle_tol"],
            npartitions=pre_relax_config["npartitions"],
        )
        wait_for_jobs(jobs)
        logging.log_stage_complete(
            logger, "deduplicating structures from Genarris", len(jobs)
        )

    # 3. Relax structures using UMA MLIP
    if "relax" in args.stages:
        logging.log_stage_start(logger, "ML-relaxation of deduplicated structures")
        from fairchem.applications.fastcsp.core.workflow.relax import (
            get_relax_config_and_dir,
            run_relax_jobs,
        )

        relax_config, relax_output_dir = get_relax_config_and_dir(config)
        jobs = run_relax_jobs(
            input_dir=root / "raw_structures",
            output_dir=relax_output_dir / "raw_structures",
            relax_config=relax_config,
        )
        wait_for_jobs(jobs)
        logging.log_stage_complete(logger, "relaxing structures", len(jobs))

    # 4. Filter, deduplicate, and rank structures
    if "filter" in args.stages:
        logging.log_stage_start(
            logger, "filtering and deduplication of ML-relaxed structures"
        )
        from fairchem.applications.fastcsp.core.workflow.filter import (
            filter_and_deduplicate_structures,
            get_post_relax_config,
        )
        from fairchem.applications.fastcsp.core.workflow.relax import (
            get_relax_config_and_dir,
        )

        relax_config, relax_output_dir = get_relax_config_and_dir(config)
        post_relax_config = get_post_relax_config(config)
        jobs = filter_and_deduplicate_structures(
            input_dir=relax_output_dir / "raw_structures",
            output_dir=relax_output_dir / "filtered_structures",
            post_relax_config=post_relax_config,
            energy_cutoff=post_relax_config["energy_cutoff"],  # kJ/mol
            density_cutoff=post_relax_config["density_cutoff"],  # g/cm³
            ltol=post_relax_config["ltol"],
            stol=post_relax_config["stol"],
            angle_tol=post_relax_config["angle_tol"],
            root_unrelaxed=root / "raw_structures",
        )
        wait_for_jobs(jobs)
        logging.log_stage_complete(
            logger, "filtering and deduplicating ML-relaxed structures", len(jobs)
        )

    # 5. (Optional) Compare predicted structures to experimental
    # using either CSD API or pymatgen StructureMatcher
    # TODO: Implementation in progress - will be available soon
    if "evaluate" in args.stages:
        logging.log_stage_start(
            logger, "evaluating for structure matches to experimental structures"
        )
        from fairchem.applications.fastcsp.core.workflow.eval import (
            compute_structure_matches,
            get_eval_config,
        )

        eval_config = get_eval_config(config)
        compute_structure_matches(
            input_dir=relax_output_dir / "filtered_structures",
            output_dir=relax_output_dir / "matched_structures",
            config=eval_config,
            molecules_file=config["molecules"],
        )
        logging.log_stage_complete(logger, "evaluation against experimental structures")

    logger.info("🎉 FastCSP workflow completed successfully!")
    logger.info("=" * 80)

    # 6. (Optional) Calculate free energies for structures
    # TODO: Implementation in progress - will be available soon
    if "free_energy" in args.stages:
        logger.info("Free energy calculations requested...")
        # calculate_free_energies(
        #     relax_output_dir / "matched_structures",
        #     relax_output_dir / "free_energy_results",
        #     config,
        # )
        logger.info("Free energy calculations functionality coming soon...")
        logger.info(
            "Please check future releases or contact the developers for updates."
        )

    # # Optional VASP validation stages
    vasp_stages = [
        "create_vasp_inputs_relaxed",
        "create_vasp_inputs_unrelaxed",
        "submit_vasp",
        "read_vasp_outputs",
    ]

    # TODO: Implementation in progress - will be available soon
    if any(stage in args.stages for stage in vasp_stages):
        try:
            from fairchem.applications.fastcsp.core.dft.vasp import (
                collate_vasp_outputs,
                create_vasp_inputs_relaxed,
                create_vasp_inputs_unrelaxed,
            )
        except ImportError:
            logger.warning("atomate2 not installed. VASP functionality disabled.")
            return
        # 7. Generates VASP inputs for ML-relaxed structures
        if "create_vasp_inputs_relaxed" in args.stages:
            logger.info("Creating VASP inputs for ML-relaxed structures.")
            create_vasp_inputs_relaxed(
                config=config,
                root=root,
            )
            logger.info("Finished creating VASP inputs for ML-relaxed structures.")

        # 7'. Generate VASP inputs for unrelaxed structures
        if "create_vasp_inputs_unrelaxed" in args.stages:
            logging.log_stage_start(
                logger, "creating VASP inputs for unrelaxed structures"
            )
            create_vasp_inputs_unrelaxed(
                relaxed_dir=relax_output_dir / "matched_structures",
                output_dir=relax_output_dir / "vasp_inputs_unrelaxed",
                unrelaxed_dir=root / "raw_structures",
                unrelaxed=True,
                sym_prec=config["vasp"]["sym_prec"],
            )
            logging.log_stage_complete(
                logger, "creating VASP inputs for unrelaxed structures"
            )

        # 8. Submit VASP jobs
        # Users should implement their own function for their specific setup
        if "submit_vasp" in args.stages:
            logging.log_stage_start(logger, "VASP job submission")
            logger.info("VASP job submission requested...")
            logger.info("Please implement your own VASP job submission function.")
            logger.info(
                "This stage is intentionally left for users to customize based on their cluster environment."
            )
            logger.info(
                "You can use the VASP input files created in the 'vasp_inputs' directory."
            )
            logger.info(
                "Example: modify submit_vasp_jobs() in fairchem.applications.fastcsp.core.dft.vasp"
            )
            logger.info("or create your own submission script for your job scheduler.")

            # Uncomment and modify this line to use your VASP submission function:
            # submit_vasp_jobs(relax_output_dir / "vasp_inputs", relax_output_dir / "vasp_jobs.txt")
            logging.log_stage_complete(logger, "VASP job submission setup")

        # 9. Read VASP outputs and compute matches against experimental data
        if "read_vasp_outputs" in args.stages:
            logging.log_stage_start(
                logger, "reading VASP outputs and computing structure matches"
            )
            # read VASP outputs
            jobs = collate_vasp_outputs(
                relax_output_dir / "matched_structures",
                relax_output_dir / "vasp_inputs",
                relax_output_dir / "vasp_structures",
            )
            wait_for_jobs(jobs)
            logger.info(f"Collated VASP outputs from {len(jobs)} jobs")

            # Compute matches for DFT-relaxed structures
            compute_structure_matches(
                relax_output_dir / "vasp_structures",
                relax_output_dir / "vasp_matched_structures",
                config["molecules"],
                config,
            )
            logging.log_stage_complete(
                logger, "reading VASP outputs and computing structure matches"
            )
