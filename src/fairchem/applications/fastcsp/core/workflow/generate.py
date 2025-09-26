"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Generation Module using Genarris

This module provides functionality for generating putative crystal structures using
the Genarris 3.0 software package. It handles the complete workflow from input
preparation to SLURM job submission for parallel structure generation.

Key Features:
- Automated Genarris input file generation from molecular conformers
- SLURM integration for parallel processing across compute clusters
- Support for multiple space groups and Z-values
- Flexible molecular input handling (SMILES, XYZ, SDF formats)

Genarris Integration:
- Creates ui.conf configuration files for each molecule/Z combination
- Generates SLURM submission scripts with proper resource allocation

Dependencies:
- Genarris 3.0 software package (external)
"""

from __future__ import annotations

import ast
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pandas as pd
import submitit
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import get_genarris_slurm_config
from tqdm import tqdm


def get_genarris_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and validate Genarris configuration from the main workflow configuration.
    """
    gnrs_config = config.get("genarris", {})
    if gnrs_config == {}:
        raise KeyError("Genarris configuration section is missing in the config file.")
    return gnrs_config


def create_gnrs_submit_script(
    gnrs_config: dict[str, Any],
    genarris_slurm_config: dict[str, Any],
    single_gnrs_folder: str | Path,
):
    """
    Create a SLURM submission script for executing Genarris crystal structure generation.

    This function generates a complete SLURM batch script that includes resource allocation,
    environment setup, and the Genarris execution command with proper MPI parallelization.

    Args:
        gnrs_config: Genarris configuration containing execution parameters:
            - mpi_launcher: MPI command to use (default: "mpirun")
            - python_cmd: Python executable path (default: "python")
            - genarris_script: Genarris main script name (default: "genarris_master.py")
        genarris_slurm_config: SLURM resource allocation parameters
        single_gnrs_folder: Directory where the SLURM script will be created

    Side Effects:
        Creates 'slurm.sh' file in the specified directory with executable permissions
    """

    slurm_script = "#!/bin/sh\n"
    for key, value in genarris_slurm_config.items():
        slurm_script += f"#SBATCH --{key}={value}\n"

    slurm_script += f"""#SBATCH --output={single_gnrs_folder}/slurm.out
#SBATCH --error={single_gnrs_folder}/slurm.err

ulimit -s unlimited
export OMP_NUM_THREADS=1

{gnrs_config.get("mpi_launcher", "mpirun")} -np {genarris_slurm_config.get("nodes", 1) * genarris_slurm_config["ntasks-per-node"]} \\
    {gnrs_config.get("python_cmd", "python")} {gnrs_config.get("genarris_script", "genarris_master.py")} {single_gnrs_folder}/ui.conf > {single_gnrs_folder}/Genarris.out
"""

    with open(single_gnrs_folder / "slurm.sh", "w") as f:
        f.write(slurm_script)


def create_gnrs_config(
    gnrs_base_config: str | Path,
    output_dir: str | Path,
    mol_name: str,
    geometry_path: str | Path,
    num_structures: int,
    spg_info: str,
    Z: int,
):
    """
    Create a Genarris configuration file (ui.conf) for specific molecule and crystal parameters.

    This function customizes a base Genarris configuration template for a specific molecule,
    space group, and Z-value combination. It handles the parameter substitution needed
    for systematic crystal structure generation.

    Args:
        gnrs_base_config: Path to the base Genarris configuration template file
        output_dir: Directory where the customized ui.conf will be created
        mol_name: Identifier for the molecule being processed
        geometry_path: Path to the molecular geometry file (XYZ, SDF, etc.)
        num_structures: Number of crystal structures to generate
        spg_info: Space group specification (number or symbol)
        Z: Number of molecules per unit cell (Z-value)

    Side Effects:
        Creates 'ui.conf' file in the output directory with molecule-specific parameters
    """
    config = ConfigParser()
    with open(gnrs_base_config) as config_file:
        config.read_file(config_file)

    config["master"]["name"] = mol_name
    config["master"]["molecule_path"] = str(geometry_path)
    config["master"]["Z"] = str(Z)
    config["generation"]["num_structures_per_spg"] = str(num_structures)
    config["generation"]["spg_distribution_type"] = spg_info

    with open(output_dir / "ui.conf", "w") as f:
        config.write(f)


def create_genarris_jobs(
    mol_info: dict[str, Any],
    gnrs_config: dict[str, Any],
    output_dir: str | Path,
    genarris_slurm_config: dict[str, Any],
    executor: submitit.AutoExecutor,
):
    """Create Genarris structure generation jobs for molecules with different Z values."""
    logger = get_central_logger()
    logger.info(f"Starting Genarris generation for {mol_info['name']}")

    gnrs_base_config = gnrs_config.get("base_config")
    if gnrs_base_config is None:
        raise KeyError("Genarris 'base_config' section is missing in the config file.")
    logger.info(f"Using Genarris base configuration: {gnrs_base_config}")

    # Parameters for each Genarris run
    gnrs_vars = gnrs_config.get("vars", {})
    if gnrs_vars == {}:
        logger.info(
            "Using default Genarris parameters: Z=1, 500 structures per all compatible space group"
        )
    else:
        logger.info(f"Genarris generation parameters: {gnrs_vars}")

    z_list = [str(z) for z in gnrs_vars.get("Z", [1])]
    num_structures_per_spg = gnrs_vars.get("num_structures_per_spg", 500)
    spg_info = gnrs_vars.get("spg_info", "standard")

    # molecule specific spg and z_list info from csv file if provided
    if gnrs_vars.get("read_spg_from_file", False):
        spg_info = str(ast.literal_eval(mol_info["spg"]))
    if gnrs_vars.get("read_z_from_file", False):
        z_list = [str(z) for z in ast.literal_eval(mol_info["z"])]

    mol = mol_info["name"]  # System name

    # conf_path can be a geometry file or
    # a path to a folder containing multiple
    # conformers in .xyz, .extxyz, or .mol formats
    allowed_extensions = [".xyz", ".extxyz", ".mol"]

    conf_path = Path(mol_info["molecule_path"])
    if conf_path.is_file():
        if Path(conf_path).suffix not in allowed_extensions:
            raise TypeError(
                f"Molecule geometry file {conf_path} for {mol} has incompatible extension."
            )
        conf_name_list = {conf_path.stem: conf_path}
    elif conf_path.is_dir():
        conf_name_list = {
            c.stem: c
            for c in conf_path.rglob("*")
            if c.is_file() and Path(c).suffix in allowed_extensions
        }
    else:
        raise ValueError(f"Wrong conformer path {conf_path} for {mol} is provided.")
    if len(conf_name_list) == 0:
        raise ValueError(f"No valid conformer for {mol} was found.")

    jobs = []
    for conf, conf_path in conf_name_list.items():  # for each conformer
        for z in z_list:
            single_gnrs_folder = output_dir / mol / conf / z
            single_gnrs_folder.mkdir(parents=True, exist_ok=True)

            # copy conformer geometry file to new folder
            shutil.copy(conf_path, single_gnrs_folder.parent)
            new_conf_path = single_gnrs_folder.parent / conf_path.name

            # Create Genarris config if it doesn't exist
            if not (single_gnrs_folder / "ui.conf").exists():
                create_gnrs_config(
                    gnrs_base_config=gnrs_base_config,
                    output_dir=single_gnrs_folder,
                    mol_name=mol,
                    geometry_path=new_conf_path,
                    Z=z,
                    num_structures=num_structures_per_spg,
                    spg_info=spg_info,
                )

            # Create SLURM submission script if it doesn't exist
            if not (single_gnrs_folder / "slurm.sh").exists():
                create_gnrs_submit_script(
                    gnrs_config=gnrs_config,
                    genarris_slurm_config=genarris_slurm_config,
                    single_gnrs_folder=single_gnrs_folder,
                )

            # Create submitit command function to execute the SLURM script
            shell = shutil.which("bash") or shutil.which("zsh") or "/bin/sh"
            gnrs_function = submitit.helpers.CommandFunction(
                # f"zsh {single_gnrs_folder / 'slurm.sh'}".split(),
                [shell, f"{single_gnrs_folder}/slurm.sh"],
                cwd=single_gnrs_folder,
            )

            # Submit job to SLURM and add to job list
            job = executor.submit(
                gnrs_function,
                single_gnrs_folder,
            )
            jobs.append(job)
    return jobs


def run_genarris_jobs(
    output_dir: str | Path,
    genarris_config: dict[str, Any],
    molecules_file: str | Path,
):
    """Execute Genarris crystal structure generation workflow."""
    logger = get_central_logger()

    # Set up base output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure SLURM parameters
    genarris_slurm_config, executor_params = get_genarris_slurm_config(genarris_config)

    # Configure SLURM executor
    executor = submitit.AutoExecutor(folder=output_dir.parent / "slurm")
    executor.update_parameters(**executor_params)

    molecules_list = pd.read_csv(molecules_file).to_dict(orient="records")

    # Create Genarris jobs for each molecule
    jobs = []
    with executor.batch():
        for mol_info in tqdm(molecules_list, desc="Creating Genarris jobs"):
            jobs += create_genarris_jobs(
                mol_info,
                genarris_config,
                output_dir,
                genarris_slurm_config,
                executor,
            )

    logger.info(
        f"Submitted {len(jobs)} Genarris jobs: {jobs[0].job_id.split('_')[0] if jobs else ''}"
    )
    return jobs


if __name__ == "__main__":
    """Example usage for Genarris crystal structure generation."""
    import yaml

    # Define path for a specific Genarris run configuration
    config_path = Path("configs/example_config.yaml")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    jobs = run_genarris_jobs(
        output_dir=Path(config["root"]).resolve() / "genarris",
        genarris_config=config["genarris"],
        molecules_file=config["molecules_file"],
    )

    logger = get_central_logger()
    logger.info(f"Started {len(jobs)} Genarris jobs")
    logger.info("Monitor progress with SLURM commands or job.wait()")
