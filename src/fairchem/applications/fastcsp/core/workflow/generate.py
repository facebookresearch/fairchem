"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Genarris structure generation module.

This module provides functionality for setting up and running Genarris jobs
for crystal structure prediction. It handles SLURM job submission, config file
generation, and batch processing of multiple molecules.
"""

from __future__ import annotations

import ast
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pandas as pd
import submitit
from tqdm import tqdm


def create_gnrs_submit_script(
    gnrs_config: dict[str, Any],
    slurm_info: dict[str, Any],
    single_gnrs_folder: str | Path,
):
    """
    Create submit.sh script for SLURM job submission.

    Generates a bash script that sets up the environment and runs Genarris
    with the specified configuration. The script includes SLURM directives
    for job submission and proper environment setup.

    Args:
        gnrs_config: Dictionary containing Genarris configuration including
                    SLURM parameters, executable paths, and MPI settings
        slurm_info: Dictionary containing SLURM job parameters
        single_gnrs_folder: Directory to save the submit script
    """

    # Build SLURM header with job parameters
    slurm_script = "#!/bin/bash\n"
    for key, value in slurm_info.items():
        slurm_script += f"#SBATCH --{key}={value}\n"

    # Add output/error file redirection and environment setup
    slurm_script += f"""#SBATCH --output={single_gnrs_folder}/slurm.out
#SBATCH --error={single_gnrs_folder}/slurm.err

ulimit -s unlimited
export OMP_NUM_THREADS=1

{gnrs_config.get("mpi_launcher", "mpirun")} -np {slurm_info.get("nodes", 1) * slurm_info["ntasks-per-node"]} \\
    {gnrs_config.get("python_cmd", "python")} {gnrs_config.get("genarris_script", "genarris_master.py")} {single_gnrs_folder}/ui.conf > {single_gnrs_folder}/Genarris.out
"""

    with open(single_gnrs_folder / "slurm.sh", "w") as f:
        f.write(slurm_script)


def create_gnrs_config(
    gnrs_base_config: str | Path,
    output_dir: str | Path,
    mol_name: str,
    geometry_path: str | Path,
    num_structures: int = 500,
    spg_info: str | list[int] | None = "standard",
    Z: int = 1,
):
    """
    Create Genarris configuration file for a specific molecule and Z value.

    Reads a base Genarris config template and customizes it for a specific
    molecule by setting the molecule name, path, Z value, and structure
    generation parameters.

    Args:
        gnrs_base_config: Path to base Genarris config template
        output_dir: Directory to save the configuration file
        mol_name: Molecule identifier (e.g., CSD refcode)
        geometry_path: Path to conformer geometry file
        Z: Number of molecules per unit cell
        num_structures: Number of structures per space group to generate
        spg_info: Space group distribution ("standard" or list of space groups)
    """
    config = ConfigParser()
    with open(gnrs_base_config) as config_file:
        config.read_file(config_file)

    # Set molecule-specific parameters
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
    slurm_info: dict[str, Any],
    executor: submitit.AutoExecutor,
):
    """
    Create Genarris crystal structure generation jobs for multiple molecules and Z values.

    For each molecule and conformer combination, creates separate Genarris jobs across
    different Z values (molecules per unit cell). Each job generates crystal structures
    using the specified space group distributions and parameters.

    Args:
        mol_info: Dictionary mapping molecule names to conformer directory paths
        gnrs_config: Genarris configuration parameters including:
            - z_list: List of Z values to explore
            - base_config_path: Path to Genarris template config
            - Various generation parameters (space groups, structure counts)
        output_dir: Base directory for organizing generated crystal structures
        executor: Configured submitit executor for SLURM job submission

    Returns:
        list[submitit.Job]: List of submitted Genarris jobs for monitoring and results collection

    Structure:
        genarris/
        ├── molecule1/
        │   ├── conformer1/
        │   │   ├── Z1/ (Genarris job output)
        │   │   ├── Z2/
        │   │   └── Z4/
        │   └── conformer2/
        └── molecule2/
    """
    print(f"Starting Genarris generation for {mol_info['name']}")

    # Genarris base config file
    gnrs_base_config = gnrs_config.get("base_config")
    if gnrs_base_config is None:
        raise KeyError("Genarris 'base_config' section is missing in the config file.")
    print(f"Genarris base configuration defined in {gnrs_base_config}")

    # parameters for each Genarris run
    gnrs_vars = gnrs_config.get("vars", {})
    if gnrs_vars == {}:
        print(
            "Genarris generation parameters are not provided. Using default parameters:"
        )
        print("Z=1 and 500 structures per all compatible space groups")
    else:
        print(f"Genarris generation parameters are {gnrs_vars} ")
    z_list = [str(z) for z in gnrs_vars.get("Z", [1])]  # by default only Z=1 is used
    num_structures_per_spg = gnrs_vars.get(
        "num_structures_per_spg", 500
    )  # by default 500 structures are generated per spacegroup
    spg_info = gnrs_vars.get(
        "spg_info", "standard"
    )  # all compatible spacegroups by default

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
            print(conf_path)
            raise TypeError(
                f"Molecule geometry file for {mol} has incompatible extension."
            )
        conf_name_list = {conf_path.stem: conf_path}
    elif conf_path.is_dir():
        conf_name_list = {
            c.stem: c
            for c in conf_path.rglob("*")
            if c.is_file() and Path(c).suffix in allowed_extensions
        }
    else:
        raise ValueError(f"Wrong conformer path for {mol} is provided.")
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
                    mol=mol,
                    new_conf_path=new_conf_path,
                    z=z,
                    num_structures_per_spg=num_structures_per_spg,
                    spg_info=spg_info,
                )

            # Create SLURM submission script if it doesn't exist
            if not (single_gnrs_folder / "slurm.sh").exists():
                create_gnrs_submit_script(
                    gnrs_config=gnrs_config,
                    slurm_info=slurm_info,
                    output_dir=single_gnrs_folder,
                )

            # Create submitit command function to execute the SLURM script
            gnrs_function = submitit.helpers.CommandFunction(
                f"bash {single_gnrs_folder / 'slurm.sh'}".split(),
                cwd=single_gnrs_folder,
            )

            # Submit job to SLURM and add to job list
            job = executor.submit(
                gnrs_function,
                single_gnrs_folder,
            )
            jobs.append(job)
    return jobs


def run_genarris_jobs(output_dir: str | Path, config: dict[str, Any]):
    """
    Main function to execute Genarris crystal structure generation workflow.

    Orchestrates the complete Genarris pipeline: sets up output directories,
    configures SLURM executor, creates jobs for all molecule/conformer/Z combinations,
    and manages job submission and monitoring.

    Args:
        output_dir: Base output directory path
        config: Master configuration dictionary containing:
            - molecule_info: Dictionary of molecules and conformer paths
            - genarris: Genarris-specific configuration including:
                - executables to run Genarris (python, master genarris script, mpirun)
                - slurm: SLURM job parameters
                - generation parameters (Z values, space groups, structure counts)

    Workflow:
        1. Create base output directory structure
        2. Extract Genarris configuration and validate
        3. Set up SLURM executor with specified parameters
        4. Create and submit jobs for all molecule/conformer combinations
        5. Return job list for monitoring and result collection

    Returns:
        list[submitit.Job]: List of submitted jobs for progress tracking

    Raises:
        KeyError: If required configuration sections are missing
        ValueError: If molecule information or paths are invalid

    Note:
        This function handles the high-level orchestration while delegating
        specific tasks to create_genarris_jobs() and related utilities.
    """
    # Set up base output directory for all Genarris results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure submitit executor with SLURM parameters
    slurm_info = config["genarris"].get("slurm_info", {})
    if slurm_info == {}:
        slurm_info = {
            "job-name": "genarris",
            "nodes": 1,
            "ntasks-per-node": 1,
            "time": 7200,  # minutes
        }
        print("SLURM info is not provided for Genarris.")
        print("Using default parameters.")
    print(f"SLURM info for Genarris. {slurm_info}")

    executor = submitit.AutoExecutor(folder=output_dir.parent / "slurm")
    executor.update_parameters(
        slurm_job_name=slurm_info["job-name"],
        nodes=slurm_info["nodes"],
        tasks_per_node=slurm_info["ntasks-per-node"],
        timeout_min=slurm_info["time"],
        slurm_use_srun=False,
        cpus_per_task=1,
    )

    molecules_file = config["molecules"]
    molecules_list = pd.read_csv(molecules_file).to_dict(orient="records")

    # Create Genarris jobs for each conformer
    jobs = []
    with executor.batch():
        for mol_info in tqdm(molecules_list):
            jobs += create_genarris_jobs(
                mol_info,
                config["genarris"],
                output_dir,
                slurm_info,
                executor,
            )

    print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id}")
    return jobs


if __name__ == "__main__":
    """
    Example usage for Genarris crystal structure generation.
    """
    import yaml

    # Define path for a specific Genarris run configuration
    config_path = Path("configs/example_config.yaml")

    # Load configuration file
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    # Execute Genarris crystal structure generation
    jobs = run_genarris_jobs(root=Path(config["root"]).resolve(), config=config)

    print(f"Started {len(jobs)} Genarris jobs")
    print("Monitor job progress with SLURM commands or job.wait() for completion")
