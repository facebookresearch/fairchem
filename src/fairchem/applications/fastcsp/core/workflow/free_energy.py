"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_slurm_config,
    submit_slurm_jobs,
)
from fairchem.applications.fastcsp.core.utils.structure import cif_to_atoms
from fairchem.applications.fastcsp.core.workflow.relax import create_calculator
from tqdm import tqdm

from fairchem.core.components.calculate.recipes.phonons import (
    calculate_vibrational_thermo,
)

if TYPE_CHECKING:
    from pathlib import Path


def get_free_energy_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract free energy configuration from the workflow config.

    Args:
        config: Full workflow configuration dictionary

    Returns:
        Free energy parameters dictionary
    """
    fe_config = config.get("free_energy", {})
    return {
        "calculator": fe_config.get("calculator", "uma_sm_1p1_omc"),
        "quasiharmonic": fe_config.get("quasiharmonic", True),
        "atom_disp": fe_config.get("atom_disp", 0.01),
        "min_lengths": fe_config.get("min_lengths", 15.0),
        "t_step": fe_config.get("t_step", 10),
        "t_max": fe_config.get("t_max", 500),
        "t_min": fe_config.get("t_min", 0),
    }


def get_free_energy_slurm_config(
    fe_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get SLURM configuration for free energy jobs.

    Args:
        fe_config: Free energy configuration dictionary.
            If None, returns default parameters.

    Returns:
        SLURM parameters for submit_slurm_jobs function
    """
    if fe_config is None:
        fe_config = {}

    full_config = {"free_energy": fe_config}
    return get_slurm_config(full_config, "free_energy", "submit_slurm_jobs")


def compute_free_energy_single(
    input_file: Path,
    output_file: Path,
    fe_config: dict[str, Any],
) -> None:
    """
    Compute vibrational free energies for structures in a single parquet file.

    Args:
        input_file: Path to input parquet file with relaxed structures
        output_file: Path to output parquet file
        fe_config: Free energy configuration parameters
    """
    logger = get_central_logger()

    if output_file.exists():
        logger.info(f"Skipping {input_file}, output already exists: {output_file}")
        return

    logger.info(f"Computing free energies for {input_file}")
    structures_df = pd.read_parquet(input_file, engine="pyarrow")

    calc = create_calculator(fe_config)

    free_energy_results = []
    for idx, row in tqdm(structures_df.iterrows(), total=len(structures_df)):
        atoms = cif_to_atoms(row["relaxed_cif"])
        if atoms is None:
            free_energy_results.append({})
            logger.warning(f"Skipping structure {idx}: could not parse relaxed_cif")
            continue

        atoms.calc = calc
        try:
            thermo = calculate_vibrational_thermo(
                atoms,
                quasiharmonic=fe_config.get("quasiharmonic", False),
                atom_disp=fe_config.get("atom_disp", 0.01),
                min_lengths=fe_config.get("min_lengths", 15.0),
                t_step=fe_config.get("t_step", 10),
                t_max=fe_config.get("t_max", 500),
                t_min=fe_config.get("t_min", 0),
            )
            free_energy_results.append(thermo)
        except Exception:
            logger.exception(f"Free energy calculation failed for structure {idx}")
            free_energy_results.append({})

    all_keys = {k for r in free_energy_results for k in r}
    for key in sorted(all_keys):
        structures_df[key] = [r.get(key) for r in free_energy_results]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    structures_df.to_parquet(output_file, engine="pyarrow", compression="zstd")
    logger.info(
        f"Wrote free energy results for {len(structures_df)} structures to {output_file}"
    )


def compute_free_energies(
    input_dir: Path,
    output_dir: Path,
    fe_config: dict[str, Any],
) -> list:
    """
    Submit free energy computation jobs for all parquet files in input_dir.

    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory for output parquet files
        fe_config: Free energy configuration parameters

    Returns:
        List of submitted SLURM jobs
    """
    logger = get_central_logger()
    slurm_params = get_free_energy_slurm_config(fe_config)

    job_args = []
    for parquet_file in sorted(input_dir.iterdir()):
        if not parquet_file.name.endswith(".parquet"):
            continue
        output_file = output_dir / parquet_file.name
        if output_file.exists():
            logger.info(f"Skipping {parquet_file.name}, already processed")
            continue
        job_args.append(
            (
                compute_free_energy_single,
                (parquet_file, output_file, fe_config),
                {},
            )
        )

    logger.info(f"Submitting {len(job_args)} free energy jobs")
    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )
