"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_slurm_config,
    submit_slurm_jobs,
)
from fairchem.applications.fastcsp.core.utils.structure import cif_to_atoms
from fairchem.applications.fastcsp.core.workflow.relax import create_calculator

from fairchem.core.components.calculate.recipes.phonons import (
    calculate_vibrational_thermo,
)


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
    row_index: int,
    fe_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute vibrational free energy for a single polymorph.

    Args:
        input_file: Path to input parquet file with relaxed structures
        row_index: Integer index of the row to process
        fe_config: Free energy configuration parameters

    Returns:
        Dictionary with free energy results, or empty dict on failure.
        Always includes 'source_file' and 'row_index' keys for reassembly.
    """
    logger = get_central_logger()

    structures_df = pd.read_parquet(input_file, engine="pyarrow")
    row = structures_df.iloc[row_index]

    result = {"source_file": str(input_file), "row_index": row_index}

    atoms = cif_to_atoms(row["relaxed_cif"])
    if atoms is None:
        logger.warning(
            f"Skipping structure at index {row_index} in {input_file}: "
            "could not parse relaxed_cif"
        )
        return result

    calc = create_calculator(fe_config)
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
        result.update(thermo)
    except Exception:
        logger.exception(
            f"Free energy calculation failed for index {row_index} in {input_file}"
        )

    return result


def collect_free_energy_results(
    jobs: list,
    output_dir: Path,
) -> None:
    """
    Collect results from per-polymorph free energy jobs and write output parquets.

    Groups results by their source file and merges free energy columns
    back into the original dataframes.

    Args:
        jobs: List of completed submitit Job objects
        output_dir: Directory for output parquet files
    """
    logger = get_central_logger()

    # Gather all results grouped by source file
    results_by_file: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        result = job.result()
        source = result.pop("source_file")
        results_by_file.setdefault(source, []).append(result)

    # Write one output parquet per input file
    for source_file_str, results in results_by_file.items():
        source_file = Path(source_file_str)
        structures_df = pd.read_parquet(source_file, engine="pyarrow")

        # Collect all thermo keys across results
        thermo_keys = {k for r in results for k in r if k != "row_index"}
        for key in sorted(thermo_keys):
            structures_df[key] = None

        # Fill in per-row results
        for r in results:
            idx = r.pop("row_index")
            for key, value in r.items():
                structures_df.loc[structures_df.index[idx], key] = value

        output_file = output_dir / source_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        structures_df.to_parquet(output_file, engine="pyarrow", compression="zstd")
        logger.info(
            f"Wrote free energy results for {len(structures_df)} structures "
            f"to {output_file}"
        )


def compute_free_energies(
    input_dir: Path,
    output_dir: Path,
    fe_config: dict[str, Any],
) -> list:
    """
    Submit one free energy job per polymorph across all parquet files.

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

        structures_df = pd.read_parquet(parquet_file, engine="pyarrow")
        job_args.extend(
            [
                (
                    compute_free_energy_single,
                    (parquet_file, row_index, fe_config),
                    {},
                )
                for row_index in range(len(structures_df))
            ]
        )

    logger.info(f"Submitting {len(job_args)} free energy jobs")
    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )
