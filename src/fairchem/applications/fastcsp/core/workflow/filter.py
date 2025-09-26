"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Post-Relaxation Structure Filtering, Deduplication, and Ranking Module

This module provides comprehensive functionality for processing ML-relaxed crystal structures
to generate a final ranked energy landscape. It implements filtering, deduplication,
and structure quality control measures.

Key Features:
- Energy and density-based filtering with configurable cutoffs
- Structure deduplication using pymatgen's StructureMatcher
- Control checks for structural integrity
- SLURM integration for scalable computation

Filtering Process:
1. Energy Filtering: Remove structures beyond energy cutoff from global minimum
2. Density Filtering: Filter structures with unrealistic densities
3. Structure Deduplication: Remove similar structures
4. Structure Quality Control: Validate chemical composition and bonding integrity
5. Ranking: Sort structures by energy to create energy landscape
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from fairchem.applications.fastcsp.core.utils.deduplicate import deduplicate_structures
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_filter_slurm_config,
    submit_slurm_jobs,
)
from fairchem.applications.fastcsp.core.utils.structure import (
    check_no_changes_in_covalent_matrix,
    cif_to_atoms,
    cif_to_structure,
)
from p_tqdm import p_map

if TYPE_CHECKING:
    from pathlib import Path


def get_post_relax_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and validate post-relaxation filtering parameters from workflow configuration.
    """
    match_config = config.get("post_relax_match_params", {})
    return {
        "energy_cutoff": match_config.get("energy-cutoff", 20.0),  # default 20 kJ/mol
        "density_cutoff": match_config.get("density-cutoff", 100),  # default 0.1 g/cm³
        "ltol": match_config.get("ltol", 0.2),  # default lattice tolerance
        "stol": match_config.get("stol", 0.3),  # default site tolerance
        "angle_tol": match_config.get(
            "angle_tol", 5
        ),  # default angle tolerance in degrees
    }


def filter_and_deduplicate_structures_single(
    input_dir: Path,
    output_dir: Path,
    energy_cutoff: float = 20,
    density_cutoff: float = 2.5,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    root_unrelaxed: Path | None = None,
):
    """
    Apply energy-based filtering and structure deduplication to a single dataset.

    Performs comprehensive filtering of crystal structures based on multiple criteria:
    - Energy cutoff relative to minimum energy structure
    - Density-based filtering to remove unphysical structures
    - Connectivity validation to ensure chemical bonds are preserved
    - Structure deduplication using pymatgen

    Args:
        root: Path to input parquet file with structure data
        output_path: Directory where filtered results will be saved
        energy_cutoff: Maximum energy above minimum (kJ/mol)
        density_cutoff: Maximum allowed density (g/cm³) for filtering
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        root_unrelaxed: Path to unrelaxed structures for comparison

    Filtering Workflow:
        1. Validate connectivity preservation during relaxation
        2. Apply density cutoff to remove unphysical structures
        3. Energy-based filtering relative to global minimum
        4. Deduplication with pymatgen StructureMatcher
        5. Save filtered and deduplicated results
    """
    logger = get_central_logger()

    # Load structure dataset from parquet format
    structures_df = pd.read_parquet(input_dir, engine="pyarrow")

    # 1. Validate connectivity preservation during ML relaxation
    if root_unrelaxed is not None:
        structures_df_unrelaxed = pd.read_parquet(
            root_unrelaxed, engine="pyarrow", columns=["structure_id", "cif"]
        )
        # Merge with unrelaxed data if requested for comparison studies
        structures_df = structures_df.merge(
            structures_df_unrelaxed,
            on="structure_id",
            how="left",
            suffixes=("", "_unrelaxed"),
        )

        # Convert CIF strings to atomic structures for connectivity analysis
        final_atoms = structures_df["relaxed_cif"].apply(cif_to_atoms)
        initial_atoms = structures_df["cif"].apply(cif_to_atoms)

        # Validate bonding network preservation during relaxation
        structures_df["connectivity_unchanged"] = p_map(
            check_no_changes_in_covalent_matrix,
            initial_atoms,
            final_atoms,
            num_cpus=120,  # Parallel processing for connectivity validation
        )

        # Save intermediate results with connectivity validation flags
        structures_df.to_parquet(
            input_dir.parent.with_suffix(".updated") / input_dir.name,
            engine="pyarrow",
            compression="zstd",
            partition_cols=["partition_id"],
        )
        logger.info(
            f"Saved updated dataframe to {input_dir.parent.with_suffix('.updated')}"
        )

    # 2. Apply multi-stage filtering workflow
    logger.info(f"Before filtering by density: {structures_df.shape}")
    structures_df = structures_df[
        structures_df["density"] < density_cutoff
    ]  # Remove unphysically dense structures
    logger.info(f"After filtering by density: {structures_df.shape}")

    # Filter by convergence status and connectivity preservation
    # TODO: keep disordered structures that fail connectivity check
    structures_df_filtered = structures_df[
        structures_df["converged"] & structures_df["connectivity_unchanged"]
    ]

    # Apply energy-based cutoff relative to global minimum
    min_energy = structures_df_filtered["energy_relaxed_per_molecule"].min()
    structures_df_filtered = structures_df_filtered[
        structures_df_filtered["energy_relaxed_per_molecule"]
        < min_energy + energy_cutoff
    ]

    # Convert CIF strings to pymatgen Structures for deduplication
    structures_df_filtered["structure"] = structures_df_filtered["relaxed_cif"].apply(
        cif_to_structure
    )

    # Apply deduplication without hash-based pre-filtering
    # (disable density/volume hashing for final deduplication)
    structures_df_deduped = deduplicate_structures(
        structures_df_filtered,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        hash_density=False,  # Disable for final deduplication
        hash_volume=False,
        remove_duplicates=False,  # Keep all structures with group assignments
    )

    # Clean up before saving - remove structure objects to reduce file size
    structures_df_deduped = structures_df_deduped.drop(columns=["structure"])

    # Save filtered and deduplicated results
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    structures_df_deduped.to_parquet(
        output_dir,
        engine="pyarrow",
        compression="zstd",
    )


def filter_and_deduplicate_structures(
    input_dir: Path,
    output_dir: Path,
    post_relax_config: dict[str, Any],
    energy_cutoff: float,
    density_cutoff: float,
    ltol: float,
    stol: float,
    angle_tol: float,
    root_unrelaxed: Path | None = None,
):
    """
    Orchestrate parallel filtering and deduplication across multiple structure datasets.

    Args:
        input_dir: Root directory containing multiple dataset directories
        output_dir: Base directory for filtered output files
        post_relax_config: Configuration dictionary containing SLURM and filtering parameters
        energy_cutoff: Energy threshold above minimum (kJ/mol)
        density_cutoff: Maximum density threshold (g/cm³)
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        root_unrelaxed: Root directory with unrelaxed structures

    Returns:
        List of submitit job objects for monitoring progress
    """
    logger = get_central_logger()

    # Get SLURM configuration
    slurm_params = get_filter_slurm_config(post_relax_config)

    # Collect all paruqet directories for processing
    direcs = list(input_dir.iterdir())

    # Prepare job arguments
    job_args = []
    for dir_path in direcs:
        output_file = output_dir / f"{dir_path.name}.parquet"

        # Skip datasets that have already been processed
        if output_file.exists():
            logger.info(f"Skipping {dir_path} because {output_file} already exists")
            continue

        unrelaxed_path = root_unrelaxed / dir_path.name if root_unrelaxed else None

        job_args.append(
            (
                filter_and_deduplicate_structures_single,
                (
                    dir_path,
                    output_file,
                    energy_cutoff,
                    density_cutoff,
                    ltol,
                    stol,
                    angle_tol,
                    unrelaxed_path,
                ),
                {},
            )
        )

    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )
