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
        "remove_problematic": match_config.get(
            "remove_problematic", False
        ),  # default remove problematic structures
        "energy_cutoff": match_config.get(
            "energy_cutoff", None
        ),  # default 10000 kJ/mol
        "density_cutoff": match_config.get("density_cutoff", None),  # default 100 g/cm³
        "assign_groups": match_config.get(
            "assign_groups", False
        ),  # default no grouping
        "ltol": match_config.get("ltol", 0.2),  # default lattice tolerance
        "stol": match_config.get("stol", 0.3),  # default site tolerance
        "angle_tol": match_config.get(
            "angle_tol", 5
        ),  # default angle tolerance in degrees
        "remove_duplicates": match_config.get(
            "remove_duplicates", False
        ),  # default no deduplication
    }


def filter_and_deduplicate_structures_single(
    input_filename: Path,
    output_filename: Path,
    remove_problematic: bool = False,
    energy_cutoff: float | None = None,
    density_cutoff: float | None = None,
    assign_groups: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    remove_duplicates: bool = False,
    root_unrelaxed: Path | None = None,
):
    """
    Apply filtering to a single dataset.

    Args:
        input_filename: Path to input parquet file with structure data
        output_filename: Path to output parquet file for filtered results
        remove_problematic: Whether to remove problematic structures (non-converged or connectivity changed)
        energy_cutoff: Maximum energy above minimum (kJ/mol)
        density_cutoff: Maximum allowed density (g/cm³) for filtering
        assign_groups: Whether to assign group IDs to similar structures
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        remove_duplicates: Whether to enable structure deduplication
        root_unrelaxed: Path to unrelaxed structures for comparison

    Filtering Workflow:
        1. Validate connectivity preservation during relaxation
        2. Deal with problematic structures (non-converged or connectivity changed)
        3. Apply density cutoff to remove unphysical structures
        4. Energy-based filtering relative to global minimum
        5. Deduplication with pymatgen StructureMatcher
        6. Save filtered and deduplicated results
    """
    logger = get_central_logger()

    # Load structure dataset from parquet format
    structures_df = pd.read_parquet(input_filename, engine="pyarrow")

    # 1. Validate connectivity preservation during ML relaxation
    # This is done only if unrelaxed structures are provided
    # Most likely this was performed at the relaxation stage already
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
            input_filename.parent.with_suffix(".updated") / input_filename.name,
            engine="pyarrow",
            compression="zstd",
            partition_cols=["partition_id"],
        )
        logger.info(
            f"Saved updated dataframe to {input_filename.parent.with_suffix('.updated')}"
        )

    # 2. Separate problematic structures for retention
    problematic_structures_df = structures_df[
        ~structures_df["converged"] | ~structures_df["connectivity_unchanged"]
    ]
    structures_df_filtered = structures_df[
        structures_df["converged"] & structures_df["connectivity_unchanged"]
    ]

    # 3. Apply multi-stage filtering workflow
    if density_cutoff is not None:
        logger.info(f"Before filtering by density: {structures_df.shape}")
        structures_df = structures_df[
            structures_df["density"] <= density_cutoff
        ]  # Remove unphysically dense structures
        logger.info(f"After filtering by density: {structures_df.shape}")

    # Apply energy-based cutoff relative to global minimum
    if energy_cutoff is not None:
        logger.info(f"Before filtering by energy: {structures_df_filtered.shape}")
        min_energy = structures_df_filtered["energy_relaxed_per_molecule"].min()
        structures_df_filtered = structures_df_filtered[
            structures_df_filtered["energy_relaxed_per_molecule"]
            <= min_energy + energy_cutoff
        ]
        logger.info(f"After filtering by energy: {structures_df_filtered.shape}")

    # Convert CIF strings to pymatgen Structures for deduplication
    structures_df_filtered["structure"] = structures_df_filtered["relaxed_cif"].apply(
        cif_to_structure
    )

    # Apply deduplication without hash-based pre-filtering
    # (disable density/volume hashing for final deduplication)
    if assign_groups:
        structures_df_filtered = deduplicate_structures(
            structures_df_filtered,
            remove_duplicates=remove_duplicates,
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            hash_density=False,
            hash_volume=False,
        )
        problematic_structures_df[
            "group_index"
        ] = -1  # Mark problematic structures with group -1

    structures_df_filtered = structures_df_filtered.drop(columns=["structure"])

    if not remove_problematic:
        # Reintegrate problematic structures if not removing them
        structures_df_filtered = pd.concat(
            [structures_df_filtered, problematic_structures_df], ignore_index=True
        )

    # Save filtered and deduplicated results
    structures_df_filtered.to_parquet(
        output_filename,
        engine="pyarrow",
        compression="zstd",
    )


def filter_and_deduplicate_structures(
    input_dir: Path,
    output_dir: Path,
    post_relax_config: dict[str, Any],
    remove_problematic: bool,
    energy_cutoff: float,
    density_cutoff: float,
    assign_groups: bool,
    ltol: float,
    stol: float,
    angle_tol: float,
    remove_duplicates: bool = False,
    root_unrelaxed: Path | None = None,
):
    """
    Orchestrate parallel filtering and deduplication across multiple structure datasets.

    Args:
        input_dir: Root directory containing multiple dataset directories
        output_dir: Base directory for filtered output files
        post_relax_config: Configuration dictionary containing SLURM and filtering parameters
        remove_problematic: Whether to remove problematic structures (non-converged or connectivity changed)
        energy_cutoff: Energy threshold above minimum (kJ/mol)
        density_cutoff: Maximum density threshold (g/cm³)
        assign_groups: Whether to assign group indices during deduplication
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        remove_duplicates: Whether to enable deduplication
        root_unrelaxed: Root directory with unrelaxed structures

    Returns:
        List of submitit job objects for monitoring progress
    """
    logger = get_central_logger()

    # Get SLURM configuration
    slurm_params = get_filter_slurm_config(post_relax_config)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare job arguments
    job_args = []
    for molecule_parquet in list(input_dir.iterdir()):
        output_filename = output_dir / f"{molecule_parquet.name}.parquet"

        # Skip datasets that have already been processed
        if output_filename.exists():
            logger.info(
                f"Skipping {molecule_parquet} because {output_filename} already exists"
            )
            continue

        unrelaxed_path = (
            root_unrelaxed / molecule_parquet.name if root_unrelaxed else None
        )

        job_args.append(
            (
                filter_and_deduplicate_structures_single,
                (
                    molecule_parquet,
                    output_filename,
                    remove_problematic,
                    energy_cutoff,
                    density_cutoff,
                    assign_groups,
                    ltol,
                    stol,
                    angle_tol,
                    remove_duplicates,
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
