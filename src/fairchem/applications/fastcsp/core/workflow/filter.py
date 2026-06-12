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

import os
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
    match_config = config.get("post_relaxation_filter", {})
    return {
        "remove_problematic": match_config.get(
            "remove_problematic", False
        ),  # default remove problematic structures
        "energy_cutoff": match_config.get(
            "energy_cutoff", None
        ),  # default 10000 kJ/mol
        "density_min_cutoff": match_config.get(
            "density_min_cutoff", None
        ),  # default no lower density bound (g/cm³)
        "density_max_cutoff": match_config.get(
            "density_max_cutoff", None
        ),  # default no upper density bound (g/cm³)
        "assign_groups": match_config.get(
            "assign_groups", False
        ),  # default no grouping
        "ltol": match_config.get("ltol", 0.2),  # default lattice tolerance
        "stol": match_config.get("stol", 0.3),  # default site tolerance
        "angle_tol": match_config.get(
            "angle_tol", 5
        ),  # default angle tolerance in degrees
        "density_bin_size": match_config.get(
            "density_bin_size", None
        ),  # default no density blocker (full mol_id+z bucket)
        "energy_bin_size": match_config.get(
            "energy_bin_size", None
        ),  # default no energy blocker (full mol_id+z bucket)
        "density_tol": match_config.get(
            "density_tol", None
        ),  # g/cc; cheap |Δρ| prefilter before sm.fit. None = off.
        "energy_tol": match_config.get(
            "energy_tol", None
        ),  # kJ/mol; cheap |ΔE| prefilter before sm.fit. None = off.
        "apply_niggli_filter": match_config.get(
            "apply_niggli_filter", False
        ),  # True = apply Niggli (a,b,c,alpha,beta,gamma) prefilter; False = skip it.
        "bin_by_conf": match_config.get(
            "bin_by_conf", False
        ),  # include conf_id in the dedup bin key
        "bin_by_z": match_config.get(
            "bin_by_z", False
        ),  # include Z in the dedup bin key
        "bin_by_spg": match_config.get(
            "bin_by_spg", False
        ),  # include spg_generated in the dedup bin key
        "remove_duplicates": match_config.get(
            "remove_duplicates", False
        ),  # default no deduplication
    }


def filter_and_deduplicate_structures_single(
    input_filename: Path,
    output_filename: Path,
    remove_problematic: bool = False,
    energy_cutoff: float | None = None,
    density_min_cutoff: float | None = None,
    density_max_cutoff: float | None = None,
    assign_groups: bool = True,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    bin_by_conf: bool = False,
    bin_by_z: bool = True,
    bin_by_spg: bool = False,
    density_bin_size: float | None = None,
    energy_bin_size: float | None = None,
    remove_duplicates: bool = False,
    root_unrelaxed: Path | None = None,
    density_tol: float | None = None,
    energy_tol: float | None = None,
    apply_niggli_filter: bool = False,
):
    """
    Apply filtering and deduplication to a single parquet dataset.

    Args:
        input_filename: Path to input parquet file with structure data
        output_filename: Path to output parquet file for filtered results
        remove_problematic: Whether to remove problematic structures (non-converged or connectivity changed)
        energy_cutoff: Maximum energy above minimum (kJ/mol)
        density_min_cutoff: Minimum allowed density (g/cm³); None = no lower bound
        density_max_cutoff: Maximum allowed density (g/cm³); None = no upper bound
        assign_groups: Whether to assign group IDs to similar structures
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        density_bin_size: Optional density blocker bin (g/cc) for hash grouping.
            Set to subdivide the (mol_id, Z) buckets and speed up clustering.
            None = no density-based bucketing (one giant bucket per (mol_id, Z)).
        energy_bin_size: Optional energy blocker bin (kJ/mol) for hash grouping.
            Set to subdivide the (mol_id, Z) buckets and speed up clustering.
            None = no energy-based bucketing.
        remove_duplicates: Whether to enable structure deduplication
        root_unrelaxed: Path to unrelaxed structures for comparison
        density_tol: Optional cheap |Δρ| prefilter (g/cc) applied before
            ``StructureMatcher.fit``. None disables the prefilter.
        energy_tol: Optional cheap |ΔE| prefilter (kJ/mol) applied
            before ``StructureMatcher.fit``. None disables the prefilter.
        apply_niggli_filter: When True, apply the Niggli (a,b,c,alpha,beta,gamma)
            prefilter before ``StructureMatcher.fit``. Default False
            (skip the Niggli reduction + check entirely).

    Filtering Workflow:
        1. Validate connectivity preservation during relaxation
        2. Deal with problematic structures (non-converged or connectivity changed)
        3. Apply density cutoffs to remove unphysical structures
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
            root_unrelaxed, engine="pyarrow", columns=["structure_id", "cif_generated"]
        )
        # Merge with unrelaxed data if requested for comparison studies
        structures_df = structures_df.merge(
            structures_df_unrelaxed,
            on="structure_id",
            how="left",
            suffixes=("", "_unrelaxed"),
        )

        # Convert CIF strings to atomic structures for connectivity analysis
        # (parallelized: cif parsing is CPU-bound and embarrassingly parallel)
        num_cpus = max(len(os.sched_getaffinity(0)), 1)
        final_atoms = p_map(
            cif_to_atoms,
            structures_df["cif_relaxed"].tolist(),
            num_cpus=num_cpus,
            desc="cif_relaxed -> atoms",
        )
        initial_atoms = p_map(
            cif_to_atoms,
            structures_df["cif_generated"].tolist(),
            num_cpus=num_cpus,
            desc="cif_generated -> atoms",
        )

        # Validate bonding network preservation during relaxation
        structures_df["validity.connectivity_unchanged"] = p_map(
            check_no_changes_in_covalent_matrix,
            initial_atoms,
            final_atoms,
            num_cpus=num_cpus,
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
    validity_cols = [
        col for col in structures_df.columns if col.startswith("validity.")
    ]
    all_valid = structures_df[validity_cols].all(axis=1)
    problematic_structures_df = structures_df[
        ~structures_df["optimizer_converged"] | ~all_valid
    ]
    structures_df_filtered = structures_df[
        structures_df["optimizer_converged"] & all_valid
    ]

    # 3. Apply multi-stage filtering workflow
    # Density window: drop rows outside [density_min_cutoff, density_max_cutoff]
    if density_min_cutoff is not None or density_max_cutoff is not None:
        logger.info(
            f"Before filtering by density ["
            f"min={density_min_cutoff}, max={density_max_cutoff}]: "
            f"{structures_df_filtered.shape}"
        )
        if density_min_cutoff is not None:
            structures_df_filtered = structures_df_filtered[
                structures_df_filtered["density_relaxed"] >= density_min_cutoff
            ]
            problematic_structures_df = problematic_structures_df[
                problematic_structures_df["density_relaxed"] >= density_min_cutoff
            ]
        if density_max_cutoff is not None:
            structures_df_filtered = structures_df_filtered[
                structures_df_filtered["density_relaxed"] <= density_max_cutoff
            ]
            problematic_structures_df = problematic_structures_df[
                problematic_structures_df["density_relaxed"] <= density_max_cutoff
            ]
        logger.info(f"After filtering by density: {structures_df_filtered.shape}")

    # Apply energy-based cutoff relative to global minimum
    if energy_cutoff is not None:
        logger.info(f"Before filtering by energy: {structures_df_filtered.shape}")
        min_energy = structures_df_filtered["energy_relaxed_per_molecule"].min()
        structures_df_filtered = structures_df_filtered[
            structures_df_filtered["energy_relaxed_per_molecule"]
            <= min_energy + energy_cutoff
        ]
        problematic_structures_df = problematic_structures_df[
            problematic_structures_df["energy_relaxed_per_molecule"]
            <= min_energy + energy_cutoff
        ]
        logger.info(f"After filtering by energy: {structures_df_filtered.shape}")

    # Convert CIF strings to pymatgen Structures for deduplication
    # (parallelized: cif parsing is CPU-bound and embarrassingly parallel)
    dedup_num_cpus = max(len(os.sched_getaffinity(0)), 1)
    structures_df_filtered["structure"] = p_map(
        cif_to_structure,
        structures_df_filtered["cif_relaxed"].tolist(),
        num_cpus=dedup_num_cpus,
        desc="cif_relaxed -> structure",
    )

    # Post-relax dedup blocker = (mol_id, Z) + optional binned density/energy.
    if assign_groups:
        # ``energy_col`` must be set whenever we want to use it either as a
        # blocker (``energy_bin_size``) or as a cheap |ΔE| prefilter
        # (``energy_tol``).
        energy_col_used = (
            "energy_relaxed_per_molecule"
            if (energy_bin_size is not None or energy_tol is not None)
            else None
        )
        structures_df_filtered = deduplicate_structures(
            structures_df_filtered,
            conf_col="conf_id" if bin_by_conf else None,
            z_col="z" if bin_by_z else None,
            spg_col="spg_generated" if bin_by_spg else None,
            density_col="density_relaxed" if density_bin_size is not None else None,
            density_bin_size=density_bin_size,
            energy_col=energy_col_used,
            energy_bin_size=energy_bin_size,
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            ignored_species=["H"],
            density_tol=density_tol,
            energy_tol=energy_tol,
            apply_niggli_filter=apply_niggli_filter,
            keep="min" if remove_duplicates else None,
            keep_col="energy_relaxed_per_molecule",
        )
        problematic_structures_df[
            "group_index"
        ] = -1  # Mark problematic structures with group -1

    structures_df_filtered = structures_df_filtered.drop(columns=["structure"])

    if not remove_problematic:
        # Reintegrate problematic structures if not removing them
        logger.info("Reintegrating problematic structures")
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
    energy_cutoff: float | None,
    density_min_cutoff: float | None,
    density_max_cutoff: float | None,
    assign_groups: bool,
    ltol: float,
    stol: float,
    angle_tol: float,
    bin_by_conf: bool = False,
    bin_by_z: bool = True,
    bin_by_spg: bool = False,
    density_bin_size: float | None = None,
    energy_bin_size: float | None = None,
    remove_duplicates: bool = False,
    root_unrelaxed: Path | None = None,
    density_tol: float | None = None,
    energy_tol: float | None = None,
    apply_niggli_filter: bool = False,
):
    """
    Submit parallel filtering jobs for multiple datasets.

    Args:
        input_dir: Root directory containing multiple dataset directories
        output_dir: Base directory for filtered output files
        post_relax_config: Configuration dictionary containing SLURM and filtering parameters
        remove_problematic: Whether to remove problematic structures (non-converged or connectivity changed)
        energy_cutoff: Energy threshold above minimum (kJ/mol)
        density_min_cutoff: Lower density bound (g/cm³); None = no lower bound
        density_max_cutoff: Upper density bound (g/cm³); None = no upper bound
        assign_groups: Whether to assign group indices during deduplication
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        remove_duplicates: Whether to enable deduplication
        root_unrelaxed: Root directory with unrelaxed structures
        density_tol: Optional cheap |Δρ| prefilter (g/cc). None = off.
        energy_tol: Optional cheap |ΔE| prefilter (kJ/mol). None = off.
        apply_niggli_filter: When True, apply Niggli (a,b,c,alpha,beta,gamma) prefilter
            before ``StructureMatcher.fit``. Default False (skip it).

    Returns:
        List of submitit job objects.
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
                    density_min_cutoff,
                    density_max_cutoff,
                    assign_groups,
                    ltol,
                    stol,
                    angle_tol,
                    bin_by_conf,
                    bin_by_z,
                    bin_by_spg,
                    density_bin_size,
                    energy_bin_size,
                    remove_duplicates,
                    unrelaxed_path,
                    density_tol,
                    energy_tol,
                    apply_niggli_filter,
                ),
                {},
            )
        )

    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )
