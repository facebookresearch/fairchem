"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure filtering utilities for FastCSP.

This module provides comprehensive filtering capabilities for crystal structures
based on energy, density, connectivity, and other criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ase.units import eV, kJ, mol
from fairchem.applications.fastcsp.core.utils.deduplicate import deduplicate_structures
from fairchem.applications.fastcsp.core.utils.slurm import submit_slurm_jobs
from fairchem.applications.fastcsp.core.utils.structure import (
    check_no_changes_in_covalent_matrix,
    check_no_changes_in_Z,
    cif_to_atoms,
    cif_to_structure,
)
from p_tqdm import p_map

if TYPE_CHECKING:
    from pathlib import Path

# Energy conversion constant for unit standardization
KJ_PER_MOL_TO_EV = eV / (kJ / mol)


def filter_and_deduplicate_structures_single(
    root: Path,
    output_path: Path,
    energy_cutoff_kj_per_mol: float = 20,
    root_unrelaxed: Path | None = None,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    density_cutoff: float = 2.5,
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
        energy_cutoff_kj_per_mol: Maximum energy above minimum (kJ/mol)
        root_unrelaxed: Path to unrelaxed structures for comparison
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        density_cutoff: Maximum allowed density (g/cm³) for filtering

    Filtering Workflow:
        1. Validate connectivity preservation during relaxation
        2. Apply density cutoff to remove unphysical structures
        3. Energy-based filtering relative to global minimum
        4. Deduplication with pymatgen StructureMatcher
        5. Save filtered and deduplicated results
    """
    # Load structure dataset from parquet format
    structures_df = pd.read_parquet(root, engine="pyarrow")

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
        final_atoms = structures_df["relaxed_cif"].swifter.apply(cif_to_atoms)
        initial_atoms = structures_df["cif"].swifter.apply(cif_to_atoms)

        # Validate bonding network preservation during relaxation
        structures_df["connectivity_unchanged"] = p_map(
            check_no_changes_in_covalent_matrix,
            initial_atoms,
            final_atoms,
            num_cpus=120,  # Parallel processing for connectivity validation
        )

        # Validate molecular unit count preservation (less strict check)
        structures_df["Z_unchanged"] = p_map(
            check_no_changes_in_Z, initial_atoms, final_atoms, num_cpus=120
        )

        # Save intermediate results with connectivity validation flags
        structures_df.to_parquet(
            root.parent.with_suffix(".updated") / root.name,
            engine="pyarrow",
            compression="zstd",
            partition_cols=["partition_id"],
        )
        print("Saved updated dataframe to ", root.parent.with_suffix(".updated"))

    # 2. Apply multi-stage filtering workflow
    print("Before filtering by density: ", structures_df.shape)
    structures_df = structures_df[
        structures_df["density"] < density_cutoff
    ]  # Remove unphysically dense structures
    print("After filtering by density: ", structures_df.shape)

    # Filter by convergence status and connectivity preservation
    # TODO: keep disordered structures that fail connectivity check
    structures_df_filtered = structures_df[
        structures_df["converged"] & structures_df["connectivity_unchanged"]
    ]

    # Apply energy-based cutoff relative to global minimum
    min_energy = structures_df_filtered["energy_relaxed_per_molecule"].min()
    energy_cutoff = energy_cutoff_kj_per_mol / KJ_PER_MOL_TO_EV  # Convert to eV
    structures_df_filtered = structures_df_filtered[
        structures_df_filtered["energy_relaxed_per_molecule"]
        < min_energy + energy_cutoff
    ]

    # Convert CIF strings to pymatgen Structures for deduplication
    structures_df_filtered["structure"] = structures_df_filtered[
        "relaxed_cif"
    ].swifter.apply(cif_to_structure)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    structures_df_deduped.to_parquet(
        output_path,
        engine="pyarrow",
        compression="zstd",
    )


def filter_and_deduplicate_structures(
    root: Path,
    output_path: Path,
    energy_cutoff_kj_per_mol: float,
    root_unrelaxed: Path | None,
    ltol: float,
    stol: float,
    angle_tol: float,
    density_cutoff: float,
):
    """
    Orchestrate parallel filtering and deduplication across multiple structure datasets.

    Args:
        root: Root directory containing multiple dataset directories
        output_path: Base directory for filtered output files
        energy_cutoff_kj_per_mol: Energy threshold above minimum (kJ/mol)
        root_unrelaxed: Root directory with unrelaxed structures
        ltol: Lattice parameter tolerance for structure matching
        stol: Site tolerance for structure matching
        angle_tol: Angle tolerance for structure matching
        density_cutoff: Maximum density threshold (g/cm³)

    Returns:
        List of submitit job objects for monitoring progress
    """
    # Collect all dataset directories for processing
    direcs = list(root.iterdir())

    # Prepare job arguments
    job_args = []
    for dir_path in direcs:
        output_file = output_path / f"{dir_path.name}.parquet"

        # Skip datasets that have already been processed
        if output_file.exists():
            print(f"Skipping {dir_path} because {output_file} already exists")
            continue

        unrelaxed_path = root_unrelaxed / dir_path.name if root_unrelaxed else None

        job_args.append(
            (
                filter_and_deduplicate_structures_single,
                (
                    dir_path,
                    output_file,
                    energy_cutoff_kj_per_mol,
                    unrelaxed_path,
                    ltol,
                    stol,
                    angle_tol,
                    density_cutoff,
                ),
                {},
            )
        )

    return submit_slurm_jobs(
        job_args,
        job_name="filter_and_deduplicate_structures",
        output_dir=root / "slurm",
        partition="ocp,learnaccel",
        cpus_per_task=80,
        mem_gb=400,
        timeout_min=1000,
    )
