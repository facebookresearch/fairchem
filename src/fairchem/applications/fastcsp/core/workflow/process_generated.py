"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Genarris output processing utilities for FastCSP.

This module handles the processing and conversion of raw Genarris output files
into standardized parquet format suitable for downstream ML processing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
from ase.io.jsonio import decode
from fairchem.applications.fastcsp.core.utils.deduplicate import deduplicate_structures
from fairchem.applications.fastcsp.core.utils.slurm import submit_slurm_jobs
from fairchem.applications.fastcsp.core.utils.structure import get_partition_id
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


def structure_to_row(
    hash_id: str, struct_dict: dict, mol_id: str, z_val: int, npartitions: int = 1000
) -> dict:
    """
    Convert crystal structure data to standardized row format for DataFrame storage.

    Args:
        hash_id: Base hash identifier for the structure
        struct_dict: Structure data in ASE JSON format from Genarris
        mol_id: Molecule identifier from the original input
        z_val: Z value (number of formula units per unit cell)
        npartitions: Number of partitions for distributed processing

    Returns:
        Standardized structure data containing:
            - mol_id: Molecule identifier
            - z: Number of formula units per unit cell
            - structure_id: Unique structure identifier
            - formula: Reduced chemical formula
            - n_atoms: Total number of atoms in unit cell
            - volume: Unit cell volume in Ų
            - cif: Crystal structure in CIF format
            - partition_id: Partition assignment for parallel processing
            - structure: Pymatgen Structure object for analysis

    Processing Steps:
        1. Decode ASE JSON format to Atoms object
        2. Convert to pymatgen Structure for analysis
        3. Extract chemical composition and geometric properties
        4. Generate CIF string representation
        5. Assign consistent partition ID for distributed processing
    """
    # Decode JSON structure data to ASE Atoms object
    atoms = decode(json.dumps(struct_dict))

    structure = AseAtomsAdaptor.get_structure(atoms)

    # Extract chemical and structural properties
    formula = structure.composition.reduced_formula
    n_atoms = len(structure)
    volume = structure.volume
    cif_str = structure.to(fmt="cif")

    # Create unique structure identifier incorporating all parameters
    hash_id_ = f"{hash_id}_{mol_id}_{z_val}"

    return {
        "mol_id": mol_id,
        "z": z_val,
        "structure_id": hash_id_,
        "formula": formula,
        "n_atoms": n_atoms,
        "volume": volume,
        "cif": cif_str,
        "partition_id": get_partition_id(hash_id_, npartitions),
        "structure": structure,
    }


def process_genarris_outputs_single(
    base_dir: Path,
    output_dir: Path,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    npartitions: int = 1000,
):
    """
    Process Genarris output files from a single molecular conformer directory.

    Converts raw Genarris JSON structure files into standardized parquet format
    with structure deduplication and metadata extraction. This function handles
    the complex directory structure of Genarris outputs and transforms them into
    a format suitable for downstream ML processing.

    Args:
        base_dir: Root directory containing Genarris output structure
                 Expected structure: mol_id/conf_id/z_val/symm_rigid_press/structures.json
        output_dir: Directory where processed parquet files will be saved
        npartitions: Number of partitions for distributed processing (default: 1000)
        ltol: Lattice parameter tolerance for structure deduplication (default: 0.2)
        stol: Site tolerance for structure deduplication (default: 0.3)
        angle_tol: Angle tolerance for structure deduplication (default: 5°)

    Processing Workflow:
        1. Scan directory structure for structures.json files
        2. Extract mol_id and Z values from directory hierarchy
        3. Parse JSON structure data and convert to standardized format
        4. Apply deduplication using pymatgen
        5. Save results in partitioned parquet format for efficient access

    Output Format:
        Creates parquet files partitioned by partition_id containing:
        - structure_id: Unique identifier for each structure
        - mol_id: Original molecule identifier
        - z: Number of formula units per unit cell
        - formula: Reduced chemical formula
        - n_atoms: Total atoms in unit cell
        - volume: Unit cell volume
        - cif: Structure in CIF format
        - group_index: Deduplication group assignment
    """
    print(f"Processing {base_dir}")
    # Search for all structures.json files in the expected Genarris directory structure
    json_files = list(base_dir.glob("**/symm_rigid_press/structures.json"))
    print(f"Found {len(json_files)} files / {base_dir}")
    all_rows = []

    # Process each JSON file containing crystal structures
    for file_path in tqdm(json_files, desc="Processing files"):
        try:
            # Extract Z value and molecule ID from nested directory structure
            # Expected path: .../mol_id/conf_id/z_val/symm_rigid_press/structures.json
            z_val = int(file_path.parents[2].name)
            mol_id = file_path.parents[4].name
        except Exception as e:
            print(f"Failed to extract mol_id or z from path {file_path}: {e}")
            continue

        # Load structure data from JSON file
        with file_path.open("r") as f:
            struct_data = json.load(f)

        # Convert each structure to standardized row format
        for hash_id, struct_dict in tqdm(
            struct_data.items(),
            desc="Processing structures",
            total=len(struct_data),
        ):
            try:
                row = structure_to_row(hash_id, struct_dict, mol_id, z_val, npartitions)
                all_rows.append(row)
            except Exception as e:
                print(f"Failed to parse structure {hash_id} in {file_path}: {e}")

    # Create DataFrame and apply deduplication
    structures_df = pd.DataFrame(all_rows)
    structures_df = deduplicate_structures(structures_df, ltol, stol, angle_tol)

    # Remove structure objects before saving to reduce file size
    structures_df = structures_df.drop(columns=["structure"])

    # Save to partitioned parquet format for efficient distributed access
    structures_df.to_parquet(
        output_dir,
        compression="zstd",
        partition_cols=["partition_id"],
    )
    print(f"Saved {len(all_rows)} structures to {output_dir}")


def process_genarris_outputs(
    base_dir: Path,
    output_dir: Path,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    npartitions: int = 1000,
):
    """
    Batch process multiple Genarris output directories using SLURM parallel execution.

    Args:
        base_dir: Root directory containing multiple molecule directories
        output_dir: Output directory where processed results will be saved
        npartitions: Number of partitions for distributed processing
        ltol: Lattice parameter tolerance for structure deduplication
        stol: Site tolerance for structure deduplication
        angle_tol: Angle tolerance for structure deduplication

    Returns:
        List of submitit job objects for monitoring execution status
    """
    # Collect all molecule/conformer combinations for processing
    job_args = []
    for mol_dir in base_dir.iterdir():
        for conf_dir in mol_dir.iterdir():
            processed_dir = output_dir / mol_dir.name / conf_dir.name

            # Skip if already processed
            if (
                processed_dir.exists()
                and len(list(processed_dir.glob("*/*.parquet"))) > 0
            ):
                print(f"Skipping {conf_dir} because {processed_dir} already exists")
                continue

            job_args.append(
                (
                    process_genarris_outputs_single,
                    (conf_dir, processed_dir, ltol, stol, angle_tol, npartitions),
                    {},
                )
            )

    # Submit SLURM jobs
    return submit_slurm_jobs(
        job_args,
        job_name="process_genarris_outputs",
        output_dir=output_dir / "slurm",
        partition="ocp,learnaccel",
        cpus_per_task=80,
        mem_gb=400,
        timeout_min=1000,
    )
