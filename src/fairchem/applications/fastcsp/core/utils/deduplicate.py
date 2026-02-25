"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Deduplication Utilities for FastCSP
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.structure import get_structure_hash
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher

if TYPE_CHECKING:
    import pandas as pd


def process_structure_group(group_data, ltol=0.2, stol=0.3, angle_tol=5):
    """
    Apply crystallographic deduplication within a pre-filtered structure group.
    """
    indices, structures = group_data

    # Handle trivial case: single structure in group
    if len(structures) == 1:
        return [(indices[0], 0)]

    # Configure StructureMatcher for crystallographic comparison
    sm = StructureMatcher(
        ltol=ltol,  # Lattice parameter tolerance
        stol=stol,  # Site position tolerance
        angle_tol=angle_tol,  # Lattice angle tolerance
    )

    # Initialize data structures for greedy clustering
    unmatched = list(enumerate(structures))  # (local_idx, structure) pairs
    group_assignments = []
    subgroup_id = 0

    # Greedy clustering: repeatedly find connected components
    while unmatched:
        # Take first unmatched structure as reference for new subgroup
        i, ref_struct = unmatched.pop(0)
        current_group = [indices[i]]  # Start new subgroup with reference
        to_remove = []

        # Find all structures that match the reference
        for j, (idx, test_struct) in enumerate(unmatched):
            if sm.fit(ref_struct, test_struct):
                current_group.append(indices[idx])
                to_remove.append(j)

        # Remove matched structures from unmatched list
        for j in sorted(to_remove, reverse=True):
            if len(unmatched) > 0:
                unmatched.pop(j)

        # Assign all structures in current group to same subgroup ID
        group_assignments.extend([(idx, subgroup_id) for idx in current_group])
        subgroup_id += 1

    return group_assignments


def deduplicate_structures(
    structures_df: pd.DataFrame,
    hash_density: bool = True,
    hash_volume: bool = True,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    remove_duplicates: bool = False,
    n_jobs: int = 70,
):
    """
    Implements a two-stage deduplication algorithm that combines hash-based pre-filtering
    with detailed crystal comparison for optimal performance on large scale.
    """
    logger = get_central_logger()

    # Stage 1: Generate hash-based groups for pre-filtering
    logger.info("Generating structure hashes for pre-filtering...")
    logger.info(f"Hashing settings - Density: {hash_density}, Volume: {hash_volume}")
    logger.info(f"Total structures to process: {len(structures_df)}")
    logger.info(f"Structure DataFrame head:\n{structures_df.head()}")
    hashes = structures_df[["structure", "z"]].apply(
        lambda x: get_structure_hash(
            x["structure"],
            x["z"],
            hash_density,  # Use density for geometric similarity grouping
            hash_volume,  # Use volume for size-based grouping
        ),
        axis=1,
    )

    # Group structures by hash for efficient pre-filtering
    hash_groups = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_groups[h].append(i)
    hash_groups = list(hash_groups.items())
    logger.info(f"Number of unique hashes: {len(hash_groups)}")

    # Stage 2: Prepare data for parallel crystallographic comparison
    groups_to_process = []
    for _, indices in hash_groups:
        # Extract structures for this hash group
        groups_to_process.append(
            (indices, structures_df["structure"].to_numpy()[indices])
        )

    # Stage 3: Parallel crystallographic deduplication within hash groups
    num_groups = len(groups_to_process)
    logger.info(f"Processing {num_groups} hash groups in parallel...")
    results = p_map(
        process_structure_group,  # Function to process each group
        groups_to_process,  # List of (indices, structures) tuples
        [ltol] * num_groups,  # Broadcast parameters to all groups
        [stol] * num_groups,
        [angle_tol] * num_groups,
        num_cpus=n_jobs,  # Parallel processing across hash groups
    )

    # Stage 4: Combine results and assign global group indices
    all_matches = []
    for (hash_val, _), group_results in zip(hash_groups, results):
        for idx, subgroup in group_results:
            # Create globally unique group identifier
            all_matches.append((idx, f"{hash_val}_{subgroup}"))

    unique_groups = len({match[1] for match in all_matches})
    logger.info(
        f"Deduplication completed: {unique_groups} unique groups from {len(all_matches)} structures"
    )

    # Stage 5: Apply group assignments to DataFrame
    all_matches.sort(key=lambda x: x[0])  # Sort by original DataFrame index
    structures_df["group_index"] = [match[1] for match in all_matches]

    # Stage 6: Optional duplicate removal (keep one representative per group)
    if remove_duplicates:
        logger.info("Removing duplicates, keeping one structure per group...")
        structures_df = structures_df.drop_duplicates(
            subset=["group_index"]
        ).reset_index(drop=True)
        logger.info(f"Structures after deduplication: {len(structures_df)}")
    return structures_df
