"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure deduplication utilities for FastCSP.

This module provides algorithms for removing duplicate crystal structures
using pymatgen StructureMatcher with hash-based pre-filtering for performance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from fairchem.applications.fastcsp.core.utils.structure import get_structure_hash
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher

if TYPE_CHECKING:
    import pandas as pd


def process_structure_group(group_data, ltol=0.2, stol=0.3, angle_tol=5):
    """
    Apply crystallographic deduplication within a pre-filtered structure group.

    Performs detailed crystallographic comparison using pymatgen StructureMatcher
    on structures that have been pre-filtered by hash-based grouping. This two-stage
    approach (hash pre-filtering + detailed comparison) dramatically improves
    deduplication efficiency for large structure datasets.

    Args:
        group_data: (indices, structures) where:
                   - indices: List of original DataFrame indices
                   - structures: List of pymatgen Structure objects
        ltol: Lattice parameter tolerance (fractional difference)
        stol: Site tolerance (Ångström) for atomic position matching
        angle_tol: Angle tolerance (degrees) for lattice angle matching

    Returns:
        List of (index, subgroup_id) tuples assigning each structure
        to a crystallographic equivalence class within the group

    Algorithm:
        1. Handle trivial case: single structure returns immediately
        2. Initialize StructureMatcher with specified tolerances
        3. Use greedy clustering approach:
           - Take first unmatched structure as reference
           - Find all structures matching the reference
           - Assign matched structures to same subgroup
           - Repeat with remaining unmatched structures
        4. Return subgroup assignments for all structures

    Complexity:
        - Worst case: O(n²) comparisons for n structures
        - Typical case: Much better due to hash pre-filtering
        - Memory efficient: processes one group at a time
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
    df: pd.DataFrame,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    n_jobs: int = 120,
    remove_duplicates: bool = True,
    hash_density: bool = True,
    hash_volume: bool = True,
):
    """
    Perform efficient deduplication of crystals.

    Implements a two-stage deduplication algorithm that combines hash-based pre-filtering
    with detailed crystal comparison for optimal performance on large scale.

    Args:
        df: DataFrame containing structures with 'structure' and 'z' columns
        ltol: Lattice parameter tolerance for StructureMatcher
        stol: Site tolerance for StructureMatcher
        angle_tol: Angle tolerance for StructureMatcher
        n_jobs: Number of parallel processes for deduplication (default: 120)
        remove_duplicates: Whether to keep only one structure per group
        hash_density: Include density in hash for geometric grouping
        hash_volume: Include volume in hash for size-based grouping

    Returns:
        DataFrame with added 'group_index' column indicating
        crystallographically equivalent structures. Optionally
        filtered to remove duplicates if remove_duplicates=True.

    Algorithm Overview:
        1. **Hash-based Pre-filtering**: Group structures by chemical formula,
           Z value, and optionally density/volume bins
        2. **Parallel Processing**: Use p_map to process hash groups in parallel
        3. **Crystallographic Comparison**: Apply StructureMatcher within each hash group
        4. **Group Assignment**: Assign unique group_index to each equivalence class
        5. **Optional Filtering**: Remove duplicates keeping one representative per group

    Performance Benefits:
        - Hash pre-filtering reduces expensive comparisons by orders of magnitude
        - Parallel processing across hash groups for optimal CPU utilization
        - Memory efficient: processes groups independently
        - Scales to datasets with millions of structures
    """
    # Stage 1: Generate hash-based groups for pre-filtering
    print("Generating structure hashes for pre-filtering...")
    hashes = df[["structure", "z"]].apply(
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
    print("Number of unique hashes: ", len(hash_groups))

    # Stage 2: Prepare data for parallel crystallographic comparison
    groups_to_process = []
    for _, indices in hash_groups:
        # Extract structures for this hash group
        groups_to_process.append((indices, df["structure"].to_numpy()[indices]))

    # Stage 3: Parallel crystallographic deduplication within hash groups
    num_groups = len(groups_to_process)
    print(f"Processing {num_groups} hash groups in parallel...")
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

    print(
        "Number of groups: ",
        len({match[1] for match in all_matches}),
        len(all_matches),
    )

    # Stage 5: Apply group assignments to DataFrame
    all_matches.sort(key=lambda x: x[0])  # Sort by original DataFrame index
    df["group_index"] = [match[1] for match in all_matches]

    # Stage 6: Optional duplicate removal (keep one representative per group)
    if remove_duplicates:
        print("Removing duplicates, keeping one structure per group...")
        df.drop_duplicates(subset=["group_index"], inplace=True)

    return df
