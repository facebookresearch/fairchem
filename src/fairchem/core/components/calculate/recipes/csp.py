"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from fairchem.core.components.calculate.recipes.local_env import construct_bond_matrix

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.analysis.local_env import NearNeighbors
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


def get_structure_hash(
    structure: Structure,
    z: int,
    use_density: bool = True,
    use_volume: bool = True,
    density_bin_size: float = 0.1,
    vol_bin_size: float = 0.2,
) -> str:
    """
    Generate a hash string for crystal structure grouping and fast pre-filtering.

    Creates a binned hash based on chemical formula and geometric properties to
    enable fast pre-filtering before expensive crystallographic comparisons.

    Args:
        structure: Pymatgen Structure object to hash
        z: Number of formula units per unit cell
        use_density: Include density in hash for geometric grouping
        use_volume: Include volume in hash for size-based grouping
        density_bin_size: Bin size for density discretization (g/cm3)
        vol_bin_size: Bin size for volume discretization (A3)

    Returns:
        Hash string combining formula, Z, and optionally density/volume bins
    """
    formula = structure.composition.reduced_formula
    hash_components = [formula, str(z)]

    if use_density:
        density = structure.density
        density_bin = round(density / density_bin_size) * density_bin_size
        hash_components.append(f"{density_bin:.1f}")

    if use_volume:
        volume = structure.volume
        vol_bin = round(volume / vol_bin_size**3) * vol_bin_size**3
        hash_components.append(f"{vol_bin:.1f}")

    return "_".join(hash_components)


def check_no_changes_in_covalent_matrix(
    initial_atoms: Atoms, final_atoms: Atoms
) -> bool:
    """
    Validate that covalent bonding network is preserved between two structures.

    Compares the covalent bonding adjacency matrices of two structures to detect
    unwanted chemical reconstructions. Useful for verifying that relaxation only
    optimizes geometry without breaking or forming chemical bonds.

    Args:
        initial_atoms: Original structure (e.g. before relaxation)
        final_atoms: Modified structure (e.g. after relaxation)

    Returns:
        True if bonding network is preserved, False otherwise.
        Returns False if either structure is None.
    """
    if initial_atoms is None or final_atoms is None:
        return False

    nn_finder = JmolNN()
    initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
    final_structure = AseAtomsAdaptor.get_structure(final_atoms)

    initial_matrix = construct_bond_matrix(initial_structure, nn_finder=nn_finder)
    final_matrix = construct_bond_matrix(final_structure, nn_finder=nn_finder)

    return np.array_equal(initial_matrix, final_matrix)


def match_and_compute_rmsd(
    reference_structure: Structure,
    relaxed_structure: Structure,
    structure_matcher: StructureMatcher | None = None,
    nn_finder: NearNeighbors | None = None,
) -> float | None:
    """
    Match two structures and compute RMSD with bond network validation.

    Uses StructureMatcher internal methods to obtain site permutations, then
    validates that the covalent bonding network is preserved. If the match is
    valid, computes a de-normalized RMSD.

    Args:
        reference_structure: Target/experimental crystal structure
        relaxed_structure: Predicted/relaxed crystal structure
        structure_matcher: Pre-configured StructureMatcher. Defaults to
            StructureMatcher(primitive_cell=False).
        nn_finder: NearNeighbors instance for bond detection. Defaults to JmolNN().

    Returns:
        RMSD value if structures match and bonds are preserved, None otherwise.
    """
    if structure_matcher is None:
        structure_matcher = StructureMatcher(primitive_cell=False)
    if nn_finder is None:
        nn_finder = JmolNN()

    # Preprocess structures (Niggli reduction, etc.)
    reference_structure, relaxed_structure, _, _ = structure_matcher._preprocess(
        reference_structure, relaxed_structure, niggli=True
    )
    # Match structures; returns (rms, max_dist, mask, cost, mapping) or None
    match = structure_matcher._match(
        reference_structure, relaxed_structure, fu=1, use_rms=True
    )
    if match is None:
        return None

    # Validate bond network with site permutations from match
    reference_matrix = construct_bond_matrix(reference_structure, nn_finder=nn_finder)
    relaxed_matrix = construct_bond_matrix(
        relaxed_structure,
        nn_finder=nn_finder,
        site_permutations=match[4],
    )
    if not np.array_equal(relaxed_matrix, reference_matrix):
        return None

    # De-normalize RMSD: pmg normalizes by (V/nsites)^(1/3)
    avg_vol = (relaxed_structure.volume + reference_structure.volume) / 2
    return match[0] * (avg_vol / len(reference_structure)) ** (1 / 3)


def match_structure_pymatgen(
    pred_structure: Structure,
    target_structure: Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    ignored_species: list[str] | None = None,
) -> tuple[bool, float | None]:
    """
    Compare two crystal structures using pymatgen StructureMatcher.

    Args:
        pred_structure: Predicted crystal structure
        target_structure: Reference crystal structure
        ltol: Lattice parameter tolerance
        stol: Site position tolerance
        angle_tol: Lattice angle tolerance (degrees)
        ignored_species: List of species to ignore in comparison (e.g. ["H"])

    Returns:
        Tuple of (is_match, rms_dist) where rms_dist is None if no match.
    """
    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        ignored_species=ignored_species,
    )
    if matcher.fit(pred_structure, target_structure):
        rms_dist = matcher.get_rms_dist(pred_structure, target_structure)[0]
        return True, rms_dist
    return False, None


def match_structures_to_references_pymatgen(
    pred_structure: Structure,
    target_structures: dict[str, Structure],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    ignore_h: bool = True,
) -> tuple[str | None, float | None]:
    """
    Compare a predicted crystal structure against a dictionary of reference structures.

    Returns the best matching reference code and its RMS distance.

    Args:
        pred_structure: Predicted crystal structure
        target_structures: Dict mapping reference codes to target Structures
        ltol: Lattice parameter tolerance
        stol: Site position tolerance
        angle_tol: Lattice angle tolerance (degrees)
        ignore_h: Whether to ignore hydrogen positions in comparison

    Returns:
        Tuple of (best_refcode, best_rms_dist), both None if no match found.
    """
    ignored_species = ["H"] if ignore_h else None
    best_match_refcode = None
    best_rmsd = float("inf")

    for refcode, target_structure in target_structures.items():
        is_match, rms_dist = match_structure_pymatgen(
            pred_structure,
            target_structure,
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            ignored_species=ignored_species,
        )
        if is_match and rms_dist < best_rmsd:
            best_match_refcode = refcode
            best_rmsd = rms_dist

    if best_match_refcode is not None:
        return best_match_refcode, best_rmsd
    return None, None


def process_structure_group(
    group_data: tuple[list[int], list[Structure]],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
) -> list[tuple[int, int]]:
    """
    Apply greedy crystallographic clustering within a pre-filtered structure group.

    Takes the first unmatched structure as a cluster representative, finds all
    structures matching it, assigns them to the same subgroup, and repeats.

    Args:
        group_data: Tuple of (indices, structures) where indices are original
            DataFrame indices and structures are pymatgen Structure objects
        ltol: Lattice parameter tolerance
        stol: Site position tolerance
        angle_tol: Lattice angle tolerance (degrees)

    Returns:
        List of (index, subgroup_id) tuples
    """
    indices, structures = group_data

    if len(structures) == 1:
        return [(indices[0], 0)]

    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    unmatched = list(enumerate(structures))
    group_assignments = []
    subgroup_id = 0

    while unmatched:
        i, ref_struct = unmatched.pop(0)
        current_group = [indices[i]]
        to_remove = []

        for j, (idx, test_struct) in enumerate(unmatched):
            if sm.fit(ref_struct, test_struct):
                current_group.append(indices[idx])
                to_remove.append(j)

        for j in sorted(to_remove, reverse=True):
            if len(unmatched) > 0:
                unmatched.pop(j)

        group_assignments.extend([(idx, subgroup_id) for idx in current_group])
        subgroup_id += 1

    return group_assignments


def _process_group_wrapper(args):
    """
    Wrapper for process_structure_group to unpack arguments for ProcessPoolExecutor.
    """
    group_data, ltol, stol, angle_tol = args
    return process_structure_group(
        group_data, ltol=ltol, stol=stol, angle_tol=angle_tol
    )


def deduplicate_structures(
    structures_df,
    hash_density: bool = True,
    hash_volume: bool = True,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    remove_duplicates: bool = False,
    n_jobs: int = 70,
):
    """
    Two-stage deduplication: hash-based pre-filtering + crystallographic comparison.

    Stage 1: Group structures by binned hash (formula, Z, density, volume).
    Stage 2: Within each group, greedy clustering via StructureMatcher.

    Args:
        structures_df: DataFrame with 'structure' (pymatgen Structure) and 'z' columns
        hash_density: Include density in hash grouping
        hash_volume: Include volume in hash grouping
        ltol: Lattice parameter tolerance for StructureMatcher
        stol: Site position tolerance for StructureMatcher
        angle_tol: Lattice angle tolerance (degrees)
        remove_duplicates: If True, keep only one structure per group
        n_jobs: Number of parallel workers

    Returns:
        DataFrame with 'group_index' column added (and duplicates removed if requested)
    """

    # Stage 1: Hash-based grouping
    logger.info(
        "Generating structure hashes (density=%s, volume=%s) for %d structures",
        hash_density,
        hash_volume,
        len(structures_df),
    )
    hashes = structures_df[["structure", "z"]].apply(
        lambda x: get_structure_hash(x["structure"], x["z"], hash_density, hash_volume),
        axis=1,
    )

    hash_groups = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_groups[h].append(i)
    hash_groups = list(hash_groups.items())
    logger.info("Number of unique hashes: %d", len(hash_groups))

    # Stage 2: Prepare groups for parallel crystallographic comparison
    groups_to_process = []
    for _, indices in hash_groups:
        groups_to_process.append(
            (indices, structures_df["structure"].to_numpy()[indices])
        )

    # Stage 3: Parallel crystallographic deduplication
    num_groups = len(groups_to_process)
    logger.info(
        "Processing %d hash groups in parallel with %d workers", num_groups, n_jobs
    )

    args_list = [(g, ltol, stol, angle_tol) for g in groups_to_process]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            tqdm(
                executor.map(_process_group_wrapper, args_list),
                total=num_groups,
                desc="Deduplicating",
            )
        )

    # Stage 4: Combine results
    all_matches = []
    for (hash_val, _), group_results in zip(hash_groups, results):
        for idx, subgroup in group_results:
            all_matches.append((idx, f"{hash_val}_{subgroup}"))

    unique_groups = len({match[1] for match in all_matches})
    logger.info(
        "Deduplication completed: %d unique groups from %d structures",
        unique_groups,
        len(all_matches),
    )

    # Stage 5: Apply group assignments
    all_matches.sort(key=lambda x: x[0])
    structures_df["group_index"] = [match[1] for match in all_matches]

    if remove_duplicates:
        logger.info("Removing duplicates, keeping one structure per group...")
        structures_df = structures_df.drop_duplicates(
            subset=["group_index"]
        ).reset_index(drop=True)
        logger.info("Structures after deduplication: %d", len(structures_df))

    return structures_df
