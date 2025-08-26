"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure conversion and manipulation utilities for FastCSP.
Structure validation utilities for FastCSP.

This module provides functions to validate that crystal structures maintain
their chemical integrity during ML-based relaxation processes.

This module provides utilities for converting between different structure formats
(CIF, ASE Atoms, pymatgen Structure) and basic structure manipulation operations.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import sparse

if TYPE_CHECKING:
    from ase import Atoms


def cif_to_structure(cif: str) -> Structure | None:
    """
    Convert CIF string to pymatgen Structure object.
    """
    return Structure.from_str(cif, fmt="cif") if cif else None


def cif_to_atoms(cif: str) -> Atoms | None:
    """
    Convert CIF string to ASE Atoms object.
    """
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """
    Generate consistent partition ID for distributed processing.

    Creates a deterministic partition identifier based on MD5 hashing of the input key.
    This ensures consistent assignment of data to processing partitions across
    different runs, enabling reproducible distributed processing.

    Args:
        key: String identifier for data element (e.g., structure_id, formula)
        npartitions: Total number of partitions for load balancing (default: 1000)

    Returns:
        Partition ID in range [0, npartitions-1]
    """
    key_encoded = key.encode("utf-8")
    md5_hash = hashlib.md5()
    md5_hash.update(key_encoded)
    consistent_hash_hex = md5_hash.hexdigest()
    consistent_hash_int = int(consistent_hash_hex, 16)
    return consistent_hash_int % npartitions


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
    This approach dramatically reduces the number of structure pairs that need
    detailed comparison during deduplication.

    Args:
        structure: Pymatgen Structure object to hash
        z: Number of formula units per unit cell
        use_density: Include density in hash for geometric grouping
        use_volume: Include volume in hash for size-based grouping
        density_bin_size: Bin size for density discretization (g/cm³)
        vol_bin_size: Bin size for volume discretization (Ų)

    Returns:
        Hash string combining formula, Z, and optionally density/volume bins

    Hashing Strategy:
        1. Start with reduced chemical formula and Z value
        2. Add binned density if use_density=True for packing similarity
        3. Add binned volume if use_volume=True for volume grouping
        4. Combine components for readable hash

    Example:
        >>> get_structure_hash(structure, z=4, use_density=True)
        "C6H4O4_4_1.5_125.2"  # Formula_Z_density_volume
    """
    # Start with chemical composition and stoichiometry
    formula = structure.composition.reduced_formula
    hash_components = [formula, str(z)]

    # Add density-based grouping if requested
    if use_density:
        density = structure.density
        # Bin density to group structures with similar packing
        density_bin = round(density / density_bin_size) * density_bin_size
        hash_components.append(f"{density_bin:.1f}")

    # Add volume-based grouping if requested
    if use_volume:
        volume = structure.volume
        # Bin volume to group structures with similar cell sizes
        vol_bin = round(volume / vol_bin_size**3) * vol_bin_size**3
        hash_components.append(f"{vol_bin:.1f}")

    # Combine all components into single hash string
    return "_".join(hash_components)


def check_no_changes_in_covalent_matrix(
    initial_atoms: Atoms, final_atoms: Atoms
) -> bool:
    """
    Validate that covalent bonding network is preserved during structure relaxation.

    Compares the covalent bonding adjacency matrices before and after ML-based
    relaxation to detect unwanted chemical reconstructions. This validation ensures
    that the relaxation process only optimizes geometry without breaking or forming
    chemical bonds, which would indicate problematic initial structures or
    relaxation failures.

    Args:
        initial_atoms: Original structure before relaxation
        final_atoms: Structure after ML-based relaxation

    Returns:
        True if bonding network is preserved, False otherwise
        Returns False if either structure is None (error handling)

    Algorithm:
        1. Convert ASE Atoms to pymatgen Structures for analysis
        2. Use JmolNN to identify covalent neighbors in both structures
        3. Build adjacency matrices representing bonding networks
        4. Compare matrices for exact equality

    Validation Purpose:
        - Detect atom overlaps that lead to artificial bonding
        - Identify relaxation artifacts that break molecular integrity
        - Filter out reconstructions that change chemical connectivity
        - Ensure ML relaxation preserves intended molecular structure
    """
    # Handle error cases where structures couldn't be processed
    if initial_atoms is None or final_atoms is None:
        return False

    # Convert ASE Atoms to pymatgen Structures for neighbor analysis
    initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
    final_structure = AseAtomsAdaptor.get_structure(final_atoms)

    # Build adjacency matrix for initial structure using Jmol bonding radii
    initial_nn_info = JmolNN().get_all_nn_info(initial_structure)
    initial_nn_matrix = np.zeros((len(initial_nn_info), len(initial_nn_info)))
    for i in range(len(initial_nn_info)):
        for j in range(len(initial_nn_info[i])):
            # Mark bonded pairs in adjacency matrix
            initial_nn_matrix[i, initial_nn_info[i][j]["site_index"]] = 1

    # Build adjacency matrix for final (relaxed) structure
    final_nn_info = JmolNN().get_all_nn_info(final_structure)
    final_nn_matrix = np.zeros((len(final_nn_info), len(final_nn_info)))
    for i in range(len(final_nn_info)):
        for j in range(len(final_nn_info[i])):
            # Mark bonded pairs in adjacency matrix
            final_nn_matrix[i, final_nn_info[i][j]["site_index"]] = 1

    # Check that both bonding networks are identical
    # Any difference indicates bond formation/breaking during relaxation
    return np.array_equal(initial_nn_matrix, final_nn_matrix)


def check_no_changes_in_Z(
    initial_atoms: Atoms | None, final_atoms: Atoms | None
) -> bool:
    """
    Verify that the number of connected molecular components (Z) is preserved during relaxation.

    Validates that ML relaxation doesn't cause molecular fragmentation or aggregation
    by comparing the number of connected components in the bonding network. This is
    a more lenient check than covalent matrix comparison, focusing on overall
    molecular integrity rather than specific bond preservation.

    Args:
        initial_atoms: Original structure before relaxation
        final_atoms: Structure after ML-based relaxation

    Returns:
        True if number of connected components is preserved, False otherwise
        Returns False if either structure is None (error handling)

    Algorithm:
        1. Build covalent bonding adjacency matrices for both structures
        2. Use scipy sparse graph algorithms to find connected components
        3. Compare the number of connected components (Z values)
        4. Return True only if Z values are identical

    Use Cases:
        - Detect molecular fragmentation during relaxation
        - Identify aggregation events that merge separate molecules
        - Quality control for structures with multiple molecular units
        - Less strict than full connectivity validation
    """
    # Handle error cases where structures couldn't be processed
    if initial_atoms is None or final_atoms is None:
        return False

    # Convert ASE Atoms to pymatgen Structures for bonding analysis
    initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
    final_structure = AseAtomsAdaptor.get_structure(final_atoms)

    # Build adjacency matrix for initial structure
    initial_nn_info = JmolNN().get_all_nn_info(initial_structure)
    initial_nn_matrix = np.zeros((len(initial_nn_info), len(initial_nn_info)))

    for i in range(len(initial_nn_info)):
        for j in range(len(initial_nn_info[i])):
            initial_nn_matrix[i, initial_nn_info[i][j]["site_index"]] = 1

    # Count connected components in initial structure using graph theory
    Z1 = sparse.csgraph.connected_components(initial_nn_matrix)[0]

    # Build adjacency matrix for final (relaxed) structure
    final_nn_info = JmolNN().get_all_nn_info(final_structure)
    final_nn_matrix = np.zeros((len(final_nn_info), len(final_nn_info)))
    for i in range(len(final_nn_info)):
        for j in range(len(final_nn_info[i])):
            final_nn_matrix[i, final_nn_info[i][j]["site_index"]] = 1

    # Count connected components in final structure
    Z2 = sparse.csgraph.connected_components(final_nn_matrix)[0]

    # Verify that the number of molecular units is preserved
    return Z1 == Z2
