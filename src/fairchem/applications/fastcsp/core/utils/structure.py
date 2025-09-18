"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Conversion, Manipulation, and Validation Utilities for FastCSP

This module provides essential utilities for handling crystal structures throughout
the FastCSP workflow. It implements efficient conversions between different structure
representations, validation algorithms for structural integrity, and functions
for high-throughput crystal structure processing.

Key Features:
- Structure hashing for efficient comparison and caching
- Distributed processing support with consistent partitioning
- Chemical composition validation and bonding analysis
- Quality control checks for structural integrity

Structure Validation:
- Atomic composition conservation (Z-number preservation)
- Covalent bonding network analysis using coordination environments

The module is designed for both individual structure operations and batch processing
of large crystal structure datasets common in high-throughput materials discovery.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase import Atoms


def cif_to_structure(cif: str) -> Structure | None:
    """
    Convert CIF (Crystallographic Information File) string to pymatgen Structure object.

    Args:
        cif: CIF format string containing crystal structure data

    Returns:
        Structure object if conversion successful, None if cif is empty/invalid
    """
    return Structure.from_str(cif, fmt="cif") if cif else None


def cif_to_atoms(cif: str) -> Atoms | None:
    """
    Convert CIF string to ASE (Atomic Simulation Environment) Atoms object.

    This function provides a direct path from CIF format to ASE Atoms objects,
    which are commonly used for structure optimization and analysis.

    Args:
        cif: CIF format string containing crystal structure data

    Returns:
        ASE Atoms object if conversion successful, None if cif is empty/invalid
    """
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """
    Generate a consistent partition ID for distributed processing of structures.

    This function creates deterministic partitioning for parallel processing,
    ensuring that structures with the same key always map to the same partition.

    Args:
        key: String identifier for the structure (e.g., molecule_name + space_group)
        npartitions: Total number of partitions for distribution (default: 1000)

    Returns:
        int: Partition ID in range [0, npartitions-1]

    Notes:
        - Deterministic: same key always produces same partition ID
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
