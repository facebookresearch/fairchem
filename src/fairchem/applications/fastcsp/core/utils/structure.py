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
from scipy.sparse import csgraph

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
    Convert CIF string to ASE Atoms object.
    """
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """
    Generate consistent partition ID from key using MD5 hash.
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
    Generate hash string for structure grouping based on formula and binned properties.

    Args:
        structure: Pymatgen Structure object to hash
        z: Number of formula units per unit cell
        use_density: Include density in hash for geometric grouping
        use_volume: Include volume in hash for size-based grouping
        density_bin_size: Bin size for density discretization (g/cm³)
        vol_bin_size: Bin size for volume discretization (Ų)

    Returns:
        Hash string like "C6H4O4_z4_d1.5_v125.2".
    """
    # Start with chemical composition
    formula = structure.composition.reduced_formula
    hash_components = [formula, "z" + str(z)]

    # Add density-based grouping if requested
    if use_density:
        density = structure.density
        # Bin density to group structures with similar packing
        density_bin = round(density / density_bin_size) * density_bin_size
        hash_components.append(f"d{density_bin:.1f}")

    # Add volume-based grouping if requested
    if use_volume:
        volume = structure.volume
        # Bin volume to group structures with similar cell sizes
        vol_bin = round(volume / vol_bin_size**3) * vol_bin_size**3
        hash_components.append(f"v{vol_bin:.1f}")

    # Combine all components into single hash string
    return "_".join(hash_components)


def check_correct_z(
    structure_or_atoms: Structure | Atoms | None,
    requested_z: int,
) -> bool:
    """
    Check whether the number of connected molecular fragments in the cell
    matches the requested number of formula units (Z).

    Mirrors the algorithm in csp_benchmark's ``check_no_changes_in_Z``
    (counts connected components of the JmolNN adjacency matrix), but
    compares the observed count against an integer ``requested_z`` instead
    of a second structure.

    Args:
        structure_or_atoms: Pymatgen ``Structure`` or ASE ``Atoms`` for the
            generated unit cell. Returns ``False`` if ``None``.
        requested_z: Integer Z value the generator was asked for.

    Returns:
        True if the JmolNN connected-component count equals ``requested_z``,
        False otherwise (or if the input is ``None``).
    """
    # Handle error cases where the structure couldn't be processed
    if structure_or_atoms is None:
        return False

    # Accept either a pymatgen Structure or an ASE Atoms; convert if needed
    if isinstance(structure_or_atoms, Structure):
        structure = structure_or_atoms
    else:
        structure = AseAtomsAdaptor.get_structure(structure_or_atoms)

    # Build adjacency matrix using Jmol bonding radii (same idiom as
    # check_no_changes_in_covalent_matrix below).
    nn_info = JmolNN().get_all_nn_info(structure)
    nn_matrix = np.zeros((len(nn_info), len(nn_info)))
    for i in range(len(nn_info)):
        for j in range(len(nn_info[i])):
            nn_matrix[i, nn_info[i][j]["site_index"]] = 1

    # Connected-component count == observed number of molecular fragments
    observed_z = csgraph.connected_components(nn_matrix)[0]
    return observed_z == requested_z


def check_no_changes_in_covalent_matrix(
    initial_atoms: Atoms, final_atoms: Atoms
) -> bool:
    """
    Check if covalent bonding network is preserved after relaxation.

    Args:
        initial_atoms: Structure before relaxation.
        final_atoms: Structure after relaxation.

    Returns:
        True if bonding network unchanged, False otherwise.
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
