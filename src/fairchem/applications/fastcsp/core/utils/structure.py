"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Conversion, Manipulation, and Validation Utilities for FastCSP

This module provides essential utilities for handling crystal structures throughout
the FastCSP workflow. Core evaluation functions (structure hashing, bond validation)
are imported from fairchem.core.components.calculate.recipes.csp.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from fairchem.core.components.calculate.recipes.csp import (
    check_no_changes_in_covalent_matrix,
    get_structure_hash,
)

if TYPE_CHECKING:
    from ase import Atoms

__all__ = [
    "cif_to_atoms",
    "cif_to_structure",
    "check_no_changes_in_covalent_matrix",
    "get_partition_id",
    "get_structure_hash",
]


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

    Args:
        cif: CIF format string containing crystal structure data

    Returns:
        ASE Atoms object if conversion successful, None if cif is empty/invalid
    """
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """
    Generate a consistent partition ID for distributed processing of structures.

    Args:
        key: String identifier for the structure (e.g., molecule_name + space_group)
        npartitions: Total number of partitions for distribution (default: 1000)

    Returns:
        int: Partition ID in range [0, npartitions-1]
    """
    key_encoded = key.encode("utf-8")
    md5_hash = hashlib.md5()
    md5_hash.update(key_encoded)
    consistent_hash_hex = md5_hash.hexdigest()
    consistent_hash_int = int(consistent_hash_hex, 16)
    return consistent_hash_int % npartitions
