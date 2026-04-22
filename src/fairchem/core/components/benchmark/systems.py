"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fairchem.core.datasets.common_structures import (
    get_fcc_crystal_by_num_atoms,
    get_water_box,
)

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class BenchmarkSystem:
    """
    A benchmark system with metadata.
    """

    name: str
    atoms: Atoms
    task_name: str


def make_benchmark_system(
    name: str,
    task_name: str,
    natoms: int = 200,
    structure_type: str = "fcc",
    num_molecules: int = 20,
    seed: int = 42,
) -> BenchmarkSystem:
    """
    Create a BenchmarkSystem from a structure type.

    Args:
        name: Human-readable identifier.
        task_name: UMA task name (omat, omol, oc20, etc.).
        natoms: Number of atoms for fcc structures.
        structure_type: One of "fcc" or "water_box".
        num_molecules: Number of molecules for water_box.
        seed: Random seed for reproducibility.

    Returns:
        A BenchmarkSystem instance.
    """
    if structure_type == "fcc":
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))
        atoms = get_fcc_crystal_by_num_atoms(natoms)
    elif structure_type == "water_box":
        atoms = get_water_box(num_molecules=num_molecules, seed=seed)
    else:
        raise ValueError(
            f"Unknown structure_type: {structure_type}. "
            "Must be 'fcc' or 'water_box'."
        )
    return BenchmarkSystem(name=name, atoms=atoms, task_name=task_name)


def get_default_benchmark_systems(seed: int = 42) -> list[BenchmarkSystem]:
    """
    Return a list of default benchmark systems.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A list of 3 BenchmarkSystem instances.
    """
    return [
        make_benchmark_system(
            name="small_molecule",
            structure_type="water_box",
            num_molecules=20,
            task_name="omol",
            seed=seed,
        ),
        make_benchmark_system(
            name="medium_bulk",
            structure_type="fcc",
            natoms=200,
            task_name="omat",
            seed=seed,
        ),
        make_benchmark_system(
            name="large_bulk",
            structure_type="fcc",
            natoms=1000,
            task_name="omat",
            seed=seed,
        ),
    ]


def make_variable_size_batch(
    sizes: list[int],
    task_name: str = "omat",
    seed: int = 42,
) -> list[BenchmarkSystem]:
    """
    Create multiple FCC systems with varying atom counts.

    Args:
        sizes: List of atom counts.
        task_name: UMA task name.
        seed: Random seed for reproducibility.

    Returns:
        A list of BenchmarkSystem instances.
    """
    return [
        make_benchmark_system(
            name=f"batch_{natoms}",
            structure_type="fcc",
            natoms=natoms,
            task_name=task_name,
            seed=seed + i,
        )
        for i, natoms in enumerate(sizes)
    ]
