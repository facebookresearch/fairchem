"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        atoms = get_fcc_crystal_by_num_atoms(natoms, seed=seed)
    elif structure_type == "water_box":
        atoms = get_water_box(num_molecules=num_molecules, seed=seed)
    else:
        raise ValueError(
            f"Unknown structure_type: {structure_type}. "
            "Must be 'fcc' or 'water_box'."
        )
    return BenchmarkSystem(name=name, atoms=atoms, task_name=task_name)


DEFAULT_ARCHETYPES: tuple[tuple[str, dict], ...] = (
    (
        "small_molecule",
        {"structure_type": "water_box", "num_molecules": 20, "task_name": "omol"},
    ),
    (
        "medium_bulk",
        {"structure_type": "fcc", "natoms": 200, "task_name": "omat"},
    ),
    (
        "large_bulk",
        {"structure_type": "fcc", "natoms": 1000, "task_name": "omat"},
    ),
)

VARIANT_SEP = "_v"


def archetype_of(name: str) -> str:
    """
    Return the archetype prefix for a variant name (e.g. "small_molecule_v007"
    -> "small_molecule"). Names without the variant separator are returned as-is.
    """
    head, sep, _ = name.rpartition(VARIANT_SEP)
    return head if sep else name


def get_default_benchmark_systems(
    seed: int = 42, n_per_archetype: int = 1
) -> list[BenchmarkSystem]:
    """
    Return default benchmark systems, optionally with multiple distinct variants
    per archetype.

    With ``n_per_archetype=1`` (default) this returns 3 systems with the
    canonical names ``small_molecule``, ``medium_bulk``, ``large_bulk``. With
    a larger value, each archetype contributes ``n_per_archetype`` distinct
    variants named ``<archetype>_v<idx>`` (e.g. ``medium_bulk_v007``); each
    variant gets its own deterministic seed so the set is reproducible across
    runs.

    Args:
        seed: Base random seed for reproducibility.
        n_per_archetype: Number of distinct variants per archetype.

    Returns:
        A flat list of 3 * ``n_per_archetype`` BenchmarkSystem instances.
    """
    out: list[BenchmarkSystem] = []
    # Per-archetype seed offsets; primes keep the per-variant seeds well-spaced
    # across archetypes so they don't collide as ``n_per_archetype`` grows.
    arch_offsets = (0, 100_003, 200_017)
    for arch_offset, (arch_name, kwargs) in zip(arch_offsets, DEFAULT_ARCHETYPES):
        for i in range(n_per_archetype):
            variant_name = (
                arch_name
                if n_per_archetype == 1
                else f"{arch_name}{VARIANT_SEP}{i:03d}"
            )
            out.append(
                make_benchmark_system(
                    name=variant_name,
                    seed=seed + arch_offset + i,
                    **kwargs,
                )
            )
    return out


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
