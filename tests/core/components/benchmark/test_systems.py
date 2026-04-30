"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from ase import Atoms

from fairchem.core.components.benchmark.systems import (
    BenchmarkSystem,
    archetype_of,
    get_default_benchmark_systems,
    make_benchmark_system,
    make_variable_size_batch,
)


def test_make_benchmark_system_fcc():
    sys = make_benchmark_system(name="fcc_200", natoms=200, task_name="omat")
    assert isinstance(sys, BenchmarkSystem)
    assert sys.name == "fcc_200"
    assert sys.task_name == "omat"
    assert isinstance(sys.atoms, Atoms)
    assert len(sys.atoms) == 200


def test_make_benchmark_system_water():
    sys = make_benchmark_system(
        name="water_60",
        structure_type="water_box",
        num_molecules=20,
        task_name="omol",
    )
    assert len(sys.atoms) == 60


def test_get_default_benchmark_systems():
    systems = get_default_benchmark_systems()
    assert len(systems) >= 3
    names = [s.name for s in systems]
    assert "small_molecule" in names
    assert "medium_bulk" in names
    assert "large_bulk" in names


def test_get_default_benchmark_systems_variants_are_distinct():
    """
    With n_per_archetype > 1 each archetype produces N distinct variants and
    the per-variant atom positions differ across draws.
    """
    n = 4
    systems = get_default_benchmark_systems(n_per_archetype=n)
    assert len(systems) == 3 * n

    archetypes = {archetype_of(s.name): [] for s in systems}
    for s in systems:
        archetypes[archetype_of(s.name)].append(s)
    assert set(archetypes) == {"small_molecule", "medium_bulk", "large_bulk"}

    for variants in archetypes.values():
        assert len(variants) == n
        # Same archetype -> same atom count
        assert len({len(v.atoms) for v in variants}) == 1
        # Different draws -> different positions
        positions = [v.atoms.get_positions() for v in variants]
        for i in range(1, n):
            assert (positions[0].shape != positions[i].shape) or not (
                positions[0] == positions[i]
            ).all()


def test_get_default_benchmark_systems_is_deterministic():
    """
    Identical calls produce bit-identical positions; different seeds diverge.
    """
    a = get_default_benchmark_systems(seed=42, n_per_archetype=2)
    b = get_default_benchmark_systems(seed=42, n_per_archetype=2)
    c = get_default_benchmark_systems(seed=43, n_per_archetype=2)
    for sa, sb in zip(a, b):
        assert (sa.atoms.get_positions() == sb.atoms.get_positions()).all()
    # At least one variant differs across seeds
    assert any(
        not (sa.atoms.get_positions() == sc.atoms.get_positions()).all()
        for sa, sc in zip(a, c)
    )


def test_archetype_of():
    assert archetype_of("medium_bulk") == "medium_bulk"
    assert archetype_of("medium_bulk_v007") == "medium_bulk"
    assert archetype_of("small_molecule_v000") == "small_molecule"


def test_make_variable_size_batch():
    batch = make_variable_size_batch(sizes=[10, 50, 100], task_name="omat")
    assert len(batch) == 3
    assert len(batch[0].atoms) == 10
    assert len(batch[2].atoms) == 100


def test_unknown_structure_type_raises():
    with pytest.raises(ValueError, match="Unknown structure_type"):
        make_benchmark_system(name="bad", structure_type="unknown", task_name="omat")
