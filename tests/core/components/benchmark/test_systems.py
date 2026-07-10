"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from fairchem.core.components.benchmark.systems import (
    BenchmarkSystem,
    get_default_benchmark_systems,
    get_size_bucket_pool,
    make_benchmark_system,
    make_variable_size_batch,
)


def test_get_size_bucket_pool():
    sizes = (20, 80)
    variants = 3
    jitter = 0.1
    pool = get_size_bucket_pool(
        sizes=sizes, variants_per_size=variants, jitter=jitter, task="omat"
    )
    assert set(pool.keys()) == set(sizes)
    for size in sizes:
        systems = pool[size]
        assert len(systems) == variants
        lo = int(size * (1 - jitter))
        hi = int(size * (1 + jitter))
        for s in systems:
            assert isinstance(s, BenchmarkSystem)
            assert s.task_name == "omat"
            assert lo <= len(s.atoms) <= hi + 1
    # deterministic given the same seed
    pool2 = get_size_bucket_pool(
        sizes=sizes, variants_per_size=variants, jitter=jitter, task="omat"
    )
    for size in sizes:
        assert [len(s.atoms) for s in pool[size]] == [len(s.atoms) for s in pool2[size]]


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


def test_make_benchmark_system_slab_adsorbate():
    sys = make_benchmark_system(
        name="slab_ads", structure_type="slab_adsorbate", task_name="oc20"
    )
    atoms = sys.atoms
    assert sys.task_name == "oc20"

    # Has OC20-style tags (0=subsurface, 1=surface, 2=adsorbate)
    tags = atoms.get_tags()
    assert set(np.unique(tags)) == {0, 1, 2}

    # Has FixAtoms on subsurface
    assert len(atoms.constraints) == 1
    assert isinstance(atoms.constraints[0], FixAtoms)
    fixed_tags = tags[atoms.constraints[0].index]
    assert np.all(fixed_tags == 0)

    # Has charge and spin in info
    assert "charge" in atoms.info
    assert "spin" in atoms.info


def test_get_default_benchmark_systems():
    systems = get_default_benchmark_systems()
    assert len(systems) >= 4
    names = [s.name for s in systems]
    assert "small_molecule" in names
    assert "medium_bulk" in names
    assert "large_bulk" in names
    assert "slab_adsorbate" in names


def test_make_variable_size_batch():
    batch = make_variable_size_batch(sizes=[10, 50, 100], task_name="omat")
    assert len(batch) == 3
    assert len(batch[0].atoms) == 10
    assert len(batch[2].atoms) == 100


def test_unknown_structure_type_raises():
    with pytest.raises(ValueError, match="Unknown structure_type"):
        make_benchmark_system(name="bad", structure_type="unknown", task_name="omat")
