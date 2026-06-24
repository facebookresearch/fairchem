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
    get_slab_adsorbate,
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
        structure_type: One of "fcc", "water_box", or "slab_adsorbate".
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
    elif structure_type == "slab_adsorbate":
        atoms = get_slab_adsorbate()
    else:
        raise ValueError(
            f"Unknown structure_type: {structure_type}. "
            "Must be 'fcc', 'water_box', or 'slab_adsorbate'."
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
        make_benchmark_system(
            name="slab_adsorbate",
            structure_type="slab_adsorbate",
            task_name="oc20",
            seed=seed,
        ),
    ]


@dataclass
class SystemPool:
    """
    A fixed collection of BenchmarkSystem entries used to materialize mixed batches.

    The pool is treated as immutable during a benchmark run: ground-truth
    inference is computed once per entry and reused across all sampled batches.
    """

    systems: list[BenchmarkSystem]

    def __post_init__(self):
        names = [s.name for s in self.systems]
        if len(set(names)) != len(names):
            raise ValueError("SystemPool requires unique system names.")

    def __len__(self) -> int:
        return len(self.systems)

    def signature(self) -> list[dict[str, int | str]]:
        """
        Stable identity for cache keying. Sorted by name so insertion order
        of the pool does not change the key.
        """
        return sorted(
            (
                {
                    "name": s.name,
                    "task_name": s.task_name,
                    "num_atoms": len(s.atoms),
                }
                for s in self.systems
            ),
            key=lambda d: d["name"],
        )


def get_diverse_benchmark_pool(
    seed: int = 42,
    size_buckets: tuple[int, ...] = (20, 100, 500),
    n_per_bucket: int = 2,
    tasks: tuple[str, ...] = ("oc20", "omat", "omol", "odac", "omc"),
    bucket_tolerance: float = 0.5,
) -> SystemPool:
    """
    Build a pool of diverse benchmark systems covering all configured UMA tasks
    and size buckets.

    Reuses ``benchmark.fake_dataset.generate_structures`` so the size
    distributions match the training-benchmark datasets. For each task, we
    generate a wide pool of structures and then sample ``n_per_bucket`` for
    each requested size bucket, picking the structures whose atom count is
    closest to the bucket center (within ``bucket_tolerance`` relative window).

    Args:
        seed: Base seed (offset per task for variety).
        size_buckets: Target atom-count anchors (small/medium/large/...).
        n_per_bucket: Variants per (task, bucket) cell.
        tasks: UMA task names to include. Each must be in
            ``fake_dataset.BENCHMARK_DATASET_SPECS``.
        bucket_tolerance: Relative window around each bucket center used for
            candidate selection (e.g. 0.5 → [0.5*b, 1.5*b]).

    Returns:
        SystemPool with up to ``len(tasks) * len(size_buckets) * n_per_bucket``
        entries (may be fewer if a task's size range cannot satisfy a bucket).
    """
    from fairchem.core.components.benchmark.fake_dataset import (
        BENCHMARK_DATASET_SPECS,
        FakeDatasetConfig,
        generate_structures,
    )

    rng = np.random.default_rng(seed)
    systems: list[BenchmarkSystem] = []

    for task_idx, task in enumerate(tasks):
        if task not in BENCHMARK_DATASET_SPECS:
            raise ValueError(
                f"Unknown task {task!r}; supported: {sorted(BENCHMARK_DATASET_SPECS)}"
            )
        spec = BENCHMARK_DATASET_SPECS[task]
        # Generate a wide candidate pool we can bucket from. We want enough
        # structures that each requested bucket has options, even after
        # filtering by size window.
        n_candidates = max(64, len(size_buckets) * n_per_bucket * 4)
        config = FakeDatasetConfig(
            name=f"pool_{task}",
            split="pool",
            n_systems=n_candidates,
            system_size_range=tuple(spec["system_size_range"]),
            energy_std=spec["energy_std"],
            forces_std=spec["forces_std"],
            energy_mean=0.0,
            src=f"/tmp/_pool_{task}.unused",  # not written; we don't call create_fake_dataset
            seed=int(rng.integers(0, 2**31)) + task_idx,
            pbc=spec["pbc"],
        )
        candidates = generate_structures(config)
        candidate_sizes = np.array([len(a) for a in candidates])

        for bucket in size_buckets:
            lo = max(1, int(bucket * (1 - bucket_tolerance)))
            hi = max(lo + 1, int(bucket * (1 + bucket_tolerance)))
            # Indices of candidates whose size lands in the window, sorted by
            # closeness to the bucket center.
            in_window = np.where((candidate_sizes >= lo) & (candidate_sizes <= hi))[0]
            if len(in_window) == 0:
                continue
            order = np.argsort(np.abs(candidate_sizes[in_window] - bucket))
            chosen = in_window[order][:n_per_bucket]
            for variant, idx in enumerate(chosen):
                atoms = candidates[int(idx)]
                systems.append(
                    BenchmarkSystem(
                        name=f"{task}_b{bucket}_v{variant}_n{len(atoms)}",
                        atoms=atoms,
                        task_name=task,
                    )
                )

    if not systems:
        raise RuntimeError(
            "No systems matched the requested (tasks, size_buckets). "
            "Try widening bucket_tolerance or picking sizes inside each "
            "task's system_size_range."
        )
    return SystemPool(systems=systems)


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
