"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from fairchem.core.datasets.samplers.max_atom_distributed_sampler import (
    MaxAtomDistributedBatchSampler,
    get_batches,
)


def test_get_batches_single_rank():
    natoms_list = np.array([5, 1, 2, 3])
    indices = np.array([0, 3, 1, 2])
    batches, atom_counts, samples_filtered = get_batches(natoms_list, indices, 2, 0)
    assert len(batches) == 2
    assert samples_filtered == 2
    assert sum(atom_counts) == 3


def test_get_large_batches():
    large_count = 1000000
    reps = 30000
    natoms_list = np.ones(large_count, dtype=np.int64) * reps
    indices = np.arange(0, large_count, 1)
    batches, atom_counts, samples_filtered = get_batches(natoms_list, indices, reps, 0)
    assert len(batches) == large_count
    assert samples_filtered == 0
    assert sum(atom_counts) == large_count * reps


@pytest.mark.parametrize(
    "num_samples, max_atoms_source, max_atoms_sample",
    [
        (10000, 100, 100),
        (10000, 100, 150),
        (1000, 1000, 1000),
    ],
)
def test_no_samples_filtered(
    num_samples: int, max_atoms_source: int, max_atoms_sample: int
):
    rng = np.random.default_rng(0)
    natoms_list = rng.integers(1, max_atoms_source, size=num_samples)
    batches, atom_counts, samples_filtered = get_batches(
        natoms_list, np.arange(0, num_samples), max_atoms_sample, 0
    )
    assert len(batches) == len(atom_counts)
    assert samples_filtered == 0


@pytest.mark.parametrize(
    "num_samples, max_atoms_source, max_atoms_sample",
    [
        (10000, 100, 10),
        (1000, 1000, 100),
    ],
)
def test_samples_filtered(
    num_samples: int, max_atoms_source: int, max_atoms_sample: int
):
    rng = np.random.default_rng(0)
    natoms_list = rng.integers(1, max_atoms_source, size=num_samples)
    batches, atom_counts, samples_filtered = get_batches(
        natoms_list, np.arange(0, num_samples), max_atoms_sample, 0
    )
    assert len(batches) == len(atom_counts)
    assert samples_filtered > 0


def get_mock_dataset(min: int, max: int, n: int):
    rng = np.random.default_rng(0)
    mock_dataset = unittest.mock.MagicMock()
    num_atoms = rng.integers(min, max, size=n)
    mock_dataset.get_metadata.return_value = num_atoms
    mock_dataset.__len__.return_value = n
    return mock_dataset


def test_sampler_single_rank():
    # all systems have 1 atom
    sampler = MaxAtomDistributedBatchSampler(
        get_mock_dataset(1, 2, 1000), max_atoms=100, num_replicas=1, rank=0, seed=0
    )
    assert len(list(sampler)) == 10


@pytest.mark.parametrize(
    "num_samples, max_atoms_source, max_atoms_sample, world_size, drop_last",
    [
        (10000, 100, 10, 2, True),
        (12345, 100, 99, 3, True),
        (1000, 100, 150, 8, True),
        (1000, 300, 600, 8, True),
        (100000, 300, 600, 8, False),
    ],
)
def test_sampler_multi_rank(
    num_samples: int,
    max_atoms_source: int,
    max_atoms_sample: int,
    world_size: int,
    drop_last: bool,
):
    rank_samples = []
    for r in range(world_size):
        sampler = MaxAtomDistributedBatchSampler(
            dataset=get_mock_dataset(1, max_atoms_source, num_samples),
            max_atoms=max_atoms_sample,
            num_replicas=world_size,
            rank=r,
            seed=0,
            drop_last=drop_last,
        )
        rank_samples.append(list(sampler))

    # assert all ranks have more than 0 sample
    assert all(len(x) > 0 for x in rank_samples)
    # assert all ranks have identical nubmer of batches
    assert len(set(len(x) for x in rank_samples)) == 1
    # assert all ranks have unique sample indices, only if drop_last is true
    if drop_last:
        total_samples = []
        total_unique_samples = set()
        for x in rank_samples:
            total_unique_samples.update(set(sum(x, [])))
            total_samples += sum(x, [])
        assert len(total_samples) == len(total_unique_samples)


def test_sampler_reproducible():
    mock_dataset = get_mock_dataset(1, 100, 1000)
    sampler1 = MaxAtomDistributedBatchSampler(
        mock_dataset, max_atoms=100, num_replicas=3, rank=0, seed=0
    )
    sampler2 = MaxAtomDistributedBatchSampler(
        mock_dataset, max_atoms=100, num_replicas=3, rank=0, seed=0
    )
    for x, y in zip(sampler1, sampler2):
        assert x == y


def test_fast_forward():
    mock_dataset = get_mock_dataset(1, 100, 1000)
    sampler1 = MaxAtomDistributedBatchSampler(
        mock_dataset, max_atoms=100, num_replicas=3, rank=0, seed=0
    )
    sampler2 = MaxAtomDistributedBatchSampler(
        mock_dataset, max_atoms=100, num_replicas=3, rank=0, seed=0
    )
    skip = 17
    sampler1.set_epoch_and_start_iteration(0, skip)
    # sampler1 should have "skip" less batches than sampler2
    assert len(list(sampler1)) + skip == len(list(sampler2))
    # manually fast forward sampler 2 by "skip" number of steps
    iter1, iter2 = iter(sampler1), iter(sampler2)
    for i in range(skip):
        next(iter2)
    for b1, b2 in zip(iter1, iter2):
        for item1, item2 in zip(b1, b2):
            assert item1 == item2
