"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Iterator

import numba as nb
import numpy as np
import torch
from torch.utils.data import Sampler

from fairchem.core.common import gp_utils

if TYPE_CHECKING:
    from fairchem.core.datasets.base_dataset import BaseDataset


@nb.njit
def get_batches(
    natoms_list: np.array,  # List of number of atoms in each sample
    indices: np.array,  # Indices of the samples
    max_atoms: int,  # Maximum number of atoms allowed in a batch
    min_atoms: int,  # Minimum number of atoms allowed in a batch
) -> tuple[list[list[int]], list[int], int]:
    """
    Greedily creates batches from a list of samples with varying numbers of atoms.

    Args:
    natoms_list: Array of number of atoms in each sample.
    indices: Array of indices of the samples.
    max_atoms: Maximum number of atoms allowed in a batch.

    Returns:
    tuple[list[list[int]], list[int], int]:
        A tuple containing a list of batches, a list of the total number of atoms in each batch,
        and the number of samples that were filtered out because they exceeded the maximum number of atoms.
    """

    # Ensure the inputs are valid
    assert max_atoms > 0
    assert len(natoms_list) > 0
    assert len(natoms_list) == len(indices)

    # Initialize variables to keep track of the batches and the total number of atoms in each batch
    batches = []  # List of batches, where each batch is a list of indices
    run_sum = 0  # Running total of the number of atoms in the current batch
    cur_batch = nb.typed.List.empty_list(nb.int64)  # Current batch being constructed
    atom_counts = nb.typed.List.empty_list(
        nb.int64
    )  # List of total number of atoms in each batch
    samples_filtered = 0  # Number of samples filtered out because they exceeded the maximum number of atoms

    # Iterate over the samples
    for idx, atoms in zip(indices, natoms_list):
        # If the sample has too many atoms, filter it out
        if atoms > max_atoms:
            samples_filtered += 1
            continue

        # If adding the sample to the current batch would not exceed the maximum number of atoms, add it
        if run_sum + atoms <= max_atoms:
            cur_batch.append(idx)
            run_sum += atoms
        # Otherwise, start a new batch
        else:
            if run_sum >= min_atoms:
                batches.append(cur_batch)
                atom_counts.append(run_sum)
            cur_batch = nb.typed.List([idx])
            run_sum = atoms

    # Add the last batch
    if run_sum >= min_atoms:
        batches.append(cur_batch)
        atom_counts.append(run_sum)

    # Return the batches, atom counts, and the number of samples filtered out
    return [list(x) for x in batches], list(atom_counts), samples_filtered


class MaxAtomDistributedBatchSampler(Sampler[list[int]]):
    """
    A custom batch sampler that distributes batches across multiple GPUs to ensure efficient training.

    Args:
    dataset (BaseDataset): The dataset to sample from.
    max_atoms (int): The maximum number of atoms allowed in a batch.
    num_replicas (int): The number of GPUs to distribute the batches across.
    rank (int): The rank of the current GPU.
    seed (int): The seed for shuffling the dataset.
    shuffle (bool): Whether to shuffle the dataset. Defaults to True.
    drop_last (bool): Whether to drop the last batch if its size is less than the maximum allowed size. Defaults to False.

    This batch sampler is designed to work with the BaseDataset class and is optimized for distributed training.
    It takes into account the number of atoms in each sample and ensures that the batches are distributed evenly across GPUs.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        max_atoms: int,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        min_atoms: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.num_replicas = num_replicas
        self.rank = rank
        assert self.num_replicas > 0
        assert self.rank < self.num_replicas

        if gp_utils.initialized():
            assert (
                min_atoms >= gp_utils.get_gp_world_size()
            ), "Min atoms needs to be at least gp world size!"

        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.epoch = 0
        self.start_iter = 0

        # we pre-create the batches here and can only do this once, otherwise everytime we get this iterator we might get a different number of batches
        self.all_batches = self._prepare_batches()
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.all_batches) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.all_batches) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.all_batches) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        assert (
            len(self.all_batches) >= self.num_replicas
        ), "there are less batches than ranks!"

    def _prepare_batches(self) -> list[int]:
        # shuffle is not optional here since the metadata tend to be sorted by atom count and the resulting batches will be highly uneven
        rng = np.random.default_rng(self.seed)
        original_indices = rng.permutation(len(self.dataset))
        # TODO: this is slow
        t0 = time.time()
        natoms_list = self.dataset.get_metadata("natoms", original_indices.tolist())
        t1 = time.time()
        indices, atoms_count, samples_filtered = get_batches(
            np.array(natoms_list), original_indices, self.max_atoms, self.min_atoms
        )
        t2 = time.time()
        logging.info(
            f"Sampler batch generation times: get natoms: {t1 - t0}, total: {t2 - t0}"
        )
        logging.info(
            f"MaxAtomDistributedSampler generated {len(indices)} batches with total atoms {np.sum(natoms_list)}, max: {max(atoms_count)}, min: {min(atoms_count)}, mean: {np.mean(atoms_count)}, std: {np.std(atoms_count)}"
        )
        logging.info(
            f"{samples_filtered} samples were removed because they exceed {self.max_atoms} atoms"
        )
        return indices

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[list[int]]:
        # based on current rank, return an iterator over batches
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.all_batches), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.all_batches)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        # slice of batch indices
        batch_slice = [self.all_batches[i] for i in indices]
        assert (
            self.start_iter < len(batch_slice)
        ), f"starting iteration {self.start_iter} must be less than size of the slice of batches! {len(batch_slice)}"
        return iter(batch_slice[self.start_iter :])

    def set_epoch_and_start_iteration(self, epoch: int, start_iter: int) -> None:
        self.epoch = epoch
        self.start_iter = start_iter
