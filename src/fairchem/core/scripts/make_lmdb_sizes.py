"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This script provides the functionality to generate metadata.npz files necessary
for load_balancing the DataLoader.

"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random

import numpy as np
from tqdm import tqdm

from fairchem.core.common.typing import assert_is_instance
from fairchem.core.datasets import AseDBDataset


def get_data(index):
    data = dataset[index]
    natoms = data.natoms

    return index, natoms


def make_lmdb_sizes(args) -> None:
    path = assert_is_instance(args.data_path, str)
    global dataset
    dataset = AseDBDataset({"src": path})
    if os.path.isdir(path):
        outpath = os.path.join(path, "metadata.npz")
    elif os.path.isfile(path):
        outpath = os.path.join(os.path.dirname(path), "metadata.npz")

    output_indices = range(len(dataset))

    pool = mp.Pool(assert_is_instance(args.num_workers, int))
    outputs = list(tqdm(pool.imap(get_data, output_indices), total=len(output_indices)))

    indices = []
    natoms = []
    for i in outputs:
        indices.append(i[0])
        natoms.append(i[1])

    _sort = np.argsort(indices)
    sorted_natoms = np.array(natoms)[_sort].flatten()

    sample_indices = random.sample(range(len(dataset)), 2000)
    for idx in tqdm(sample_indices, desc="Verifying metadata consistency"):
        assert natoms[idx] == len(dataset.get_atoms(idx))
    np.savez(outpath, natoms=sorted_natoms)


def get_lmdb_sizes_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to S2EF directory or IS2R* .lmdb file",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Num of workers to parallelize across",
    )
    return parser


if __name__ == "__main__":
    parser = get_lmdb_sizes_parser()
    args: argparse.Namespace = parser.parse_args()
    make_lmdb_sizes(args)
