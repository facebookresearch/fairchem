"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import ase
import e3nn
import numba
import numpy as np
import psutil
import pymatgen.core as pc
import torch
import torch.cuda as tc
import torch_geometric as tg
from ase.db import connect

import fairchem.core as om


def describe_fairchem():
    """Print some system information that could be useful in debugging."""
    print(sys.executable, sys.version)

    commit_hash = (
        subprocess.check_output(
            [
                "git",
                "-C",
                om.__path__[0],
                "describe",
                "--always",
            ]
        )
        .strip()
        .decode("ascii")
    )
    print(f"fairchem repo is at git commit: {commit_hash}")
    print(f"numba: {numba.__version__}")
    print(f"numpy: {np.version.version}")
    print(f"ase: {ase.__version__}")
    print(f"e3nn: {e3nn.__version__}")
    print(f"pymatgen: {pc.__version__}")
    print(f"torch: {torch.version.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda: is_available: {tc.is_available()}")
    if tc.is_available():
        print("  __CUDNN VERSION:", torch.backends.cudnn.version())
        print("  __Number CUDA Devices:", torch.cuda.device_count())
        print("  __CUDA Device Name:", torch.cuda.get_device_name(0))
        print(
            "  __CUDA Device Total Memory [GB]:",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    print(f"torch geometric: {tg.__version__}")
    print()
    print(f"Platform: {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Virtual memory: {psutil.virtual_memory()}")
    print(f"  Swap memory: {psutil.swap_memory()}")
    print(f'  Disk usage: {psutil.disk_usage("/")}')


def train_test_val_split(
    ase_db, ttv=(0.8, 0.1, 0.1), files=("train.db", "test.db", "val.db"), seed=42
):
    """Split an ase db into train, test and validation dbs.

    ase_db: path to an ase db containing all the data.
    ttv: a tuple containing the fraction of train, test and val data. This will be normalized.
    files: a tuple of filenames to write the splits into. An exception is raised if these exist.
           You should delete them first.
    seed: an integer for the random number generator seed

    Returns the absolute path to files.
    """

    for db in files:
        if os.path.exists(db):
            raise Exception("{db} exists. Please delete it before proceeding.")

    src = connect(ase_db)
    N = src.count()

    ttv = np.array(ttv)
    ttv /= ttv.sum()

    train_end = int(N * ttv[0])
    test_end = train_end + int(N * ttv[1])

    train = connect(files[0])
    test = connect(files[1])
    val = connect(files[2])

    ids = np.arange(1, N + 1)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(ids)

    for _id in ids[0:train_end]:
        row = src.get(id=int(_id))
        # set add_additional_information to ensure row.data is added to atoms.info
        train.write(row.toatoms(add_additional_information=True))

    for _id in ids[train_end:test_end]:
        row = src.get(id=int(_id))
        # set add_additional_information to ensure row.data is added to atoms.info
        test.write(row.toatoms(add_additional_information=True))

    for _id in ids[test_end:]:
        row = src.get(id=int(_id))
        # set add_additional_information to ensure row.data is added to atoms.info
        val.write(row.toatoms(add_additional_information=True))

    return [Path(f).absolute() for f in files]
