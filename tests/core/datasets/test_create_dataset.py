"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase.db import connect

from fairchem.core.datasets import create_dataset
from fairchem.core.datasets.base_dataset import BaseDataset


@pytest.fixture()
def lmdb_database(structures):
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_atoms = []
        mod2 = []
        mod3 = []
        asedb_fn = f"{tmpdirname}/asedb.aselmdb"
        with connect(asedb_fn) as database:
            for _, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)
                num_atoms.append(len(atoms))
                mod2.append(len(atoms) % 2)
                mod3.append(len(atoms) % 3)
        np.savez(
            f"{tmpdirname}/metadata.npz",
            natoms=num_atoms,
            mod2=mod2,
            mod3=mod3,
        )
        yield asedb_fn


def test_subset_to(structures, lmdb_database):
    # select all
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        "subset_to": [{"op": "abs_le", "metadata_key": "mod2", "rhv": 10}],
    }

    assert len(create_dataset(config, split="train")) == len(structures)

    # select a subset of indices fed as list
    indices = [1, 2]
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        "subset_to": [{"op": "in", "metadata_key": "mod2", "rhv": indices}],
    }

    assert len(create_dataset(config, split="train")) == len(indices)

    # select a subset of indices fed as path
    with tempfile.TemporaryDirectory() as tmpdirname:
        indices_path = f"{tmpdirname}/indices.txt"
        indices = [1, 2]
        with open(indices_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")

        config = {
            "format": "ase_db",
            "src": str(lmdb_database),
            "subset_to": [{"op": "in", "metadata_key": "mod2", "rhv": indices_path}],
        }

        assert len(create_dataset(config, split="train")) == len(indices)

    # only select those that have mod2==0
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        "subset_to": [{"op": "abs_le", "metadata_key": "mod2", "rhv": 0}],
    }
    assert len(create_dataset(config, split="train")) == len(
        [s for s in structures if len(s) % 2 == 0]
    )

    # only select those that have mod2==0 and mod3==0
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        "subset_to": [
            {"op": "abs_le", "metadata_key": "mod2", "rhv": 0},
            {"op": "abs_le", "metadata_key": "mod2", "rhv": 0},
        ],
    }
    assert len(create_dataset(config, split="train")) == len(
        [s for s in structures if len(s) % 2 == 0]
    )
    assert len([s for s in structures if len(s) % 2 == 0]) > 0


@pytest.mark.parametrize("max_atoms", [3, None])
@pytest.mark.parametrize(
    "key, value", [("first_n", 2), ("sample_n", 2), ("no_shuffle", True)]
)
def test_create_dataset(key, value, max_atoms, structures, lmdb_database):
    # now create a config
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        key: value,
        "max_atoms": max_atoms,
    }

    dataset = create_dataset(config, split="train")
    if max_atoms is not None:
        structures = [s for s in structures if len(s) <= max_atoms]
        assert all(
            natoms <= max_atoms
            for natoms in dataset.metadata["natoms"][range(len(dataset))]
        )
    if key == "first_n":  # this assumes first_n are not shuffled
        assert all(
            np.allclose(a1.cell.array, a2.cell.numpy())
            for a1, a2 in zip(structures[:value], dataset)
        )
        assert all(
            np.allclose(a1.numbers, a2.atomic_numbers)
            for a1, a2 in zip(structures[:value], dataset)
        )
    elif key == "sample_n":
        assert len(dataset) == value
    else:  # no shuffle all of them are in there
        assert all(
            np.allclose(a1.cell.array, a2.cell.numpy())
            for a1, a2 in zip(structures, dataset)
        )
        assert all(
            np.allclose(a1.numbers, a2.atomic_numbers)
            for a1, a2 in zip(structures, dataset)
        )


# make sure we cant sample more than the number of elements in the dataset with sample_n
def test_sample_n_dataset(lmdb_database):
    with pytest.raises(ValueError):
        _ = create_dataset(
            config={
                "format": "ase_db",
                "src": str(lmdb_database),
                "sample_n": 100,
            },
            split="train",
        )


def test_diff_seed_sample_dataset(lmdb_database):
    dataset_a = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 0,
        },
        split="train",
    )
    dataset_b = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 0,
        },
        split="train",
    )
    assert (dataset_a.indices == dataset_b.indices).all()
    dataset_b = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 1,
        },
        split="train",
    )
    assert not (dataset_a.indices == dataset_b.indices).all()


def test_del_dataset():
    class _Dataset(BaseDataset):
        def __init__(self, fn) -> None:
            super().__init__(config={})
            self.fn = fn
            open(self.fn, "a").close()

        def __del__(self):
            os.remove(self.fn)

    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + "/test"
        d = _Dataset(fn)
        del d
        assert not os.path.exists(fn)
