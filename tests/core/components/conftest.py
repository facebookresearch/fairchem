"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from itertools import product
from random import choice

import numpy as np
import pytest
from ase.db import connect
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from fairchem.core.datasets import AseDBDataset

# these are copied verbatim from https://github.com/facebookresearch/fairchem/blob/main/tests/core/conftest.py


@pytest.fixture(scope="session")
def dummy_element_refs():
    # create some dummy elemental energies from ionic radii (ignore deuterium and tritium included in pmg)
    return np.concatenate(
        [[0], [e.average_ionic_radius for e in Element if e.name not in ("D", "T")]]
    )


@pytest.fixture(scope="session")
def dummy_binary_dataset_path(tmpdir_factory, dummy_element_refs):
    # a dummy dataset with binaries with energy that depends on composition only plus noise
    all_binaries = list(product(list(Element), repeat=2))
    rng = np.random.default_rng(seed=0)

    tmpdir = tmpdir_factory.mktemp("dataset")
    with connect(str(tmpdir / "dummy.aselmdb")) as db:
        for i in range(1000):
            elements = choice(all_binaries)
            structure = Structure.from_prototype("cscl", species=elements, a=2.0)
            energy = (
                sum(e.average_ionic_radius for e in elements)
                + 0.05 * rng.random() * dummy_element_refs.mean()
            )
            atoms = structure.to_ase_atoms()
            db.write(
                atoms,
                data={
                    "sid": i,
                    "energy": energy,
                    "forces": rng.random((2, 3)),
                    "stress": rng.random((3, 3)),
                },
            )

    return tmpdir / "dummy.aselmdb"


@pytest.fixture(scope="session")
def dummy_binary_dataset(dummy_binary_dataset_path):
    return AseDBDataset(
        config={
            "src": str(dummy_binary_dataset_path),
            "a2g_args": {"r_data_keys": ["energy", "forces", "stress"]},
        }
    )
