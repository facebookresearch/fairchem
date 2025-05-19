"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from ase import build


@pytest.fixture()
def atoms_list():
    atoms_list = [
        build.bulk("Cu", "fcc", a=3.8, cubic=True),
        build.bulk("NaCl", crystalstructure="rocksalt", a=5.8),
    ]
    for atoms in atoms_list:
        atoms.rattle(stdev=0.05, seed=0)
    return atoms_list
