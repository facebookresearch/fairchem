"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pymatgen.analysis.local_env import NearNeighbors
    from pymatgen.core import Structure


def construct_bond_matrix(
    structure: Structure,
    nn_finder: NearNeighbors,
    site_permutations: ArrayLike | None = None,
) -> np.ndarray:
    """
    Constructs a bond matrix for a given crystal structure.

    This function uses a near neigbor algorithm from pymatgen to determine bonds between atoms in the structure and
    creates an adjacency matrix where 1 indicates a bond between atoms and 0 indicates no bond.

    Args:
        structure: A pymatgen Structure object representing the crystal structure
        nn_finder: A NearNeighbors instance to determine near neighbor lists
        site_permutations: A numpy array containing the site permutations if the covalent matrix should be constructed
            with sites in an order different than the order in which they appear in the given structure

    Returns:
        np.ndarray: A square matrix where matrix[i,j] = 1 if atoms i and j share a covalent bond,
            and 0 otherwise
    """
    site_permutations = (
        site_permutations
        if site_permutations is not None
        else np.arange(len(structure), dtype=int)
    )

    nn_info = nn_finder.get_all_nn_info(structure)
    nn_matrix = np.zeros((len(nn_info), len(nn_info)), dtype=int)
    for i, ii in enumerate(site_permutations):
        for j in range(len(nn_info[i])):
            nn_matrix[ii, site_permutations[nn_info[i][j]["site_index"]]] = 1
    return nn_matrix
