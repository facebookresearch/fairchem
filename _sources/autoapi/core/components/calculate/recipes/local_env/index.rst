core.components.calculate.recipes.local_env
===========================================

.. py:module:: core.components.calculate.recipes.local_env

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.local_env.construct_bond_matrix


Module Contents
---------------

.. py:function:: construct_bond_matrix(structure: pymatgen.core.Structure, nn_finder: pymatgen.analysis.local_env.NearNeighbors, site_permutations: numpy.typing.ArrayLike | None = None) -> numpy.ndarray

   Constructs a bond matrix for a given crystal structure.

   This function uses a near neigbor algorithm from pymatgen to determine bonds between atoms in the structure and
   creates an adjacency matrix where 1 indicates a bond between atoms and 0 indicates no bond.

   :param structure: A pymatgen Structure object representing the crystal structure
   :param nn_finder: A NearNeighbors instance to determine near neighbor lists
   :param site_permutations: A numpy array containing the site permutations if the covalent matrix should be constructed
                             with sites in an order different than the order in which they appear in the given structure

   :returns:

             A square matrix where matrix[i,j] = 1 if atoms i and j share a covalent bond,
                 and 0 otherwise
   :rtype: np.ndarray


