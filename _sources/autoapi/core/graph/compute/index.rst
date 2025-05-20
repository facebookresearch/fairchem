core.graph.compute
==================

.. py:module:: core.graph.compute

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.graph.compute.get_pbc_distances
   core.graph.compute.generate_graph


Module Contents
---------------

.. py:function:: get_pbc_distances(pos, edge_index, cell, cell_offsets, neighbors, return_offsets: bool = False, return_distance_vec: bool = False)

.. py:function:: generate_graph(data: dict, cutoff: float, max_neighbors: int, enforce_max_neighbors_strictly: bool, radius_pbc_version: int, pbc: torch.Tensor) -> dict

   Generate a graph representation from atomic structure data.

   :param data: A dictionary containing a batch of molecular structures.
                It should have the following keys:
                    - 'pos' (torch.Tensor): Positions of the atoms.
                    - 'cell' (torch.Tensor): Cell vectors of the molecular structures.
                    - 'natoms' (torch.Tensor): Number of atoms in each molecular structure.
   :type data: dict
   :param cutoff: The maximum distance between atoms to consider them as neighbors.
   :type cutoff: float
   :param max_neighbors: The maximum number of neighbors to consider for each atom.
   :type max_neighbors: int
   :param enforce_max_neighbors_strictly: Whether to strictly enforce the maximum number of neighbors.
   :type enforce_max_neighbors_strictly: bool
   :param radius_pbc_version: the version of radius_pbc impl
   :param pbc: The periodic boundary conditions in 3 dimensions, defaults to [True,True,True] for 3D pbc
   :type pbc: list[bool]

   :returns:

             A dictionary containing the generated graph with the following keys:
                 - 'edge_index' (torch.Tensor): Indices of the edges in the graph.
                 - 'edge_distance' (torch.Tensor): Distances between the atoms connected by the edges.
                 - 'edge_distance_vec' (torch.Tensor): Vectors representing the distances between the atoms connected by the edges.
                 - 'cell_offsets' (torch.Tensor): Offsets of the cell vectors for each edge.
                 - 'offset_distances' (torch.Tensor): Distances between the atoms connected by the edges, including the cell offsets.
                 - 'neighbors' (torch.Tensor): Number of neighbors for each atom.
   :rtype: dict


