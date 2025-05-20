core.graph.radius_graph_pbc
===========================

.. py:module:: core.graph.radius_graph_pbc

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.graph.radius_graph_pbc.sum_partitions
   core.graph.radius_graph_pbc.get_counts
   core.graph.radius_graph_pbc.compute_neighbors
   core.graph.radius_graph_pbc.get_max_neighbors_mask
   core.graph.radius_graph_pbc.radius_graph_pbc
   core.graph.radius_graph_pbc.canonical_pbc
   core.graph.radius_graph_pbc.radius_graph_pbc_v2


Module Contents
---------------

.. py:function:: sum_partitions(x: torch.Tensor, partition_idxs: torch.Tensor) -> torch.Tensor

.. py:function:: get_counts(x: torch.Tensor, length: int)

.. py:function:: compute_neighbors(data, edge_index)

.. py:function:: get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold, degeneracy_tolerance: float = 0.01, enforce_max_strictly: bool = False)

   Give a mask that filters out edges so that each atom has at most
   `max_num_neighbors_threshold` neighbors.
   Assumes that `index` is sorted.

   Enforcing the max strictly can force the arbitrary choice between
   degenerate edges. This can lead to undesired behaviors; for
   example, bulk formation energies which are not invariant to
   unit cell choice.

   A degeneracy tolerance can help prevent sudden changes in edge
   existence from small changes in atom position, for example,
   rounding errors, slab relaxation, temperature, etc.


.. py:function:: radius_graph_pbc(data, radius, max_num_neighbors_threshold, enforce_max_neighbors_strictly: bool = False, pbc: torch.Tensor | None = None)

.. py:function:: canonical_pbc(data, pbc: torch.Tensor | None)

.. py:function:: radius_graph_pbc_v2(data, radius, max_num_neighbors_threshold, enforce_max_neighbors_strictly: bool = False, pbc: torch.Tensor | None = None)

