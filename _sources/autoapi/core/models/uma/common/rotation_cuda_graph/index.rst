core.models.uma.common.rotation_cuda_graph
==========================================

.. py:module:: core.models.uma.common.rotation_cuda_graph

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.models.uma.common.rotation_cuda_graph.YTOL


Classes
-------

.. autoapisummary::

   core.models.uma.common.rotation_cuda_graph.RotMatWignerCudaGraph


Functions
---------

.. autoapisummary::

   core.models.uma.common.rotation_cuda_graph.capture_rotmat_and_wigner_with_make_graph_callable
   core.models.uma.common.rotation_cuda_graph.edge_rot_and_wigner_graph_capture_region
   core.models.uma.common.rotation_cuda_graph.create_masks
   core.models.uma.common.rotation_cuda_graph.init_edge_rot_mat_cuda_graph
   core.models.uma.common.rotation_cuda_graph.euler_from_edge_rot_mat
   core.models.uma.common.rotation_cuda_graph.eulers_to_wigner


Module Contents
---------------

.. py:data:: YTOL
   :value: 0.999999


.. py:class:: RotMatWignerCudaGraph

   .. py:attribute:: graph_mod
      :value: None



   .. py:attribute:: graph_capture_count
      :value: 0



   .. py:attribute:: max_edge_size
      :value: None



   .. py:method:: _capture_graph(edge_dist_vec: torch.Tensor, jds: list[torch.Tensor])


   .. py:method:: get_rotmat_and_wigner(edge_dist_vec: torch.Tensor, jds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]


.. py:function:: capture_rotmat_and_wigner_with_make_graph_callable(edge_dist_vec: torch.Tensor, jds: list[torch.Tensor])

.. py:function:: edge_rot_and_wigner_graph_capture_region(edge_distance_vecs: torch.Tensor, Jd_buffers: list[torch.Tensor], x_hat: torch.Tensor, mask: torch.Tensor, neg_mask: torch.Tensor)

.. py:function:: create_masks(edge_distance_vec: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]

.. py:function:: init_edge_rot_mat_cuda_graph(edge_distance_vec: torch.Tensor, mask: torch.Tensor, neg_mask: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor

.. py:function:: euler_from_edge_rot_mat(edge_rot_mat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

.. py:function:: eulers_to_wigner(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, start_lmax: int, end_lmax: int, Jd: list[torch.Tensor]) -> torch.Tensor

