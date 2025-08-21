core.models.uma.common.rotation
===============================

.. py:module:: core.models.uma.common.rotation

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.models.uma.common.rotation.init_edge_rot_euler_angles
   core.models.uma.common.rotation.wigner_D
   core.models.uma.common.rotation._z_rot_mat
   core.models.uma.common.rotation.eulers_to_wigner


Module Contents
---------------

.. py:function:: init_edge_rot_euler_angles(edge_distance_vec)

.. py:function:: wigner_D(lv: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, _Jd: list[torch.Tensor]) -> torch.Tensor

.. py:function:: _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor

.. py:function:: eulers_to_wigner(eulers: torch.Tensor, start_lmax: int, end_lmax: int, Jd: list[torch.Tensor]) -> torch.Tensor

   set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.


