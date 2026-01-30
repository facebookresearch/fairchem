core.models.utils.irreps
========================

.. py:module:: core.models.utils.irreps

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.models.utils.irreps.cg_change_mat
   core.models.utils.irreps.irreps_sum


Module Contents
---------------

.. py:function:: cg_change_mat(ang_mom: int, device: str = 'cpu') -> torch.tensor

.. py:function:: irreps_sum(ang_mom: int) -> int

   Returns the sum of the dimensions of the irreps up to the specified angular momentum.

   :param ang_mom: max angular momenttum to sum up dimensions of irreps


