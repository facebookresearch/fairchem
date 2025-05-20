core.units.mlip_unit._metrics
=============================

.. py:module:: core.units.mlip_unit._metrics

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.units.mlip_unit._metrics.NONE_SLICE


Classes
-------

.. autoapisummary::

   core.units.mlip_unit._metrics.Metrics


Functions
---------

.. autoapisummary::

   core.units.mlip_unit._metrics.metrics_dict
   core.units.mlip_unit._metrics.cosine_similarity
   core.units.mlip_unit._metrics.mae
   core.units.mlip_unit._metrics.mse
   core.units.mlip_unit._metrics.rmse
   core.units.mlip_unit._metrics.per_atom_mae
   core.units.mlip_unit._metrics.per_atom_mse
   core.units.mlip_unit._metrics.magnitude_error
   core.units.mlip_unit._metrics.forcesx_mae
   core.units.mlip_unit._metrics.forcesx_mse
   core.units.mlip_unit._metrics.forcesy_mae
   core.units.mlip_unit._metrics.forcesy_mse
   core.units.mlip_unit._metrics.forcesz_mae
   core.units.mlip_unit._metrics.forcesz_mse
   core.units.mlip_unit._metrics.energy_forces_within_threshold
   core.units.mlip_unit._metrics.energy_within_threshold
   core.units.mlip_unit._metrics.average_distance_within_threshold
   core.units.mlip_unit._metrics.min_diff
   core.units.mlip_unit._metrics.get_metrics_fn


Module Contents
---------------

.. py:data:: NONE_SLICE

.. py:class:: Metrics

   .. py:attribute:: metric
      :type:  float
      :value: 0.0



   .. py:attribute:: total
      :type:  float
      :value: 0.0



   .. py:attribute:: numel
      :type:  int
      :value: 0



   .. py:method:: __iadd__(other)


.. py:function:: metrics_dict(metric_fun: Callable) -> Callable

   Wrap up the return of a metrics function


.. py:function:: cosine_similarity(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE)

.. py:function:: mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: rmse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> torch.Tensor

.. py:function:: per_atom_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: per_atom_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> torch.Tensor

.. py:function:: magnitude_error(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE, p: int = 2) -> torch.Tensor

.. py:function:: forcesx_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> Metrics

.. py:function:: forcesx_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = NONE_SLICE) -> Metrics

.. py:function:: forcesy_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: forcesy_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: forcesz_mae(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: forcesz_mse(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: energy_forces_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: energy_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: average_distance_within_threshold(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], key: collections.abc.Hashable = None) -> Metrics

.. py:function:: min_diff(pred_pos: torch.Tensor, dft_pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor)

   Calculate the minimum difference between predicted and target positions considering periodic boundary conditions.


.. py:function:: get_metrics_fn(function_name: str) -> Callable

