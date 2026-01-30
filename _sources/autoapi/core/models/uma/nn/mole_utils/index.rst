core.models.uma.nn.mole_utils
=============================

.. py:module:: core.models.uma.nn.mole_utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.models.uma.nn.mole_utils.fairchem_cpp_found
   core.models.uma.nn.mole_utils.fairchem_cpp_found


Classes
-------

.. autoapisummary::

   core.models.uma.nn.mole_utils.MOLEInterface


Functions
---------

.. autoapisummary::

   core.models.uma.nn.mole_utils.recursive_replace_so2m0_linear
   core.models.uma.nn.mole_utils.recursive_replace_so2_MOLE
   core.models.uma.nn.mole_utils.recursive_replace_so2_linear
   core.models.uma.nn.mole_utils.recursive_replace_all_linear
   core.models.uma.nn.mole_utils.recursive_replace_notso2_linear
   core.models.uma.nn.mole_utils.model_search_and_replace
   core.models.uma.nn.mole_utils.replace_linear_with_shared_linear
   core.models.uma.nn.mole_utils.replace_MOLE_with_linear
   core.models.uma.nn.mole_utils.replace_linear_with_MOLE
   core.models.uma.nn.mole_utils.convert_model_to_MOLE_model


Module Contents
---------------

.. py:data:: fairchem_cpp_found
   :value: False


.. py:data:: fairchem_cpp_found
   :value: True


.. py:class:: MOLEInterface

   .. py:method:: set_MOLE_coefficients(atomic_numbers_full, batch_full, csd_mixed_emb) -> None


   .. py:method:: set_MOLE_sizes(nsystems, batch_full, edge_index) -> None


   .. py:method:: log_MOLE_stats() -> None


   .. py:method:: merge_MOLE_model(data)


.. py:function:: recursive_replace_so2m0_linear(model, replacement_factory)

.. py:function:: recursive_replace_so2_MOLE(model, replacement_factory)

.. py:function:: recursive_replace_so2_linear(model, replacement_factory)

.. py:function:: recursive_replace_all_linear(model, replacement_factory)

.. py:function:: recursive_replace_notso2_linear(model, replacement_factory)

.. py:function:: model_search_and_replace(model, module_search_function, replacement_factory, layers=None)

.. py:function:: replace_linear_with_shared_linear(existing_linear_module, cache)

.. py:function:: replace_MOLE_with_linear(existing_mole_module: fairchem.core.models.uma.nn.mole.MOLE)

.. py:function:: replace_linear_with_MOLE(existing_linear_module, global_mole_tensors, num_experts, mole_layer_type, cache=None)

.. py:function:: convert_model_to_MOLE_model(model, num_experts: int = 8, mole_dropout: float = 0.0, mole_expert_coefficient_norm: str = 'softmax', act=torch.nn.SiLU, layers_mole=None, use_composition_embedding: bool = False, mole_layer_type: str = 'pytorch', mole_single: bool = False, mole_type: str = 'so2')

