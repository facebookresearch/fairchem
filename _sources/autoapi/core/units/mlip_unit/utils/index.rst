core.units.mlip_unit.utils
==========================

.. py:module:: core.units.mlip_unit.utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.units.mlip_unit.utils.load_inference_model
   core.units.mlip_unit.utils.tf32_context_manager
   core.units.mlip_unit.utils.update_configs


Module Contents
---------------

.. py:function:: load_inference_model(checkpoint_location: str, overrides: dict | None = None, use_ema: bool = False) -> tuple[torch.nn.Module, fairchem.core.units.mlip_unit.api.inference.MLIPInferenceCheckpoint]

.. py:function:: tf32_context_manager()

.. py:function:: update_configs(original_config, new_config)

