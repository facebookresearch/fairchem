core.units.mlip_unit.api.inference
==================================

.. py:module:: core.units.mlip_unit.api.inference

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the LICENSE
   file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.units.mlip_unit.api.inference.CHARGE_RANGE
   core.units.mlip_unit.api.inference.SPIN_RANGE
   core.units.mlip_unit.api.inference.DEFAULT_CHARGE
   core.units.mlip_unit.api.inference.DEFAULT_SPIN_OMOL
   core.units.mlip_unit.api.inference.DEFAULT_SPIN
   core.units.mlip_unit.api.inference.NAME_TO_INFERENCE_SETTING


Classes
-------

.. autoapisummary::

   core.units.mlip_unit.api.inference.UMATask
   core.units.mlip_unit.api.inference.MLIPInferenceCheckpoint
   core.units.mlip_unit.api.inference.InferenceSettings


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.api.inference.inference_settings_default
   core.units.mlip_unit.api.inference.inference_settings_turbo
   core.units.mlip_unit.api.inference.inference_settings_traineval
   core.units.mlip_unit.api.inference.guess_inference_settings


Module Contents
---------------

.. py:class:: UMATask

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.


   .. py:attribute:: OMOL
      :value: 'omol'



   .. py:attribute:: OMAT
      :value: 'omat'



   .. py:attribute:: ODAC
      :value: 'odac'



   .. py:attribute:: OC20
      :value: 'oc20'



   .. py:attribute:: OMC
      :value: 'omc'



.. py:data:: CHARGE_RANGE

.. py:data:: SPIN_RANGE
   :value: [0, 100]


.. py:data:: DEFAULT_CHARGE
   :value: 0


.. py:data:: DEFAULT_SPIN_OMOL
   :value: 1


.. py:data:: DEFAULT_SPIN
   :value: 0


.. py:class:: MLIPInferenceCheckpoint

   .. py:attribute:: model_config
      :type:  dict


   .. py:attribute:: model_state_dict
      :type:  dict


   .. py:attribute:: ema_state_dict
      :type:  dict


   .. py:attribute:: tasks_config
      :type:  dict


.. py:class:: InferenceSettings

   .. py:attribute:: tf32
      :type:  bool
      :value: False



   .. py:attribute:: activation_checkpointing
      :type:  bool | None
      :value: None



   .. py:attribute:: merge_mole
      :type:  bool
      :value: False



   .. py:attribute:: compile
      :type:  bool
      :value: False



   .. py:attribute:: wigner_cuda
      :type:  bool | None
      :value: None



   .. py:attribute:: external_graph_gen
      :type:  bool | None
      :value: None



   .. py:attribute:: internal_graph_gen_version
      :type:  int | None
      :value: None



.. py:function:: inference_settings_default()

.. py:function:: inference_settings_turbo()

.. py:function:: inference_settings_traineval()

.. py:data:: NAME_TO_INFERENCE_SETTING

.. py:function:: guess_inference_settings(settings: str | InferenceSettings)

