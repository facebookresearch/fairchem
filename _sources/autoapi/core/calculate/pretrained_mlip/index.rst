core.calculate.pretrained_mlip
==============================

.. py:module:: core.calculate.pretrained_mlip

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.calculate.pretrained_mlip._MODEL_CKPTS
   core.calculate.pretrained_mlip.available_models


Classes
-------

.. autoapisummary::

   core.calculate.pretrained_mlip.HuggingFaceCheckpoint
   core.calculate.pretrained_mlip.PretrainedModels


Functions
---------

.. autoapisummary::

   core.calculate.pretrained_mlip.get_predict_unit


Module Contents
---------------

.. py:class:: HuggingFaceCheckpoint

   .. py:attribute:: filename
      :type:  str


   .. py:attribute:: repo_id
      :type:  Literal['facebook/UMA']


   .. py:attribute:: subfolder
      :type:  str | None
      :value: None



   .. py:attribute:: revision
      :type:  str | None
      :value: None



.. py:class:: PretrainedModels

   .. py:attribute:: checkpoints
      :type:  dict[str, HuggingFaceCheckpoint]


.. py:data:: _MODEL_CKPTS

.. py:data:: available_models

.. py:function:: get_predict_unit(model_name: str, inference_settings: fairchem.core.units.mlip_unit.InferenceSettings | str = 'default', overrides: dict | None = None, device: Literal['cuda', 'cpu'] | None = None) -> fairchem.core.units.mlip_unit.MLIPPredictUnit

   Retrieves a prediction unit for a specified model.

   :param model_name: Name of the model to load from available pretrained models.
   :param inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
                              (optimized for speed but requires fixed atomic composition). Advanced use cases can
                              use a custom InferenceSettings object.
   :param overrides: Optional dictionary of settings to override default inference settings.
   :param device: Optional torch device to load the model onto. If None, uses the default device.

   :returns: An initialized MLIPPredictUnit ready for making predictions.

   :raises KeyError: If the specified model_name is not found in available models.


