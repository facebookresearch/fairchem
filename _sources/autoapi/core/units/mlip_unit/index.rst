core.units.mlip_unit
====================

.. py:module:: core.units.mlip_unit

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the LICENSE
   file in the root directory of this source tree.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/units/mlip_unit/_metrics/index
   /autoapi/core/units/mlip_unit/api/index
   /autoapi/core/units/mlip_unit/mlip_unit/index
   /autoapi/core/units/mlip_unit/predict/index
   /autoapi/core/units/mlip_unit/utils/index


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.load_predict_unit


Package Contents
----------------

.. py:function:: load_predict_unit(path: str | pathlib.Path, inference_settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings | str = 'default', overrides: dict | None = None, device: Literal['cuda', 'cpu'] | None = None, atom_refs: dict | None = None) -> fairchem.core.units.mlip_unit.predict.MLIPPredictUnit

   Load a MLIPPredictUnit from a checkpoint file.

   :param path: Path to the checkpoint file
   :param inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
                              (optimized for speed but requires fixed atomic composition). Advanced use cases can
                              use a custom InferenceSettings object.
   :param overrides: Optional dictionary of settings to override default inference settings.
   :param device: Optional torch device to load the model onto.
   :param atom_refs: Optional dictionary of isolated atom reference energies.

   :returns: A MLIPPredictUnit instance ready for inference


