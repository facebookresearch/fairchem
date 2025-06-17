core.units.mlip_unit.predict
============================

.. py:module:: core.units.mlip_unit.predict

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.units.mlip_unit.predict.MLIPPredictUnit


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.predict.collate_predictions
   core.units.mlip_unit.predict.get_dataset_to_tasks_map


Module Contents
---------------

.. py:function:: collate_predictions(predict_fn)

.. py:class:: MLIPPredictUnit(inference_model_path: str, device: str = 'cpu', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.InferenceSettings | None = None, seed: int = 41, atom_refs: dict | None = None)

   Bases: :py:obj:`torchtnt.framework.PredictUnit`\ [\ :py:obj:`fairchem.core.datasets.atomic_data.AtomicData`\ ]


   The PredictUnit is an interface that can be used to organize your prediction logic. The core of it is the ``predict_step`` which
   is an abstract method where you can define the code you want to run each iteration of the dataloader.

   To use the PredictUnit, create a class which subclasses :class:`~torchtnt.framework.unit.PredictUnit`.
   Then implement the ``predict_step`` method on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
   In addition, you can override ``get_next_predict_batch`` to modify the default batch fetching behavior.
   Below is a simple example of a user's subclass of :class:`~torchtnt.framework.unit.PredictUnit` that implements a basic ``predict_step``.

   .. code-block:: python

     from torchtnt.framework.unit import PredictUnit

     Batch = Tuple[torch.tensor, torch.tensor]
     # specify type of the data in each batch of the dataloader to allow for typechecking

     class MyPredictUnit(PredictUnit[Batch]):
         def __init__(
             self,
             module: torch.nn.Module,
         ):
             super().__init__()
             self.module = module

         def predict_step(self, state: State, data: Batch) -> torch.tensor:
             inputs, targets = data
             outputs = self.module(inputs)
             return outputs

     predict_unit = MyPredictUnit(module=...)


   .. py:attribute:: atom_refs


   .. py:attribute:: tasks


   .. py:attribute:: dataset_to_tasks


   .. py:attribute:: device


   .. py:attribute:: lazy_model_intialized
      :value: False



   .. py:attribute:: inference_mode


   .. py:attribute:: merged_on
      :value: None



   .. py:property:: direct_forces
      :type: bool



   .. py:property:: datasets
      :type: list[str]



   .. py:method:: seed(seed: int)


   .. py:method:: move_to_device()


   .. py:method:: predict_step(state: torchtnt.framework.State, data: fairchem.core.datasets.atomic_data.AtomicData) -> dict[str, torch.tensor]

      Core required method for user to implement. This method will be called at each iteration of the
      predict dataloader, and can return any data the user wishes.
      Optionally can be decorated with ``@torch.inference_mode()`` for improved performance.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the prediction run.
      :param data: one batch of prediction data.



   .. py:method:: get_composition_charge_spin_dataset(data)


   .. py:method:: predict(data: fairchem.core.datasets.atomic_data.AtomicData, undo_element_references: bool = True) -> dict[str, torch.tensor]


.. py:function:: get_dataset_to_tasks_map(tasks: Sequence[fairchem.core.units.mlip_unit.mlip_unit.Task]) -> dict[str, list[fairchem.core.units.mlip_unit.mlip_unit.Task]]

   Create a mapping from dataset names to their associated tasks.

   :param tasks: A sequence of Task objects to be organized by dataset

   :returns: A dictionary mapping dataset names (str) to lists of Task objects
             that are associated with that dataset


