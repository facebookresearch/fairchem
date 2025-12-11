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

   core.units.mlip_unit.predict.MLIPPredictUnitProtocol
   core.units.mlip_unit.predict.MLIPPredictUnit
   core.units.mlip_unit.predict.MLIPWorkerLocal
   core.units.mlip_unit.predict.MLIPWorker
   core.units.mlip_unit.predict.ParallelMLIPPredictUnit
   core.units.mlip_unit.predict.BatchServerPredictUnit


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.predict.collate_predictions
   core.units.mlip_unit.predict.merge_uma_model
   core.units.mlip_unit.predict.get_dataset_to_tasks_map
   core.units.mlip_unit.predict.move_tensors_to_cpu


Module Contents
---------------

.. py:function:: collate_predictions(predict_fn)

.. py:class:: MLIPPredictUnitProtocol

   Bases: :py:obj:`Protocol`


   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto[T](Protocol):
           def meth(self) -> T:
               ...


   .. py:method:: predict(data: fairchem.core.datasets.atomic_data.AtomicData, undo_element_references: bool) -> dict


   .. py:property:: dataset_to_tasks
      :type: dict[str, list]



.. py:function:: merge_uma_model(model, data)

.. py:class:: MLIPPredictUnit(inference_model_path: str, device: str = 'cpu', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.InferenceSettings | None = None, seed: int = 41, atom_refs: dict | None = None, form_elem_refs: dict | None = None, assert_on_nans: bool = False)

   Bases: :py:obj:`torchtnt.framework.PredictUnit`\ [\ :py:obj:`fairchem.core.datasets.atomic_data.AtomicData`\ ], :py:obj:`MLIPPredictUnitProtocol`


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


   .. py:attribute:: form_elem_refs


   .. py:attribute:: tasks


   .. py:attribute:: _dataset_to_tasks


   .. py:attribute:: device


   .. py:attribute:: lazy_model_intialized
      :value: False



   .. py:attribute:: inference_settings


   .. py:attribute:: merged_on
      :value: None



   .. py:attribute:: assert_on_nans


   .. py:property:: direct_forces
      :type: bool



   .. py:property:: dataset_to_tasks
      :type: dict[str, list]



   .. py:method:: set_seed(seed: int)


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


.. py:function:: move_tensors_to_cpu(data)

   Recursively move all PyTorch tensors in a nested data structure to CPU.

   :param data: Input data structure (dict, list, tuple, tensor, or other)

   :returns: Data structure with all tensors moved to CPU


.. py:class:: MLIPWorkerLocal(worker_id: int, world_size: int, predictor_config: dict, master_port: int | None = None, master_address: str | None = None)

   .. py:attribute:: worker_id


   .. py:attribute:: world_size


   .. py:attribute:: predictor_config


   .. py:attribute:: master_address


   .. py:attribute:: master_port


   .. py:attribute:: is_setup
      :value: False



   .. py:attribute:: last_received_atomic_data
      :value: None



   .. py:method:: get_master_address_and_port()


   .. py:method:: get_device_for_local_rank()


   .. py:method:: _distributed_setup()


   .. py:method:: predict(data: fairchem.core.datasets.atomic_data.AtomicData, use_nccl: bool = False) -> dict[str, torch.tensor] | None


.. py:class:: MLIPWorker(worker_id: int, world_size: int, predictor_config: dict, master_port: int | None = None, master_address: str | None = None)

   Bases: :py:obj:`MLIPWorkerLocal`


.. py:class:: ParallelMLIPPredictUnit(inference_model_path: str, device: str = 'cpu', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.InferenceSettings | None = None, seed: int = 41, atom_refs: dict | None = None, form_elem_refs: dict | None = None, assert_on_nans: bool = False, num_workers: int = 1, num_workers_per_node: int = 8, log_level: int = logging.INFO)

   Bases: :py:obj:`MLIPPredictUnitProtocol`


   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto[T](Protocol):
           def meth(self) -> T:
               ...


   .. py:attribute:: inference_settings


   .. py:attribute:: _dataset_to_tasks


   .. py:attribute:: atomic_data_on_device
      :value: None



   .. py:attribute:: workers
      :value: []



   .. py:attribute:: local_rank0


   .. py:method:: predict(data: fairchem.core.datasets.atomic_data.AtomicData) -> dict[str, torch.tensor]


   .. py:property:: dataset_to_tasks
      :type: dict[str, list]



.. py:class:: BatchServerPredictUnit(server_handle)

   Bases: :py:obj:`MLIPPredictUnitProtocol`


   PredictUnit wrapper that uses Ray Serve for batched inference.

   This provides a clean interface compatible with MLIPPredictUnitProtocol
   while leveraging Ray Serve's batching capabilities under the hood.


   .. py:attribute:: server_handle


   .. py:method:: predict(data: fairchem.core.datasets.atomic_data.AtomicData, undo_element_references: bool = True) -> dict

      :param data: AtomicData object (single system)
      :param undo_element_references: Whether to undo element references

      :returns: Prediction dictionary



   .. py:property:: dataset_to_tasks
      :type: dict



   .. py:property:: atom_refs
      :type: dict | None



   .. py:property:: inference_settings
      :type: fairchem.core.units.mlip_unit.InferenceSettings



