core.units.mlip_unit.mlip_unit
==============================

.. py:module:: core.units.mlip_unit.mlip_unit

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.units.mlip_unit.mlip_unit.UNIT_RESUME_CONFIG
   core.units.mlip_unit.mlip_unit.UNIT_INFERENCE_CHECKPOINT
   core.units.mlip_unit.mlip_unit.DEFAULT_EXCLUDE_KEYS


Classes
-------

.. autoapisummary::

   core.units.mlip_unit.mlip_unit.OutputSpec
   core.units.mlip_unit.mlip_unit.TrainStrategy
   core.units.mlip_unit.mlip_unit.Task
   core.units.mlip_unit.mlip_unit.MLIPTrainEvalUnit
   core.units.mlip_unit.mlip_unit.MLIPEvalUnit


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.mlip_unit.filter_inference_only_tasks
   core.units.mlip_unit.mlip_unit.convert_train_checkpoint_to_inference_checkpoint
   core.units.mlip_unit.mlip_unit.initialize_finetuning_model
   core.units.mlip_unit.mlip_unit.get_output_mask
   core.units.mlip_unit.mlip_unit.get_output_masks
   core.units.mlip_unit.mlip_unit.compute_loss
   core.units.mlip_unit.mlip_unit.compute_metrics
   core.units.mlip_unit.mlip_unit.mt_collater_adapter
   core.units.mlip_unit.mlip_unit._get_consine_lr_scheduler
   core.units.mlip_unit.mlip_unit._get_optimizer_wd
   core.units.mlip_unit.mlip_unit._reshard_fsdp
   core.units.mlip_unit.mlip_unit.set_sampler_state


Module Contents
---------------

.. py:data:: UNIT_RESUME_CONFIG
   :value: 'resume.yaml'


.. py:data:: UNIT_INFERENCE_CHECKPOINT
   :value: 'inference_ckpt.pt'


.. py:class:: OutputSpec

   .. py:attribute:: dim
      :type:  list[int]


   .. py:attribute:: dtype
      :type:  str


.. py:class:: TrainStrategy

   Bases: :py:obj:`fairchem.core.common.utils.StrEnum`


   Enum where members are also (and must be) strings


   .. py:attribute:: DDP
      :value: 'ddp'



   .. py:attribute:: FSDP
      :value: 'fsdp'



.. py:class:: Task

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: level
      :type:  str


   .. py:attribute:: property
      :type:  str


   .. py:attribute:: out_spec
      :type:  OutputSpec


   .. py:attribute:: normalizer
      :type:  fairchem.core.modules.normalization.normalizer.Normalizer


   .. py:attribute:: datasets
      :type:  list[str]


   .. py:attribute:: loss_fn
      :type:  torch.nn.Module | None
      :value: None



   .. py:attribute:: element_references
      :type:  Optional[fairchem.core.modules.normalization.element_references.ElementReferences]
      :value: None



   .. py:attribute:: metrics
      :type:  list[str]


   .. py:attribute:: train_on_free_atoms
      :type:  bool
      :value: True



   .. py:attribute:: eval_on_free_atoms
      :type:  bool
      :value: True



   .. py:attribute:: inference_only
      :type:  bool
      :value: False



.. py:data:: DEFAULT_EXCLUDE_KEYS
   :value: ['id', 'fid', 'absolute_idx', 'target_pos', 'ref_energy', 'pbc', 'nads', 'oc22',...


.. py:function:: filter_inference_only_tasks(tasks: Sequence[Task]) -> list[Task]

   Filter out tasks that are marked as inference_only.


.. py:function:: convert_train_checkpoint_to_inference_checkpoint(dcp_checkpoint_loc: str, checkpoint_loc: str) -> None

.. py:function:: initialize_finetuning_model(checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None, strict: bool = True) -> torch.nn.Module

.. py:function:: get_output_mask(batch: fairchem.core.datasets.atomic_data.AtomicData, task: Task) -> dict[str, torch.Tensor]

   Get a dictionary of boolean masks for each task and dataset in a batch.

   Comment(@abhshkdz): Structures in our `batch` are a mix from various
   sources, e.g. OC20, OC22, etc. That means for each loss computation,
   we need to pull out the attribute of interest from each structure.
   E.g. oc20_energy from OC20 structures, oc22_energy from OC22
   structures etc. Set up those mappings here. Supports two kinds for
   now: 1) for each structure-level output, mapping from output head
   to boolean indexing map for `out` and `batch`, s.t. we can index like
   batch.oc20_energy[oc20_map] for oc20_energy loss calculation. 2) for
   each atom-level output, a similar mapping from output head to boolean
   indexing map. s.t. we can index like batch.oc20_forces[oc20_map].


.. py:function:: get_output_masks(batch: fairchem.core.datasets.atomic_data.AtomicData, tasks: Sequence[Task]) -> dict[str, torch.Tensor]

   Same as above but for a list of tasks.


.. py:function:: compute_loss(tasks: Sequence[Task], predictions: dict[str, torch.Tensor], batch: fairchem.core.datasets.atomic_data.AtomicData) -> dict[str, float]

   Compute loss given a sequence of tasks

   :param tasks: a sequence of Task
   :param predictions: dictionary of predictions
   :param batch: data batch

   :returns: dictionary of losses for each task


.. py:function:: compute_metrics(task: Task, predictions: dict[str, torch.Tensor], batch: fairchem.core.datasets.atomic_data.AtomicData, dataset_name: str | None = None) -> dict[str:Metrics]

   Compute metrics and update running metrics for a given task

   :param task: a Task
   :param predictions: dictionary of predictions
   :param batch: data batch
   :param dataset_name: optional, if given compute metrics for given task using only labels from the given dataset
   :param running_metrics: optional dictionary of previous metrics to update.

   :returns: dictionary of (updated) metrics


.. py:function:: mt_collater_adapter(tasks: list[Task], exclude_keys: list[str] = DEFAULT_EXCLUDE_KEYS)

.. py:function:: _get_consine_lr_scheduler(warmup_factor: float, warmup_epochs: float, lr_min_factor: float, n_iters_per_epoch: int, optimizer: torch.optim.Optimizer, epochs: Optional[int] = None, steps: Optional[int] = None) -> torch.optim.lr_scheduler.LRScheduler

.. py:function:: _get_optimizer_wd(optimizer_fn: callable, model: torch.nn.Module) -> torch.optim.Optimizer

.. py:function:: _reshard_fsdp(model: torch.nn.Module) -> None

.. py:function:: set_sampler_state(state: torchtnt.framework.State, epoch: int, step_start: int) -> None

.. py:class:: MLIPTrainEvalUnit(job_config: omegaconf.DictConfig, model: torch.nn.Module, optimizer_fn: callable, cosine_lr_scheduler_fn: callable, tasks: list[Task], bf16: bool = False, print_every: int = 10, clip_grad_norm: float | None = None, ema_decay: float = 0.999, train_strategy: TrainStrategy = TrainStrategy.DDP, debug_checksums_save_path: str | None = None, profile_flops: bool = False, save_inference_ckpt: bool = True)

   Bases: :py:obj:`torchtnt.framework.TrainUnit`\ [\ :py:obj:`fairchem.core.datasets.atomic_data.AtomicData`\ ], :py:obj:`torchtnt.framework.EvalUnit`\ [\ :py:obj:`fairchem.core.datasets.atomic_data.AtomicData`\ ], :py:obj:`torch.distributed.checkpoint.stateful.Stateful`, :py:obj:`fairchem.core.components.train.train_runner.Checkpointable`


   The TrainUnit is an interface that can be used to organize your training logic. The core of it is the ``train_step`` which
   is an abstract method where you can define the code you want to run each iteration of the dataloader.

   To use the TrainUnit, create a class which subclasses TrainUnit. Then implement the ``train_step`` method on your class, and optionally
   implement any of the hooks, which allow you to control the behavior of the loop at different points.

   In addition, you can override ``get_next_train_batch`` to modify the default batch fetching behavior.

   Below is a simple example of a user's subclass of TrainUnit that implements a basic ``train_step``, and the ``on_train_epoch_end`` hook.

   .. code-block:: python

     from torchtnt.framework.unit import TrainUnit

     Batch = Tuple[torch.tensor, torch.tensor]
     # specify type of the data in each batch of the dataloader to allow for typechecking

     class MyTrainUnit(TrainUnit[Batch]):
         def __init__(
             self,
             module: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
         ):
             super().__init__()
             self.module = module
             self.optimizer = optimizer
             self.lr_scheduler = lr_scheduler

         def train_step(self, state: State, data: Batch) -> None:
             inputs, targets = data
             outputs = self.module(inputs)
             loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
             loss.backward()

             self.optimizer.step()
             self.optimizer.zero_grad()

         def on_train_epoch_end(self, state: State) -> None:
             # step the learning rate scheduler
             self.lr_scheduler.step()

     train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)


   .. py:attribute:: job_config


   .. py:attribute:: tasks


   .. py:attribute:: profile_flops


   .. py:attribute:: save_inference_ckpt


   .. py:attribute:: bf16


   .. py:attribute:: autocast_enabled


   .. py:attribute:: autocast_dtype


   .. py:attribute:: finetune_model_full_config


   .. py:attribute:: optimizer


   .. py:attribute:: logger


   .. py:attribute:: debug_checksums_save_path


   .. py:attribute:: print_every


   .. py:attribute:: clip_grad_norm


   .. py:attribute:: dp_world_size


   .. py:attribute:: num_params


   .. py:attribute:: ema_decay


   .. py:attribute:: ema_model
      :value: None



   .. py:attribute:: train_strategy


   .. py:attribute:: eval_unit


   .. py:attribute:: cosine_lr_scheduler_fn


   .. py:attribute:: scheduler
      :value: None



   .. py:attribute:: lazy_state_location
      :value: None



   .. py:method:: load_scheduler(train_dataloader_size: int) -> int


   .. py:method:: on_train_start(state: torchtnt.framework.State) -> None

      Hook called before training starts.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the training run.



   .. py:method:: on_train_epoch_start(state: torchtnt.framework.State) -> None

      Hook called before a train epoch starts.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the training run.



   .. py:method:: train_step(state: torchtnt.framework.State, data: fairchem.core.datasets.atomic_data.AtomicData) -> None

      Core required method for user to implement. This method will be called at each iteration of the
      train dataloader, and can return any data the user wishes.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the training run.
      :param data: one batch of training data.



   .. py:method:: on_train_end(state: torchtnt.framework.State) -> None

      Hook called after training ends.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the training run.



   .. py:method:: state_dict() -> dict[str, Any]

      Objects should return their state_dict representation as a dictionary.
      The output of this function will be checkpointed, and later restored in
      `load_state_dict()`.

      .. warning::
          Because of the inplace nature of restoring a checkpoint, this function
          is also called during `torch.distributed.checkpoint.load`.


      :returns: The objects state dict
      :rtype: Dict



   .. py:method:: load_state_dict(state_dict: dict[str, Any])

      Restore the object's state from the provided state_dict.

      :param state_dict: The state dict to restore from



   .. py:method:: eval_step(state: torchtnt.framework.State, data: fairchem.core.datasets.atomic_data.AtomicData) -> None

      Core required method for user to implement. This method will be called at each iteration of the
      eval dataloader, and can return any data the user wishes.
      Optionally can be decorated with ``@torch.inference_mode()`` for improved performance.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the evaluation run.
      :param data: one batch of evaluation data.



   .. py:method:: on_eval_epoch_start(state: torchtnt.framework.State) -> None

      Hook called before a new eval epoch starts.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the evaluation run.



   .. py:method:: on_eval_epoch_end(state: torchtnt.framework.State) -> None

      Hook called after an eval epoch ends.

      :param state: a :class:`~torchtnt.framework.state.State` object containing metadata about the evaluation run.



   .. py:method:: get_finetune_model_config() -> omegaconf.DictConfig | None


   .. py:method:: save_state(checkpoint_location: str) -> None

      Save the unit state to a checkpoint path

      :param checkpoint_location: The checkpoint path to save to



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Loads the state given a checkpoint path

      :param checkpoint_location: The checkpoint path to restore from



   .. py:method:: _execute_load_state(checkpoint_location: str | None) -> None


.. py:class:: MLIPEvalUnit(job_config: omegaconf.DictConfig, model: torch.nn.Module, tasks: Sequence[Task], bf16: bool = False)

   Bases: :py:obj:`torchtnt.framework.EvalUnit`\ [\ :py:obj:`fairchem.core.datasets.atomic_data.AtomicData`\ ]


   The EvalUnit is an interface that can be used to organize your evaluation logic. The core of it is the ``eval_step`` which
   is an abstract method where you can define the code you want to run each iteration of the dataloader.

   To use the EvalUnit, create a class which subclasses :class:`~torchtnt.framework.unit.EvalUnit`.
   Then implement the ``eval_step`` method on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
   In addition, you can override ``get_next_eval_batch`` to modify the default batch fetching behavior.
   Below is a simple example of a user's subclass of :class:`~torchtnt.framework.unit.EvalUnit` that implements a basic ``eval_step``.

   .. code-block:: python

     from torchtnt.framework.unit import EvalUnit

     Batch = Tuple[torch.tensor, torch.tensor]
     # specify type of the data in each batch of the dataloader to allow for typechecking

     class MyEvalUnit(EvalUnit[Batch]):
         def __init__(
             self,
             module: torch.nn.Module,
         ):
             super().__init__()
             self.module = module

         def eval_step(self, state: State, data: Batch) -> None:
             inputs, targets = data
             outputs = self.module(inputs)
             loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

     eval_unit = MyEvalUnit(module=...)


   .. py:attribute:: job_config


   .. py:attribute:: model


   .. py:attribute:: tasks


   .. py:attribute:: running_metrics
      :type:  dict[str, dict[str, dict[str, fairchem.core.units.mlip_unit._metrics.Metrics]]]


   .. py:attribute:: total_loss_metrics
      :type:  fairchem.core.units.mlip_unit._metrics.Metrics


   .. py:attribute:: total_atoms
      :type:  int
      :value: 0



   .. py:attribute:: total_runtime
      :type:  float
      :value: 0



   .. py:attribute:: logger


   .. py:attribute:: autocast_enabled


   .. py:attribute:: autocast_dtype


   .. py:method:: setup_train_eval_unit(model: torch.nn.Module) -> None


   .. py:method:: on_eval_epoch_start(state: torchtnt.framework.State) -> None

      Reset all metrics, and make sure model is in eval mode.



   .. py:method:: eval_step(state: torchtnt.framework.State, data: fairchem.core.datasets.atomic_data.AtomicData) -> None

      Evaluates the model on a batch of data.



   .. py:method:: on_eval_epoch_end(state: torchtnt.framework.State) -> dict

      Aggregate all metrics and log.



