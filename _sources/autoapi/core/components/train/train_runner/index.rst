core.components.train.train_runner
==================================

.. py:module:: core.components.train.train_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.train.train_runner.TrainCheckpointCallback
   core.components.train.train_runner.TrainEvalRunner


Functions
---------

.. autoapisummary::

   core.components.train.train_runner.get_most_recent_viable_checkpoint_path


Module Contents
---------------

.. py:function:: get_most_recent_viable_checkpoint_path(checkpoint_dir: str | None) -> str | None

.. py:class:: TrainCheckpointCallback(checkpoint_every_n_steps: int, max_saved_checkpoints: int = 2)

   Bases: :py:obj:`torchtnt.framework.callback.Callback`


   A Callback is an optional extension that can be used to supplement your loop with additional functionality. Good candidates
   for such logic are ones that can be re-used across units. Callbacks are generally not intended for modeling code; this should go
   in your `Unit <https://www.internalfb.com/intern/staticdocs/torchtnt/framework/unit.html>`_. To write your own callback,
   subclass the Callback class and add your own code into the hooks.

   Below is an example of a basic callback which prints a message at various points during execution.

   .. code-block:: python

     from torchtnt.framework.callback import Callback
     from torchtnt.framework.state import State
     from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit

     class PrintingCallback(Callback):
         def on_train_start(self, state: State, unit: TTrainUnit) -> None:
             print("Starting training")

         def on_train_end(self, state: State, unit: TTrainUnit) -> None:
             print("Ending training")

         def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
             print("Starting evaluation")

         def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
             print("Ending evaluation")

         def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
             print("Starting prediction")

         def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
             print("Ending prediction")

   To use a callback, instantiate the class and pass it in the ``callbacks`` parameter to the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`,
   :py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point.

   .. code-block:: python

     printing_callback = PrintingCallback()
     train(train_unit, train_dataloader, callbacks=[printing_callback])


   .. py:attribute:: checkpoint_every_n_steps


   .. py:attribute:: max_saved_checkpoints


   .. py:attribute:: save_callback
      :value: None



   .. py:attribute:: load_callback
      :value: None



   .. py:attribute:: checkpoint_dir
      :value: None



   .. py:method:: set_runner_callbacks(save_callback: callable, load_callback: callable, checkpoint_dir: str) -> None


   .. py:method:: on_train_step_start(state: torchtnt.framework.state.State, unit: torchtnt.framework.unit.TTrainUnit) -> None

      Hook called before a new train step starts.



   .. py:method:: on_train_end(state: torchtnt.framework.state.State, unit: torchtnt.framework.unit.TTrainUnit) -> None

      Hook called after training ends.



.. py:class:: TrainEvalRunner(train_dataloader: torch.utils.data.dataloader, eval_dataloader: torch.utils.data.dataloader, train_eval_unit: Union[torchtnt.framework.TrainUnit, torchtnt.framework.EvalUnit, torch.distributed.checkpoint.stateful.Stateful], callbacks: list[torchtnt.framework.callback.Callback] | None = None, max_epochs: int | None = 1, evaluate_every_n_steps: Optional[int] = None, max_steps: int | None = None, save_inference_ckpt: bool = True)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: train_dataloader


   .. py:attribute:: eval_dataloader


   .. py:attribute:: train_eval_unit


   .. py:attribute:: callbacks


   .. py:attribute:: max_epochs


   .. py:attribute:: max_steps


   .. py:attribute:: evaluate_every_n_steps


   .. py:attribute:: save_inference_ckpt


   .. py:attribute:: checkpoint_callback


   .. py:method:: run() -> None


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


