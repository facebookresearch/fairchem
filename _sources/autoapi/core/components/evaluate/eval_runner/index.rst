core.components.evaluate.eval_runner
====================================

.. py:module:: core.components.evaluate.eval_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.evaluate.eval_runner.EvalRunner


Module Contents
---------------

.. py:class:: EvalRunner(dataloader: torch.utils.data.dataloader, eval_unit: torchtnt.framework.EvalUnit, callbacks: list[torchtnt.framework.callback.Callback] | None = None, max_steps_per_epoch: int | None = None)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: dataloader


   .. py:attribute:: eval_unit


   .. py:attribute:: callbacks


   .. py:attribute:: max_steps_per_epoch


   .. py:method:: run() -> None


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


