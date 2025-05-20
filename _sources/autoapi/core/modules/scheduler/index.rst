core.modules.scheduler
======================

.. py:module:: core.modules.scheduler

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.modules.scheduler.CosineLRLambda
   core.modules.scheduler.LRScheduler


Functions
---------

.. autoapisummary::

   core.modules.scheduler.warmup_lr_lambda


Module Contents
---------------

.. py:function:: warmup_lr_lambda(current_step: int, optim_config)

   Returns a learning rate multiplier.
   Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
   and then gets multiplied by `lr_gamma` every time a milestone is crossed.


.. py:class:: CosineLRLambda(warmup_epochs: int, warmup_factor: float, epochs: int, lr_min_factor: float)

   .. py:attribute:: warmup_epochs


   .. py:attribute:: lr_warmup_factor


   .. py:attribute:: max_epochs


   .. py:attribute:: lr_min_factor


   .. py:method:: __call__(current_step: int) -> float


.. py:class:: LRScheduler(optimizer, config)

   Learning rate scheduler class for torch.optim learning rate schedulers

   .. rubric:: Notes

   If no learning rate scheduler is specified in the config the default
   scheduler is warmup_lr_lambda (fairchem.core.common.utils) not no scheduler,
   this is for backward-compatibility reasons. To run without a lr scheduler
   specify scheduler: "Null" in the optim section of the config.

   :param optimizer: torch optim object
   :type optimizer: obj
   :param config: Optim dict from the input config
   :type config: dict


   .. py:attribute:: optimizer


   .. py:attribute:: config


   .. py:method:: step(metrics=None, epoch=None) -> None


   .. py:method:: filter_kwargs(config)


   .. py:method:: get_lr()


