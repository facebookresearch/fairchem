core.components.calculate.pairwise_ct_runner
============================================

.. py:module:: core.components.calculate.pairwise_ct_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.calculate.pairwise_ct_runner.PairwiseCountRunner


Functions
---------

.. autoapisummary::

   core.components.calculate.pairwise_ct_runner.count_pairs


Module Contents
---------------

.. py:function:: count_pairs(tensor, max_value=100)

.. py:class:: PairwiseCountRunner(dataset_cfg='/checkpoint/ocp/shared/pairwise_data/preview_config.yaml', ds_name='omat', radius=3.5, portion=0.01)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Perform a single point calculation of several structures/molecules.

   This class handles the single point calculation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: dataset_cfg


   .. py:attribute:: ds_name


   .. py:attribute:: radius


   .. py:attribute:: portion


   .. py:method:: run()


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


