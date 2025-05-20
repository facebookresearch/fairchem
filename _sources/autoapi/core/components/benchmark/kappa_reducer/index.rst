core.components.benchmark.kappa_reducer
=======================================

.. py:module:: core.components.benchmark.kappa_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.benchmark.kappa_reducer.mbd_installed


Classes
-------

.. autoapisummary::

   core.components.benchmark.kappa_reducer.Kappa103Reducer


Module Contents
---------------

.. py:data:: mbd_installed
   :value: True


.. py:class:: Kappa103Reducer(benchmark_name: str, target_data_path: Optional[str] = None, target_data_keys: collections.abc.Sequence[str] | None = None, index_name: str | None = 'mp_id')

   Bases: :py:obj:`fairchem.core.components.benchmark.benchmark_reducer.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.kappa_runner.KappaRunner]


      The runner type this reducer is associated with.


   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute Matbench discovery metrics for relaxed energy and structure predictions.

      :param results: DataFrame containing prediction results with energy values
      :param run_name: Identifier for the current evaluation run

      :returns: DataFrame containing computed metrics for different material subsets



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool

      Save the current state of the reducer to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :param is_preemption: Whether the save is due to preemption

      :returns: Success status of the save operation
      :rtype: bool



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Load reducer state from a checkpoint.

      :param checkpoint_location: Location to load the checkpoint from, or None



