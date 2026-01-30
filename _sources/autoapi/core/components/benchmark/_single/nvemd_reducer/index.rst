core.components.benchmark._single.nvemd_reducer
===============================================

.. py:module:: core.components.benchmark._single.nvemd_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.benchmark._single.nvemd_reducer.NVEMDReducer


Functions
---------

.. autoapisummary::

   core.components.benchmark._single.nvemd_reducer.moving_avg
   core.components.benchmark._single.nvemd_reducer.get_te_drift


Module Contents
---------------

.. py:function:: moving_avg(x, window=20)

.. py:function:: get_te_drift(filename)

.. py:class:: NVEMDReducer(benchmark_name: str)

   Bases: :py:obj:`fairchem.core.components.benchmark._benchmark_reducer.BenchmarkReducer`


   Benchmark reducer interface class.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and `runner_config` attributes are set at
      runtime to those given in the config file.

   See the Reducer interface class for implementation details.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig

   .. attribute:: runner_config

      a managed attributed that gives access to the calling runner config

      :type: DictConfig


   .. py:attribute:: benchmark_name


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.NVEMDRunner]


      The runner type this reducer is associated with.


   .. py:method:: join_results(results_dir: str, glob_pattern: str) -> pandas.DataFrame

      Join results from multiple JSON files into a single DataFrame.

      :param results_dir: Directory containing result files
      :param glob_pattern: Pattern to match result files

      :returns: Combined DataFrame containing all results



   .. py:method:: save_results(results: list, results_dir: str) -> None

      Save joined results to a compressed json file

      :param results: results: Combined results from join_results
      :param results_dir: Directory containing result files



   .. py:method:: compute_metrics(results: list, run_name: str) -> pandas.DataFrame

      Compute Matbench discovery metrics for relaxed energy and structure predictions.

      :param results: DataFrame containing prediction results with energy values
      :param run_name: Identifier for the current evaluation run

      :returns: DataFrame containing computed metrics for different material subsets



   .. py:method:: save_metrics(metrics: pandas.DataFrame, results_dir: str) -> None

      Save computed metrics to a compressed JSON file.

      :param metrics: DataFrame containing the computed metrics
      :param results_dir: Directory where metrics will be saved



   .. py:method:: log_metrics(metrics: pandas.DataFrame, run_name: str) -> None

      Log metrics to the configured logger if available.

      :param metrics: DataFrame containing the computed metrics
      :param run_name: Name of the current run



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool

      Save the current state of the reducer to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :param is_preemption: Whether the save is due to preemption

      :returns: Success status of the save operation
      :rtype: bool



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Load reducer state from a checkpoint.

      :param checkpoint_location: Location to load the checkpoint from, or None



