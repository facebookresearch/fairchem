core.components.benchmark.benchmark_reducer
===========================================

.. py:module:: core.components.benchmark.benchmark_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.benchmark.benchmark_reducer.R
   core.components.benchmark.benchmark_reducer.M


Classes
-------

.. autoapisummary::

   core.components.benchmark.benchmark_reducer.BenchmarkReducer
   core.components.benchmark.benchmark_reducer.JsonDFReducer


Module Contents
---------------

.. py:data:: R

.. py:data:: M

.. py:class:: BenchmarkReducer

   Bases: :py:obj:`fairchem.core.components.reducer.Reducer`


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


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.calculate_runner.CalculateRunner]


      The runner type this reducer is associated with.


   .. py:property:: glob_pattern

      Returns the glob pattern used to find result files from the runner.


   .. py:property:: logger
      :type: fairchem.core.common.logger.WandBSingletonLogger | None


      Returns a logger instance if conditions are met, otherwise None.

      :returns: Logger instance if running on main rank with logging enabled
      :rtype: WandBSingletonLogger or None


   .. py:method:: join_results(results_dir: str, glob_pattern: str) -> R
      :abstractmethod:


      Join results from multiple files into a single result object.

      :param results_dir: Directory containing result files
      :param glob_pattern: Pattern to match result files

      :returns: Combined results object of type R



   .. py:method:: save_results(results: R, results_dir: str) -> None
      :abstractmethod:


      Save joined results to file

      :param results: results: Combined results from join_results
      :param results_dir: Directory containing result files



   .. py:method:: compute_metrics(results: R, run_name: str) -> M
      :abstractmethod:


      Compute metrics from the joined results.

      :param results: Combined results from join_results
      :param run_name: Name of the current run

      :returns: Metrics object of type M



   .. py:method:: save_metrics(metrics: M, results_dir: str) -> None
      :abstractmethod:


      Save computed metrics to disk.

      :param metrics: Metrics object to save
      :param results_dir: Directory to save metrics to



   .. py:method:: log_metrics(metrics: M, run_name: str)
      :abstractmethod:


      Log metrics to the configured logger.

      :param metrics: Metrics object to log
      :param run_name: Name of the current run



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool
      :abstractmethod:


      Save the current state of the reducer to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :param is_preemption: Whether the save is due to preemption

      :returns: Success status of the save operation
      :rtype: bool



   .. py:method:: load_state(checkpoint_location: str | None) -> None
      :abstractmethod:


      Load reducer state from a checkpoint.

      :param checkpoint_location: Location to load the checkpoint from, or None



   .. py:method:: reduce()

      Join results, compute metrics, save and log resulting metrics.

      .. note:: Re-implementing this method in derived classes is discouraged.



.. py:class:: JsonDFReducer(benchmark_name: str, target_data_path: str | None = None, target_data_keys: collections.abc.Sequence[str] | None = None, index_name: str | None = None)

   Bases: :py:obj:`BenchmarkReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: index_name


   .. py:attribute:: benchmark_name


   .. py:attribute:: target_data


   .. py:attribute:: target_data_keys


   .. py:method:: load_targets(path: str, index_name: str | None) -> pandas.DataFrame
      :staticmethod:


      Load target data from a JSON file into a pandas DataFrame.

      :param path: Path to the target JSON file
      :param index_name: Optional name of the column to use as index

      :returns: DataFrame containing the target data, sorted by index



   .. py:method:: join_results(results_dir: str, glob_pattern: str) -> pandas.DataFrame

      Join results from multiple JSON files into a single DataFrame.

      :param results_dir: Directory containing result files
      :param glob_pattern: Pattern to match result files

      :returns: Combined DataFrame containing all results



   .. py:method:: save_results(results: pandas.DataFrame, results_dir: str) -> None

      Save joined results to a compressed json file

      :param results: results: Combined results from join_results
      :param results_dir: Directory containing result files



   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute mean absolute error metrics for common columns between results and targets.

      :param results: DataFrame containing prediction results
      :param run_name: Name of the current run, used as index in the metrics DataFrame

      :returns: DataFrame containing computed metrics with run_name as index



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



