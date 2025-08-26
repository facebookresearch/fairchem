core.components.benchmark.omol_reducer
======================================

.. py:module:: core.components.benchmark.omol_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.benchmark.omol_reducer.R
   core.components.benchmark.omol_reducer.M


Classes
-------

.. autoapisummary::

   core.components.benchmark.omol_reducer.OMolReducer


Module Contents
---------------

.. py:data:: R

.. py:data:: M

.. py:class:: OMolReducer(benchmark_name: str, evaluator: Callable | None = None, benchmark_labels: str | None = None)

   Bases: :py:obj:`fairchem.core.components.benchmark.benchmark_reducer.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: benchmark_name


   .. py:attribute:: benchmark_labels


   .. py:attribute:: evaluator


   .. py:method:: join_results(results_dir: str, glob_pattern: str) -> pandas.DataFrame

      Join results from multiple JSON files into a single DataFrame.

      :param results_dir: Directory containing result files
      :param glob_pattern: Pattern to match result files

      :returns: Combined DataFrame containing all results



   .. py:method:: save_results(results: pandas.DataFrame, results_dir: str) -> None

      Save joined results to a compressed json file

      :param results: results: Combined results from join_results
      :param results_dir: Directory containing result files



   .. py:method:: compute_metrics(results: dict, run_name: str) -> pandas.DataFrame

      Compute mean absolute error metrics for common columns between results and targets.

      :param results: DataFrame containing prediction results
      :param run_name: Name of the current run, used as index in the metrics DataFrame

      :returns: DataFrame containing computed metrics with run_name as index



