core.components.benchmark.adsorbml_reducer
==========================================

.. py:module:: core.components.benchmark.adsorbml_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.benchmark.adsorbml_reducer.R
   core.components.benchmark.adsorbml_reducer.M


Classes
-------

.. autoapisummary::

   core.components.benchmark.adsorbml_reducer.AdsorbMLReducer


Module Contents
---------------

.. py:data:: R

.. py:data:: M

.. py:class:: AdsorbMLReducer(benchmark_name: str, target_data_key: str | None = None, index_name: str | None = None, threshold: float = 0.1)

   Bases: :py:obj:`fairchem.core.components.benchmark.benchmark_reducer.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: index_name


   .. py:attribute:: benchmark_name


   .. py:attribute:: target_data_key


   .. py:attribute:: threshold


   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute mean absolute error metrics for common columns between results and targets.

      :param results: DataFrame containing prediction results
      :param run_name: Name of the current run, used as index in the metrics DataFrame

      :returns: DataFrame containing computed metrics with run_name as index



