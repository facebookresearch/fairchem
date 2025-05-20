core.components.benchmark.materials_discovery_reducer
=====================================================

.. py:module:: core.components.benchmark.materials_discovery_reducer

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.benchmark.materials_discovery_reducer.mbd_installed
   core.components.benchmark.materials_discovery_reducer.MP2020Compatibility


Classes
-------

.. autoapisummary::

   core.components.benchmark.materials_discovery_reducer.MaterialsDiscoveryReducer


Functions
---------

.. autoapisummary::

   core.components.benchmark.materials_discovery_reducer.as_dict_handler


Module Contents
---------------

.. py:data:: mbd_installed
   :value: True


.. py:data:: MP2020Compatibility

.. py:function:: as_dict_handler(obj: Any) -> dict[str, Any] | None

   Pass this to json.dump(default=) or as pandas.to_json(default_handler=) to
   serialize Python classes with as_dict(). Warning: Objects without a as_dict() method
   are replaced with None in the serialized data.

   From matbench_discovery: https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/data.py


.. py:class:: MaterialsDiscoveryReducer(benchmark_name: str, target_data_path: str, cse_data_path: str | None = None, elemental_references_path: str | None = None, index_name: str | None = None, corrections: pymatgen.entries.compatibility.Compatibility | None = MP2020Compatibility, max_error_threshold: float = 5.0, analyze_geo_opt: bool = True, geo_symprec: float = 1e-05)

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: _corrections


   .. py:attribute:: _max_error_threshold


   .. py:attribute:: _elemental_references_path


   .. py:attribute:: _cse_data_path


   .. py:attribute:: _analyze_geo_opt


   .. py:attribute:: _geo_symprec


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.RelaxationRunner]


      The runner type this reducer is associated with.


   .. py:method:: load_targets(path: str, index_name: str | None) -> pandas.DataFrame
      :staticmethod:


      Load target data from a JSON file into a pandas DataFrame.

      :param path: Path to the target JSON file
      :param index_name: Optional name of the column to use as index

      :returns: DataFrame containing the target data, sorted by index



   .. py:method:: _load_elemental_ref_energies(elemental_references_path: str) -> dict[str, float]
      :staticmethod:



   .. py:method:: _load_computed_structure_entries(cse_data_path: str, results: pandas.DataFrame) -> pandas.DataFrame
      :staticmethod:


      Convert prediction results to computed structure entries with updated energies and structures.

      :returns: DataFrame of computed structure entries indexed by material IDs



   .. py:method:: _apply_corrections(computed_structure_entries: list[pymatgen.entries.computed_entries.ComputedStructureEntry]) -> None

      Apply compatibility corrections to computed structure entries.

      :param computed_structure_entries: List of ComputedStructureEntry objects to apply corrections to

      :raises ValueError: If not all entries were successfully processed after applying corrections



   .. py:method:: _analyze_relaxed_geometry(pred_structures: dict[str, pymatgen.core.Structure], target_structures: dict[str, pymatgen.core.Structure]) -> dict[str, float]

      Analyze geometry of relaxed structures and calculate RMSD wrt to the target structures.

      :param pred_structures: Dictionary mapping material IDs to predicted Structure objects
      :param target_structures: Dictionary mapping material IDs to target Structure objects

      :returns: Dictionary containing geometric analysis metrics



   .. py:method:: join_results(results_dir: str, glob_pattern: str) -> pandas.DataFrame

      Join results from multiple relaxation JSON files into a single DataFrame.

      Joins results for relaxed energy, applies compatibility corrections, and computes formation energy
      w.r.t to MP reference structures in MatBench Discovery

      :param results_dir: Directory containing result files
      :param glob_pattern: Pattern to match result files

      :returns: Combined DataFrame containing all results



   .. py:method:: save_results(results: pandas.DataFrame, results_dir: str) -> None

      Save joined results to a single file

      Saves the results in two formats:
      1. CSV file containing only numerical data
      2. JSON file containing all data including relaxed structures

      :param results: DataFrame containing the prediction results
      :param results_dir: Directory path where result files will be saved



   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute Matbench discovery metrics for relaxed energy and structure predictions.

      :param results: DataFrame containing prediction results with energy values
      :param run_name: Identifier for the current evaluation run

      :returns: DataFrame containing computed metrics for different material subsets



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



