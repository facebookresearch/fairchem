core.components.benchmark
=========================

.. py:module:: core.components.benchmark

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/components/benchmark/_benchmark_reducer/index
   /autoapi/core/components/benchmark/_single/index


Classes
-------

.. autoapisummary::

   core.components.benchmark.JsonDFReducer
   core.components.benchmark.AdsorbMLReducer
   core.components.benchmark.AdsorptionReducer
   core.components.benchmark.Kappa103Reducer
   core.components.benchmark.MaterialsDiscoveryReducer
   core.components.benchmark.NVEMDReducer
   core.components.benchmark.OMCPolymorphReducer
   core.components.benchmark.OMolReducer
   core.components.benchmark.InferenceBenchRunner


Package Contents
----------------

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



.. py:class:: AdsorbMLReducer(benchmark_name: str, target_data_key: str | None = None, index_name: str | None = None, threshold: float = 0.1)

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


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



.. py:class:: AdsorptionReducer(benchmark_name: str, target_data_key: str | None = None, index_name: str | None = None)

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: index_name


   .. py:attribute:: benchmark_name


   .. py:attribute:: target_data_key


   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute mean absolute error metrics for common columns between results and targets.

      :param results: DataFrame containing prediction results
      :param run_name: Name of the current run, used as index in the metrics DataFrame

      :returns: DataFrame containing computed metrics with run_name as index



.. py:class:: Kappa103Reducer(benchmark_name: str, target_data_path: Optional[str] = None, target_data_keys: collections.abc.Sequence[str] | None = None, index_name: str | None = 'mp_id')

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.KappaRunner]


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



.. py:class:: OMCPolymorphReducer(benchmark_name: str, target_data_key: str, molecule_id_key: str, calculate_structural_metrics: bool = False, index_name: str | None = None)

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


   A common pandas DataFrame reducer for benchmarks

   Results are assumed to be saved as json files that can be read into pandas dataframes.
   Only mean absolute error is computed for common columns in the predicted results and target data


   .. py:attribute:: _molecule_id_key


   .. py:attribute:: _calc_structural_metrics


   .. py:property:: runner_type
      :type: type[fairchem.core.components.calculate.SinglePointRunner | fairchem.core.components.calculate.RelaxationRunner]


      The runner type this reducer is associated with.


   .. py:method:: compute_metrics(results: pandas.DataFrame, run_name: str) -> pandas.DataFrame

      Compute OMC polymorph metrics for single point or relaxed energy and structure predictions.

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



.. py:class:: OMolReducer(benchmark_name: str, evaluator: Callable | None = None, benchmark_labels: str | None = None)

   Bases: :py:obj:`fairchem.core.components.benchmark.JsonDFReducer`


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



.. py:class:: InferenceBenchRunner(run_dir_root, model_checkpoints: dict[str, str], natoms_list: list[int] | None = None, input_system: dict | None = None, timeiters: int = 10, seed: int = 1, device='cuda', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings = inference_settings_default(), generate_traces: bool = False, dataset_name: str = 'omat')

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: natoms_list


   .. py:attribute:: input_system


   .. py:attribute:: device


   .. py:attribute:: seed


   .. py:attribute:: timeiters


   .. py:attribute:: model_checkpoints


   .. py:attribute:: run_dir


   .. py:attribute:: overrides


   .. py:attribute:: inference_settings


   .. py:attribute:: generate_traces


   .. py:attribute:: dataset_name


   .. py:method:: run() -> None


   .. py:method:: save_state(_)


   .. py:method:: load_state(_)


