core.components.calculate
=========================

.. py:module:: core.components.calculate

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/components/calculate/adsorbml_runner/index
   /autoapi/core/components/calculate/adsorption_runner/index
   /autoapi/core/components/calculate/adsorption_singlepoint_runner/index
   /autoapi/core/components/calculate/calculate_runner/index
   /autoapi/core/components/calculate/elasticity_runner/index
   /autoapi/core/components/calculate/kappa_runner/index
   /autoapi/core/components/calculate/nve_md_runner/index
   /autoapi/core/components/calculate/omol_runner/index
   /autoapi/core/components/calculate/pairwise_ct_runner/index
   /autoapi/core/components/calculate/phonon_runner/index
   /autoapi/core/components/calculate/recipes/index
   /autoapi/core/components/calculate/relaxation_runner/index
   /autoapi/core/components/calculate/singlepoint_runner/index


Classes
-------

.. autoapisummary::

   core.components.calculate.CalculateRunner
   core.components.calculate.ElasticityRunner
   core.components.calculate.RelaxationRunner


Package Contents
----------------

.. py:class:: CalculateRunner(calculator: ase.calculators.Calculator, input_data: collections.abc.Sequence)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Runner to run calculations/predictions using an ASE-like calculator and save results to file.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   See the Runner interface class for implementation details.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig

   .. attribute:: result_glob_pattern

      glob pattern of results written to file

      :type: str


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: '*'



   .. py:attribute:: _calculator


   .. py:attribute:: _input_data


   .. py:attribute:: _already_calculated
      :value: False



   .. py:property:: calculator
      :type: ase.calculators.Calculator


      Get the calculator instance.

      :returns: The ASE-like calculator used for calculations
      :rtype: Calculator


   .. py:property:: input_data
      :type: collections.abc.Sequence


      Get the input data.

      :returns: The input data to be processed
      :rtype: Sequence


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> R
      :abstractmethod:


      Run any calculation using an ASE like Calculator.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: Results of the calculation
      :rtype: R



   .. py:method:: write_results(results: R, results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None
      :abstractmethod:


      Write results to file in results_dir.

      :param results: Results from the calculation
      :type results: R
      :param results_dir: Directory to write results to
      :type results_dir: str
      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool
      :abstractmethod:


      Save the current state of the calculation to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :type checkpoint_location: str
      :param is_preemption: Whether this save is due to preemption. Defaults to False.
      :type is_preemption: bool, optional

      :returns: True if state was successfully saved, False otherwise
      :rtype: bool



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Load a previously saved state from a checkpoint.

      :param checkpoint_location: Location of the checkpoint to load, or None if no checkpoint
      :type checkpoint_location: str | None



   .. py:method:: run()

      Run the actual calculation and save results.

      Creates the results directory if it doesn't exist, runs the calculation,
      and writes the results to the specified directory.

      .. note:: Re-implementing this method in derived classes is discouraged.



.. py:class:: ElasticityRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.AseDBDataset)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


   Calculate elastic tensor for a set of structures.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'elasticity_*-*.json.gz'



   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Calculate elastic properties for a batch of structures.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: List of dictionaries containing elastic properties for each structure



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

      Write calculation results to a compressed JSON file.

      :param results: List of dictionaries containing elastic properties
      :param results_dir: Directory path where results will be saved
      :param job_num: Index of the current job
      :param num_jobs: Total number of jobs



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool

      Save the current state of the calculation to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :type checkpoint_location: str
      :param is_preemption: Whether this save is due to preemption. Defaults to False.
      :type is_preemption: bool, optional

      :returns: True if state was successfully saved, False otherwise
      :rtype: bool



.. py:class:: RelaxationRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.AseDBDataset, calculate_properties: collections.abc.Sequence[str], save_relaxed_atoms: bool = True, normalize_properties_by: dict[str, str] | None = None, save_target_properties: collections.abc.Sequence[str] | None = None, **relax_kwargs)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


   Relax a sequence of several structures/molecules.

   This class handles the relaxation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'relaxation_*-*.json.gz'



   .. py:attribute:: _calculate_properties


   .. py:attribute:: _save_relaxed_atoms


   .. py:attribute:: _normalize_properties_by


   .. py:attribute:: _save_target_properties


   .. py:attribute:: _relax_kwargs


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Perform relaxation calculations on a subset of structures.

      Splits the input data into chunks and processes the chunk corresponding to job_num.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: list[dict[str, Any]] - List of dictionaries containing calculation results



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

      Write calculation results to a compressed JSON file.

      :param results: List of dictionaries containing elastic properties
      :param results_dir: Directory path where results will be saved
      :param job_num: Index of the current job
      :param num_jobs: Total number of jobs



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool

      Save the current state of the calculation to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :type checkpoint_location: str
      :param is_preemption: Whether this save is due to preemption. Defaults to False.
      :type is_preemption: bool, optional

      :returns: True if state was successfully saved, False otherwise
      :rtype: bool



