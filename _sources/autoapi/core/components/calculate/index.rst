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

   /autoapi/core/components/calculate/_calculate_runner/index
   /autoapi/core/components/calculate/_single/index
   /autoapi/core/components/calculate/recipes/index


Classes
-------

.. autoapisummary::

   core.components.calculate.AdsorbMLRunner
   core.components.calculate.AdsorptionRunner
   core.components.calculate.AdsorptionSinglePointRunner
   core.components.calculate.ElasticityRunner
   core.components.calculate.KappaRunner
   core.components.calculate.NVEMDRunner
   core.components.calculate.OMolRunner
   core.components.calculate.PairwiseCountRunner
   core.components.calculate.MDRPhononRunner
   core.components.calculate.RelaxationRunner
   core.components.calculate.SinglePointRunner


Package Contents
----------------

.. py:class:: AdsorbMLRunner(calculator: ase.calculators.calculator.Calculator, input_data_path: str, place_on_relaxed_slab: bool = False, save_relaxed_atoms: bool = True, adsorption_energy_model: bool = False, num_placements: int = 100, optimizer_cls: type[ase.optimize.Optimizer] = LBFGS, fmax: float = 0.02, steps: int = 300)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Run the AdsorbML pipeline to identify the global minima adsorption energy.
   The option to also relax a clean surface is also provided.

   This class handles the relaxation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'adsorbml_*-*.json.gz'



   .. py:attribute:: _save_relaxed_atoms


   .. py:attribute:: place_on_relaxed_slab


   .. py:attribute:: adsorption_energy_model


   .. py:attribute:: num_placements


   .. py:attribute:: fmax


   .. py:attribute:: steps


   .. py:attribute:: optimizer_cls


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



.. py:class:: AdsorptionRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence, save_relaxed_atoms: bool = True, relax_surface: bool = False, optimizer_cls: type[ase.optimize.Optimizer] = LBFGS, fmax: float = 0.05, steps: int = 300)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Relax an adsorbate+surface configuration to compute the adsorption energy.
   The option to also relax a clean surface is also provided.

   This class handles the relaxation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.

   Input data is an AseDBDataset where each atoms object is organized as
   follows:
       atoms: adsorbate+surface configuration
       atoms.info = {
           gas_ref: float,
           dft_relaxed_adslab_energy: float,
           dft_relaxed_slab_energy: float,
           initial_slab_atoms: ase.Atoms, # Required if relax_surface=True
       }


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'adsorption_*-*.json.gz'



   .. py:attribute:: _save_relaxed_atoms


   .. py:attribute:: fmax


   .. py:attribute:: steps


   .. py:attribute:: relax_surface


   .. py:attribute:: optimizer_cls


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



.. py:class:: AdsorptionSinglePointRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence, evaluate_total_energy: bool = False, adsorption_energy_model: bool = False)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Singlepoint evaluator for OC20 Adsorption systems

   OC20 originally reported adsorption energies. This runner provides the
   ability to compute adsorption energy S2EF numbers by referencing to the
   provided slab atoms object. Total energy S2EF evaluations are also
   possible.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'adsorption-singlepoint_*-*.json.gz'



   .. py:attribute:: evaluate_total_energy


   .. py:attribute:: adsorption_energy_model


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

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



.. py:class:: ElasticityRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


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



.. py:class:: KappaRunner(calculator, input_data, displacement: float = 0.03)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Calculate elastic tensor for a set of structures.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'kappa103_dist*_*-*.json.gz'



   .. py:attribute:: displacement


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Run any calculation using an ASE like Calculator.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: Results of the calculation
      :rtype: R



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

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

      Save the current state of the calculation to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :type checkpoint_location: str
      :param is_preemption: Whether this save is due to preemption. Defaults to False.
      :type is_preemption: bool, optional

      :returns: True if state was successfully saved, False otherwise
      :rtype: bool



.. py:class:: NVEMDRunner(calculator: ase.calculators.calculator.Calculator, input_data: collections.abc.Sequence[tuple(Atoms, float)], time_step: float = 5, steps: float = 2000, save_frequency: int = 10)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Perform a single point calculation of several structures/molecules.

   This class handles the single point calculation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'thermo_*-*.log'



   .. py:attribute:: time_step


   .. py:attribute:: steps


   .. py:attribute:: save_frequency


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Perform singlepoint calculations on a subset of structures.

      Splits the input data into chunks and processes the chunk corresponding to job_num.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: list[dict[str, Any]] - List of dictionaries containing calculation results



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

      Write calculation results to a compressed JSON file.

      :param results: List of dictionaries containing energy and forces results
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



.. py:class:: OMolRunner(calculator: ase.calculators.calculator.Calculator, input_data: dict, benchmark_name: str, benchmark: Callable)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Runner for OMol's evaluation tasks.


   .. py:attribute:: result_glob_pattern


   .. py:attribute:: benchmark_name


   .. py:attribute:: benchmark


   .. py:attribute:: input_keys


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Perform calculations on a subset of structures.

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


.. py:class:: MDRPhononRunner(calculator: ase.calculators.calculator.Calculator, input_data: Sequence[dict], displacement: float = 0.01)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Calculate elastic tensor for a set of structures.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'mdr_phonon_dist*_*-*.json.gz'



   .. py:attribute:: displacement


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Run any calculation using an ASE like Calculator.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: Results of the calculation
      :rtype: R



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

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

      Save the current state of the calculation to a checkpoint.

      :param checkpoint_location: Location to save the checkpoint
      :type checkpoint_location: str
      :param is_preemption: Whether this save is due to preemption. Defaults to False.
      :type is_preemption: bool, optional

      :returns: True if state was successfully saved, False otherwise
      :rtype: bool



.. py:class:: RelaxationRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence, calculate_properties: collections.abc.Sequence[str] = ['energy'], save_relaxed_atoms: bool = True, normalize_properties_by: dict[str, str] | None = None, save_target_properties: collections.abc.Sequence[str] | None = None, **relax_kwargs)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


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



.. py:class:: SinglePointRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence, calculate_properties: collections.abc.Sequence[str] = ['energy'], normalize_properties_by: dict[str, str] | None = None, save_target_properties: collections.abc.Sequence[str] | None = None, save_atoms: bool = True)

   Bases: :py:obj:`fairchem.core.components.calculate._calculate_runner.CalculateRunner`


   Perform a single point calculation of several structures/molecules.

   This class handles the single point calculation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'singlepoint_*-*.json.gz'



   .. py:attribute:: _calculate_properties


   .. py:attribute:: _normalize_properties_by


   .. py:attribute:: _save_target_properties


   .. py:attribute:: _save_atoms


   .. py:method:: calculate(job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]

      Perform singlepoint calculations on a subset of structures.

      Splits the input data into chunks and processes the chunk corresponding to job_num.

      :param job_num: Current job number in array job. Defaults to 0.
      :type job_num: int, optional
      :param num_jobs: Total number of jobs in array. Defaults to 1.
      :type num_jobs: int, optional

      :returns: list[dict[str, Any]] - List of dictionaries containing calculation results



   .. py:method:: write_results(results: list[dict[str, Any]], results_dir: str, job_num: int = 0, num_jobs: int = 1) -> None

      Write calculation results to a compressed JSON file.

      :param results: List of dictionaries containing energy and forces results
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



