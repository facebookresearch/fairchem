core.components.calculate.adsorption_singlepoint_runner
=======================================================

.. py:module:: core.components.calculate.adsorption_singlepoint_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.calculate.adsorption_singlepoint_runner.AdsorptionSinglePointRunner


Module Contents
---------------

.. py:class:: AdsorptionSinglePointRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.AseDBDataset, evaluate_total_energy: bool = False, adsorption_energy_model: bool = False)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


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



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Load a previously saved state from a checkpoint.

      :param checkpoint_location: Location of the checkpoint to load, or None if no checkpoint
      :type checkpoint_location: str | None



