core.components.calculate.adsorbml_runner
=========================================

.. py:module:: core.components.calculate.adsorbml_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.calculate.adsorbml_runner.pmg_installed


Classes
-------

.. autoapisummary::

   core.components.calculate.adsorbml_runner.AdsorbMLRunner


Module Contents
---------------

.. py:data:: pmg_installed
   :value: True


.. py:class:: AdsorbMLRunner(calculator: ase.calculators.calculator.Calculator, input_data_path: str, place_on_relaxed_slab: bool = False, save_relaxed_atoms: bool = True, adsorption_energy_model: bool = False, num_placements: int = 100, optimizer_cls: type[ase.optimize.Optimizer] = LBFGS, fmax: float = 0.02, steps: int = 300)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


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



