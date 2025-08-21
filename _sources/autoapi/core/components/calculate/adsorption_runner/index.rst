core.components.calculate.adsorption_runner
===========================================

.. py:module:: core.components.calculate.adsorption_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.calculate.adsorption_runner.AdsorptionRunner


Module Contents
---------------

.. py:class:: AdsorptionRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.AseDBDataset, save_relaxed_atoms: bool = True, relax_surface: bool = False, optimizer_cls: type[ase.optimize.Optimizer] = LBFGS, fmax: float = 0.05, steps: int = 300)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


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



