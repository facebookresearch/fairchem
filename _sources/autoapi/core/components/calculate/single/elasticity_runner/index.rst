core.components.calculate.single.elasticity_runner
==================================================

.. py:module:: core.components.calculate.single.elasticity_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.calculate.single.elasticity_runner.eVA3_to_GPa


Classes
-------

.. autoapisummary::

   core.components.calculate.single.elasticity_runner.ElasticityRunner


Module Contents
---------------

.. py:data:: eVA3_to_GPa

.. py:class:: ElasticityRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.atoms_sequence.AtomsSequence)

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



