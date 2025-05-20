core.components.calculate.singlepoint_runner
============================================

.. py:module:: core.components.calculate.singlepoint_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.calculate.singlepoint_runner.SinglePointRunner


Module Contents
---------------

.. py:class:: SinglePointRunner(calculator: ase.calculators.calculator.Calculator, input_data: fairchem.core.datasets.AseDBDataset, calculate_properties: collections.abc.Sequence[str], normalize_properties_by: dict[str, str] | None = None, save_target_properties: collections.abc.Sequence[str] | None = None)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


   Perform a single point calculation of several structures/molecules.

   This class handles the single point calculation of atomic structures using a specified calculator,
   processes the input data in chunks, and saves the results.


   .. py:attribute:: result_glob_pattern
      :type:  ClassVar[str]
      :value: 'singlepoint_*-*.json.gz'



   .. py:attribute:: _calculate_properties


   .. py:attribute:: _normalize_properties_by


   .. py:attribute:: _save_target_properties


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



   .. py:method:: load_state(checkpoint_location: str | None) -> None

      Load a previously saved state from a checkpoint.

      :param checkpoint_location: Location of the checkpoint to load, or None if no checkpoint
      :type checkpoint_location: str | None



