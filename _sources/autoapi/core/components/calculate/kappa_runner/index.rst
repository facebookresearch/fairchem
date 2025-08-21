core.components.calculate.kappa_runner
======================================

.. py:module:: core.components.calculate.kappa_runner

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.calculate.kappa_runner.mbd_installed


Classes
-------

.. autoapisummary::

   core.components.calculate.kappa_runner.KappaRunner


Functions
---------

.. autoapisummary::

   core.components.calculate.kappa_runner.get_kappa103_data_list


Module Contents
---------------

.. py:data:: mbd_installed
   :value: True


.. py:function:: get_kappa103_data_list(reference_data_path: str, debug=False)

.. py:class:: KappaRunner(calculator, input_data, displacement: float = 0.03)

   Bases: :py:obj:`fairchem.core.components.calculate.CalculateRunner`


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



