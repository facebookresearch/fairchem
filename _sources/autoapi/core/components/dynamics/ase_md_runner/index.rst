core.components.dynamics.ase_md_runner
======================================

.. py:module:: core.components.dynamics.ase_md_runner


Classes
-------

.. autoapisummary::

   core.components.dynamics.ase_md_runner.ASELangevinUMARunner


Module Contents
---------------

.. py:class:: ASELangevinUMARunner(atoms_list: list[ase.Atoms], model_name: str = 'uma-s-1p1', settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings | Literal['default', 'other options'] = 'default', task_name: str = 'omat', timestep_fs: float = 1.0, temp_k: float = 300.0, friction_ps_inv: float = 0.001, steps_total: int = 1000, warmup_steps: int = 10, workers: int = 0)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: atoms_list


   .. py:attribute:: timestep_fs


   .. py:attribute:: temp_k


   .. py:attribute:: friction_ps_inv


   .. py:attribute:: steps_total


   .. py:attribute:: warmup_steps


   .. py:attribute:: settings


   .. py:attribute:: model_name


   .. py:attribute:: workers


   .. py:attribute:: task_name


   .. py:method:: run()


   .. py:method:: _print_summary(results)

      Print a summary table of all results



   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


