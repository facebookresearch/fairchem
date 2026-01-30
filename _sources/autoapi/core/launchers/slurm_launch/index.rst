core.launchers.slurm_launch
===========================

.. py:module:: core.launchers.slurm_launch


Classes
-------

.. autoapisummary::

   core.launchers.slurm_launch.SlurmSPMDProgram


Functions
---------

.. autoapisummary::

   core.launchers.slurm_launch._get_slurm_env
   core.launchers.slurm_launch.map_job_config_to_dist_config
   core.launchers.slurm_launch.remove_runner_state_from_submission
   core.launchers.slurm_launch.runner_wrapper
   core.launchers.slurm_launch._set_seeds
   core.launchers.slurm_launch._set_deterministic_mode
   core.launchers.slurm_launch.slurm_launch
   core.launchers.slurm_launch.local_launch


Module Contents
---------------

.. py:function:: _get_slurm_env() -> fairchem.core.launchers.api.SlurmEnv

.. py:function:: map_job_config_to_dist_config(job_cfg: fairchem.core.launchers.api.JobConfig) -> dict

.. py:function:: remove_runner_state_from_submission(log_folder: str, job_id: str) -> None

.. py:function:: runner_wrapper(config: omegaconf.DictConfig, run_type: fairchem.core.launchers.api.RunType = RunType.RUN)

.. py:function:: _set_seeds(seed: int) -> None

.. py:function:: _set_deterministic_mode() -> None

.. py:class:: SlurmSPMDProgram

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   Entrypoint for a SPMD program launched via submitit on slurm.
   This assumes all ranks run the identical copy of this code


   .. py:attribute:: config
      :value: None



   .. py:attribute:: runner
      :value: None



   .. py:attribute:: reducer
      :value: None



   .. py:method:: __call__(dict_config: omegaconf.DictConfig, run_type: fairchem.core.launchers.api.RunType = RunType.RUN) -> None


   .. py:method:: _init_logger() -> None


   .. py:method:: checkpoint(*args, **kwargs) -> submitit.helpers.DelayedSubmission

      Resubmits the same callable with the same arguments



.. py:function:: slurm_launch(cfg: omegaconf.DictConfig, log_dir: str) -> None

.. py:function:: local_launch(cfg: omegaconf.DictConfig, log_dir: str)

   Launch locally with torch elastic (for >1 workers) or just single process


