core._cli
=========

.. py:module:: core._cli

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core._cli.ALLOWED_TOP_LEVEL_KEYS
   core._cli.LOG_DIR_NAME
   core._cli.CHECKPOINT_DIR_NAME
   core._cli.RESULTS_DIR
   core._cli.CONFIG_FILE_NAME
   core._cli.PREEMPTION_STATE_DIR_NAME


Classes
-------

.. autoapisummary::

   core._cli.SchedulerType
   core._cli.DeviceType
   core._cli.RunType
   core._cli.DistributedInitMethod
   core._cli.SlurmConfig
   core._cli.SchedulerConfig
   core._cli.SlurmEnv
   core._cli.Metadata
   core._cli.JobConfig
   core._cli.Submitit


Functions
---------

.. autoapisummary::

   core._cli._set_seeds
   core._cli._set_deterministic_mode
   core._cli._get_slurm_env
   core._cli.remove_runner_state_from_submission
   core._cli.map_job_config_to_dist_config
   core._cli.get_canonical_config
   core._cli.get_hydra_config_from_yaml
   core._cli._runner_wrapper
   core._cli.main


Module Contents
---------------

.. py:data:: ALLOWED_TOP_LEVEL_KEYS

.. py:data:: LOG_DIR_NAME
   :value: 'logs'


.. py:data:: CHECKPOINT_DIR_NAME
   :value: 'checkpoints'


.. py:data:: RESULTS_DIR
   :value: 'results'


.. py:data:: CONFIG_FILE_NAME
   :value: 'canonical_config.yaml'


.. py:data:: PREEMPTION_STATE_DIR_NAME
   :value: 'preemption_state'


.. py:class:: SchedulerType

   Bases: :py:obj:`fairchem.core.common.utils.StrEnum`


   Enum where members are also (and must be) strings


   .. py:attribute:: LOCAL
      :value: 'local'



   .. py:attribute:: SLURM
      :value: 'slurm'



.. py:class:: DeviceType

   Bases: :py:obj:`fairchem.core.common.utils.StrEnum`


   Enum where members are also (and must be) strings


   .. py:attribute:: CPU
      :value: 'cpu'



   .. py:attribute:: CUDA
      :value: 'cuda'



.. py:class:: RunType

   Bases: :py:obj:`fairchem.core.common.utils.StrEnum`


   Enum where members are also (and must be) strings


   .. py:attribute:: RUN
      :value: 'run'



   .. py:attribute:: REDUCE
      :value: 'reduce'



.. py:class:: DistributedInitMethod

   Bases: :py:obj:`fairchem.core.common.utils.StrEnum`


   Enum where members are also (and must be) strings


   .. py:attribute:: TCP
      :value: 'tcp'



   .. py:attribute:: FILE
      :value: 'file'



.. py:class:: SlurmConfig

   .. py:attribute:: mem_gb
      :type:  int
      :value: 80



   .. py:attribute:: timeout_hr
      :type:  int
      :value: 168



   .. py:attribute:: cpus_per_task
      :type:  int
      :value: 8



   .. py:attribute:: partition
      :type:  Optional[str]
      :value: None



   .. py:attribute:: qos
      :type:  Optional[str]
      :value: None



   .. py:attribute:: account
      :type:  Optional[str]
      :value: None



   .. py:attribute:: additional_parameters
      :type:  Optional[dict]
      :value: None



.. py:class:: SchedulerConfig

   .. py:attribute:: mode
      :type:  SchedulerType


   .. py:attribute:: distributed_init_method
      :type:  DistributedInitMethod


   .. py:attribute:: ranks_per_node
      :type:  int
      :value: 1



   .. py:attribute:: num_nodes
      :type:  int
      :value: 1



   .. py:attribute:: num_array_jobs
      :type:  int
      :value: 1



   .. py:attribute:: slurm
      :type:  SlurmConfig


.. py:class:: SlurmEnv

   .. py:attribute:: job_id
      :type:  Optional[str]
      :value: None



   .. py:attribute:: raw_job_id
      :type:  Optional[str]
      :value: None



   .. py:attribute:: array_job_id
      :type:  Optional[str]
      :value: None



   .. py:attribute:: array_task_id
      :type:  Optional[str]
      :value: None



   .. py:attribute:: restart_count
      :type:  Optional[str]
      :value: None



.. py:class:: Metadata

   .. py:attribute:: commit
      :type:  str


   .. py:attribute:: log_dir
      :type:  str


   .. py:attribute:: checkpoint_dir
      :type:  str


   .. py:attribute:: results_dir
      :type:  str


   .. py:attribute:: config_path
      :type:  str


   .. py:attribute:: preemption_checkpoint_dir
      :type:  str


   .. py:attribute:: cluster_name
      :type:  str


   .. py:attribute:: array_job_num
      :type:  int
      :value: 0



   .. py:attribute:: slurm_env
      :type:  SlurmEnv


.. py:class:: JobConfig

   .. py:attribute:: run_name
      :type:  str


   .. py:attribute:: timestamp_id
      :type:  str


   .. py:attribute:: run_dir
      :type:  str


   .. py:attribute:: device_type
      :type:  DeviceType


   .. py:attribute:: debug
      :type:  bool
      :value: False



   .. py:attribute:: scheduler
      :type:  SchedulerConfig


   .. py:attribute:: logger
      :type:  Optional[dict]
      :value: None



   .. py:attribute:: seed
      :type:  int
      :value: 0



   .. py:attribute:: deterministic
      :type:  bool
      :value: False



   .. py:attribute:: runner_state_path
      :type:  Optional[str]
      :value: None



   .. py:attribute:: metadata
      :type:  Optional[Metadata]
      :value: None



   .. py:attribute:: graph_parallel_group_size
      :type:  Optional[int]
      :value: None



   .. py:method:: __post_init__() -> None


.. py:function:: _set_seeds(seed: int) -> None

.. py:function:: _set_deterministic_mode() -> None

.. py:function:: _get_slurm_env() -> SlurmEnv

.. py:function:: remove_runner_state_from_submission(log_folder: str, job_id: str) -> None

.. py:class:: Submitit

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   Derived callable classes are requeued after timeout with their current
   state dumped at checkpoint.

   __call__ method must be implemented to make your class a callable.

   .. note::

      The following implementation of the checkpoint method resubmits the full current
      state of the callable (self) with the initial argument. You may want to replace the method to
      curate the state (dump a neural network to a standard format and remove it from
      the state so that not to pickle it) and change/remove the initial parameters.


   .. py:attribute:: config
      :value: None



   .. py:attribute:: runner
      :value: None



   .. py:attribute:: reducer
      :value: None



   .. py:method:: __call__(dict_config: omegaconf.DictConfig, run_type: RunType = RunType.RUN) -> None


   .. py:method:: _init_logger() -> None


   .. py:method:: checkpoint(*args, **kwargs) -> submitit.helpers.DelayedSubmission

      Resubmits the same callable with the same arguments



.. py:function:: map_job_config_to_dist_config(job_cfg: JobConfig) -> dict

.. py:function:: get_canonical_config(config: omegaconf.DictConfig) -> omegaconf.DictConfig

.. py:function:: get_hydra_config_from_yaml(config_yml: str, overrides_args: list[str]) -> omegaconf.DictConfig

.. py:function:: _runner_wrapper(config: omegaconf.DictConfig, run_type: RunType = RunType.RUN)

.. py:function:: main(args: argparse.Namespace | None = None, override_args: list[str] | None = None)

