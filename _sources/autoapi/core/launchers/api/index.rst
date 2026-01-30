core.launchers.api
==================

.. py:module:: core.launchers.api

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.launchers.api.ALLOWED_TOP_LEVEL_KEYS
   core.launchers.api.LOG_DIR_NAME
   core.launchers.api.CHECKPOINT_DIR_NAME
   core.launchers.api.RESULTS_DIR
   core.launchers.api.CONFIG_FILE_NAME
   core.launchers.api.PREEMPTION_STATE_DIR_NAME


Classes
-------

.. autoapisummary::

   core.launchers.api.SchedulerType
   core.launchers.api.DeviceType
   core.launchers.api.RunType
   core.launchers.api.DistributedInitMethod
   core.launchers.api.SlurmConfig
   core.launchers.api.RayClusterConfig
   core.launchers.api.SchedulerConfig
   core.launchers.api.SlurmEnv
   core.launchers.api.Metadata
   core.launchers.api.JobConfig


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



.. py:class:: RayClusterConfig

   .. py:attribute:: head_gpus
      :type:  int
      :value: 0



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


   .. py:attribute:: use_ray
      :type:  bool
      :value: False



   .. py:attribute:: ray_cluster
      :type:  RayClusterConfig


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



   .. py:attribute:: recursive_instantiate_runner
      :type:  bool
      :value: True



   .. py:method:: __post_init__() -> None


