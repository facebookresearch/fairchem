core.launchers.ray_on_slurm_launch
==================================

.. py:module:: core.launchers.ray_on_slurm_launch


Classes
-------

.. autoapisummary::

   core.launchers.ray_on_slurm_launch.SPMDWorker
   core.launchers.ray_on_slurm_launch.SPMDController


Functions
---------

.. autoapisummary::

   core.launchers.ray_on_slurm_launch.ray_entrypoint
   core.launchers.ray_on_slurm_launch.ray_on_slurm_launch


Module Contents
---------------

.. py:class:: SPMDWorker(job_config: omegaconf.DictConfig, runner_config: omegaconf.DictConfig, worker_id: int, world_size: int, device: str, gp_size: int | None = None, master_addr: str | None = None, master_port: int | None = None)

   .. py:attribute:: runner_config


   .. py:attribute:: master_address


   .. py:attribute:: master_port


   .. py:attribute:: worker_id


   .. py:attribute:: device


   .. py:attribute:: gp_size


   .. py:attribute:: world_size


   .. py:attribute:: job_config


   .. py:attribute:: distributed_setup
      :value: False



   .. py:method:: _distributed_setup(worker_id: int, world_size: int, master_address: str, master_port: int, device: str, gp_size: int | None)


   .. py:method:: get_master_address_and_port()


   .. py:method:: run()


.. py:class:: SPMDController(job_config: omegaconf.DictConfig, runner_config: omegaconf.DictConfig)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: job_config


   .. py:attribute:: runner_config


   .. py:attribute:: device


   .. py:attribute:: world_size


   .. py:attribute:: gp_group_size


   .. py:attribute:: ranks_per_node


   .. py:attribute:: num_nodes


   .. py:attribute:: workers


   .. py:method:: run()


   .. py:method:: save_state(checkpoint_location: str, is_preemption: bool = False) -> bool


   .. py:method:: load_state(checkpoint_location: str | None) -> None


.. py:function:: ray_entrypoint(runner_config: omegaconf.DictConfig, recursive_instantiate_runner: bool)

.. py:function:: ray_on_slurm_launch(config: omegaconf.DictConfig, log_dir: str)

