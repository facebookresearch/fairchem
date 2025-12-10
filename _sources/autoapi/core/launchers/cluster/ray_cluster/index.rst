core.launchers.cluster.ray_cluster
==================================

.. py:module:: core.launchers.cluster.ray_cluster


Attributes
----------

.. autoapisummary::

   core.launchers.cluster.ray_cluster.start_ip_pattern
   core.launchers.cluster.ray_cluster.PayloadReturnT


Classes
-------

.. autoapisummary::

   core.launchers.cluster.ray_cluster.HeadInfo
   core.launchers.cluster.ray_cluster.RayClusterState
   core.launchers.cluster.ray_cluster.CheckpointableRayJob
   core.launchers.cluster.ray_cluster.RayCluster


Functions
---------

.. autoapisummary::

   core.launchers.cluster.ray_cluster.kill_proc_tree
   core.launchers.cluster.ray_cluster.find_free_port
   core.launchers.cluster.ray_cluster.scancel
   core.launchers.cluster.ray_cluster.mk_symlinks
   core.launchers.cluster.ray_cluster._ray_head_script
   core.launchers.cluster.ray_cluster.worker_script


Module Contents
---------------

.. py:function:: kill_proc_tree(pid, including_parent=True)

.. py:function:: find_free_port()

.. py:function:: scancel(job_ids: list[str])

   Cancel the SLURM jobs with the given job IDs.

   This function takes a list of job IDs.

   :param job_ids: A list of job IDs to cancel.
   :type job_ids: List[str]


.. py:data:: start_ip_pattern
   :value: "ray start --address='([0-9\\.]+):([0-9]+)'"


.. py:data:: PayloadReturnT

.. py:function:: mk_symlinks(target_dir: pathlib.Path, job_type: str, paths: submitit.core.utils.JobPaths)

   Create symlinks for the job's stdout and stderr in the target directory with a nicer name.


.. py:class:: HeadInfo

   information about the head node that we can share to workers


   .. py:attribute:: hostname
      :type:  Optional[str]
      :value: None



   .. py:attribute:: port
      :type:  Optional[int]
      :value: None



   .. py:attribute:: temp_dir
      :type:  Optional[str]
      :value: None



.. py:class:: RayClusterState(rdv_dir: Optional[pathlib.Path] = None, cluster_id: Optional[str] = None)

   This class is responsible for managing the state of the Ray cluster. It is useful to keep track
   of the head node and the workers, and to make sure they are all ready before starting the payload.

   It relies on storing info in a rendezvous directory so they can be shared async between jobs.

   :param rdv_dir: The directory where the rendezvous information will be stored. Defaults to ~/.fairray.
   :type rdv_dir: Path
   :param cluster_id: A unique identifier for the cluster. Defaults to a random UUID. You only want to set this if you want to connect to an existing cluster.
   :type cluster_id: str


   .. py:attribute:: rendezvous_rootdir


   .. py:attribute:: _cluster_id


   .. py:property:: cluster_id
      :type: str


      Returns the unique identifier for the cluster.


   .. py:property:: rendezvous_dir
      :type: pathlib.Path


      Returns the path to the directory where the rendezvous information is stored.


   .. py:property:: jobs_dir
      :type: pathlib.Path


      Returns the path to the directory where job information is stored.


   .. py:property:: _head_json
      :type: pathlib.Path


      Returns the path to the JSON file containing head node information.


   .. py:method:: is_head_ready() -> bool

      Checks if the head node information is available and ready.



   .. py:method:: head_info() -> Optional[HeadInfo]

      Retrieves the head node information from the stored JSON file.

      :returns: The head node information if available, otherwise None.
      :rtype: Optional[HeadInfo]



   .. py:method:: save_head_info(head_info: HeadInfo)

      Saves the head node information to a JSON file.

      :param head_info: The head node information to save.
      :type head_info: HeadInfo



   .. py:method:: reset_state()

      Resets the head node information by removing the stored JSON file, useful for preemption resumes



   .. py:method:: clean()

      Removes the rendezvous directory and all its contents.



   .. py:method:: add_job(job: submitit.Job)

      Adds a job to the jobs directory by creating a JSON file with the job's information.

      :param job: The job to add.
      :type job: submitit.Job



   .. py:method:: list_job_ids() -> list[str]

      Lists all job IDs stored in the jobs directory.



.. py:class:: CheckpointableRayJob(cluster_state: RayClusterState, worker_wait_timeout_seconds: int, payload: Optional[Callable[Ellipsis, PayloadReturnT]], **kwargs)

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   A checkpointable Ray job that can restart itself upon failure or preemption.
   It gang schedules the head and worker nodes together to keep preemption logic simple.


   .. py:attribute:: cluster_state


   .. py:attribute:: worker_wait_timeout_seconds


   .. py:attribute:: payload


   .. py:attribute:: kwargs


   .. py:method:: __call__()


   .. py:method:: checkpoint() -> submitit.helpers.DelayedSubmission

      Resubmits the same callable with the same arguments



.. py:function:: _ray_head_script(cluster_state: RayClusterState, worker_wait_timeout_seconds: int, payload: Optional[Callable[Ellipsis, PayloadReturnT]] = None, **kwargs)

   Start the head node of the Ray cluster on slurm.


.. py:function:: worker_script(cluster_state: RayClusterState, worker_wait_timeout_seconds: int, start_wait_time_seconds: int = 60)

   start an array of worker nodes for the Ray cluster on slurm. Waiting on the head node first.


.. py:class:: RayCluster(log_dir: pathlib.Path = Path('raycluster_logs'), rdv_dir: Optional[pathlib.Path] = None, cluster_id: Optional[str] = None, worker_wait_timeout_seconds: int = 60)

   A RayCluster offers tools to start a Ray cluster (head and wokers) on slurm with the correct settings.

   args:

   log_dir: Path to the directory where logs will be stored. Defaults to "raycluster_logs" in the working directory. All slurm logs will go there,
   and it also creates symlinks to the stdout/stderr of each jobs with nicer name (head, worker_0, worker_1, ..., driver_0, etc). There interesting
   logs will be in the driver_N.err file, you should tail that.
   rdv_dir: Path to the directory where the rendezvous information will be stored. Defaults to ~/.fairray. Useful if you are trying to recover an existing cluster.
   cluster_id: A unique identifier for the cluster. Defaults to a random UUID. You only want to set this if you want to connect to an existing cluster.
   worker_wait_timeout_seconds (int): The number of seconds ray will wait for a worker to be ready before giving up. Defaults to 60 seconds. If you are scheduling
       workers in a queue that takes time for allocation, you might want to increase this otherwise your ray payload will fail, not finding resources.



   .. py:attribute:: state


   .. py:attribute:: output_dir


   .. py:attribute:: log_dir


   .. py:attribute:: worker_wait_timeout_seconds


   .. py:attribute:: is_shutdown
      :value: False



   .. py:attribute:: num_worker_groups
      :value: 0



   .. py:attribute:: num_drivers
      :value: 0



   .. py:attribute:: head_started
      :value: False



   .. py:attribute:: jobs
      :type:  list[submitit.Job]
      :value: []



   .. py:method:: start_head_and_workers(requirements: dict[str, int | str], name: str = 'default', executor: str = 'slurm', payload: Optional[Callable[Ellipsis, PayloadReturnT]] = None, **kwargs)


   .. py:method:: start_head(requirements: dict[str, int | str], name: str = 'default', executor: str = 'slurm', payload: Optional[Callable[Ellipsis, PayloadReturnT]] = None, **kwargs) -> str

      Start the head node of the Ray cluster on slurm. You should do this first. Interesting requirements: qos, partition, time, gpus, cpus-per-task, mem-per-gpu, etc.



   .. py:method:: start_workers(num_workers: int, requirements: dict[str, int | str], name: str = 'default', executor: str = 'slurm') -> list[str]

      Start an array of worker nodes of the Ray cluster on slurm. You should do this after starting a head.
      Interesting requirements: qos, partition, time, gpus, cpus-per-task, mem-per-gpu, etc.
      You can call this multiple times to start an heterogeneous cluster.



   .. py:method:: shutdown()

      Cancel all slurms jobs and get rid of rdv directory.



   .. py:method:: __enter__()


   .. py:method:: __exit__(exc_type, exc_value, traceback)


