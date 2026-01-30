core.common.distutils
=====================

.. py:module:: core.common.distutils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.common.distutils.T
   core.common.distutils.DISTRIBUTED_PORT
   core.common.distutils.CURRENT_DEVICE_TYPE_STR


Functions
---------

.. autoapisummary::

   core.common.distutils.os_environ_get_or_throw
   core.common.distutils.get_init_method
   core.common.distutils.setup
   core.common.distutils.cleanup
   core.common.distutils.initialized
   core.common.distutils.get_rank
   core.common.distutils.get_world_size
   core.common.distutils.is_master
   core.common.distutils.synchronize
   core.common.distutils.broadcast
   core.common.distutils.broadcast_object_list
   core.common.distutils.all_reduce
   core.common.distutils.all_gather
   core.common.distutils.gather_objects
   core.common.distutils.assign_device_for_local_rank
   core.common.distutils.get_device_for_local_rank
   core.common.distutils.setup_env_local
   core.common.distutils.setup_env_local_multi_gpu


Module Contents
---------------

.. py:data:: T

.. py:data:: DISTRIBUTED_PORT
   :value: 13356


.. py:data:: CURRENT_DEVICE_TYPE_STR
   :value: 'CURRRENT_DEVICE_TYPE'


.. py:function:: os_environ_get_or_throw(x: str) -> str

.. py:function:: get_init_method(init_method, world_size: int | None, rank: int | None = None, node_list: str | None = None, filename: str | None = None)

   Get the initialization method for a distributed job based on the specified method type.

   :param init_method: The initialization method type, either "tcp" or "file".
   :param world_size: The total number of processes in the distributed job.
   :param rank: The rank of the current process (optional).
   :param node_list: The list of nodes for SLURM-based distributed job (optional, used with "tcp").
   :param filename: The shared file path for file-based initialization (optional, used with "file").

   :returns: The initialization method string to be used by PyTorch's distributed module.

   :raises ValueError: If an invalid init_method is provided.


.. py:function:: setup(config) -> None

.. py:function:: cleanup() -> None

.. py:function:: initialized() -> bool

.. py:function:: get_rank() -> int

.. py:function:: get_world_size() -> int

.. py:function:: is_master() -> bool

.. py:function:: synchronize() -> None

.. py:function:: broadcast(tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False) -> None

.. py:function:: broadcast_object_list(object_list: list[Any], src: int, group=dist.group.WORLD, device: str | None = None) -> None

.. py:function:: all_reduce(data, group=dist.group.WORLD, average: bool = False, device=None) -> torch.Tensor

.. py:function:: all_gather(data, group=dist.group.WORLD, device=None) -> list[torch.Tensor]

.. py:function:: gather_objects(data: T, group: torch.distributed.ProcessGroup = dist.group.WORLD) -> list[T]

   Gather a list of pickleable objects into rank 0


.. py:function:: assign_device_for_local_rank(cpu: bool, local_rank: int) -> None

.. py:function:: get_device_for_local_rank() -> str

.. py:function:: setup_env_local()

.. py:function:: setup_env_local_multi_gpu(rank: int, port: int, address: str = 'localhost')

