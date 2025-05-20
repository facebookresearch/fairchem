core.datasets.samplers.max_atom_distributed_sampler
===================================================

.. py:module:: core.datasets.samplers.max_atom_distributed_sampler

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.datasets.samplers.max_atom_distributed_sampler.MaxAtomDistributedBatchSampler


Functions
---------

.. autoapisummary::

   core.datasets.samplers.max_atom_distributed_sampler.get_batches


Module Contents
---------------

.. py:function:: get_batches(natoms_list: numpy.array, indices: numpy.array, max_atoms: int, min_atoms: int) -> tuple[list[list[int]], list[int], int]

   Greedily creates batches from a list of samples with varying numbers of atoms.

   Args:
   natoms_list: Array of number of atoms in each sample.
   indices: Array of indices of the samples.
   max_atoms: Maximum number of atoms allowed in a batch.

   Returns:
   tuple[list[list[int]], list[int], int]:
       A tuple containing a list of batches, a list of the total number of atoms in each batch,
       and the number of samples that were filtered out because they exceeded the maximum number of atoms.


.. py:class:: MaxAtomDistributedBatchSampler(dataset: fairchem.core.datasets.base_dataset.BaseDataset, max_atoms: int, num_replicas: int, rank: int, seed: int, shuffle: bool = True, drop_last: bool = False, min_atoms: int = 0)

   Bases: :py:obj:`torch.utils.data.Sampler`\ [\ :py:obj:`list`\ [\ :py:obj:`int`\ ]\ ]


   A custom batch sampler that distributes batches across multiple GPUs to ensure efficient training.

   Args:
   dataset (BaseDataset): The dataset to sample from.
   max_atoms (int): The maximum number of atoms allowed in a batch.
   num_replicas (int): The number of GPUs to distribute the batches across.
   rank (int): The rank of the current GPU.
   seed (int): The seed for shuffling the dataset.
   shuffle (bool): Whether to shuffle the dataset. Defaults to True.
   drop_last (bool): Whether to drop the last batch if its size is less than the maximum allowed size. Defaults to False.

   This batch sampler is designed to work with the BaseDataset class and is optimized for distributed training.
   It takes into account the number of atoms in each sample and ensures that the batches are distributed evenly across GPUs.


   .. py:attribute:: dataset


   .. py:attribute:: max_atoms


   .. py:attribute:: min_atoms


   .. py:attribute:: num_replicas


   .. py:attribute:: rank


   .. py:attribute:: seed


   .. py:attribute:: shuffle


   .. py:attribute:: drop_last


   .. py:attribute:: epoch
      :value: 0



   .. py:attribute:: start_iter
      :value: 0



   .. py:attribute:: all_batches


   .. py:attribute:: total_size


   .. py:method:: _prepare_batches() -> list[int]


   .. py:method:: __len__() -> int


   .. py:method:: __iter__() -> Iterator[list[int]]


   .. py:method:: set_epoch_and_start_iteration(epoch: int, start_iter: int) -> None


