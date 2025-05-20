core.datasets.mt_concat_dataset
===============================

.. py:module:: core.datasets.mt_concat_dataset

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.datasets.mt_concat_dataset.T_co


Classes
-------

.. autoapisummary::

   core.datasets.mt_concat_dataset.ConcatDataset


Functions
---------

.. autoapisummary::

   core.datasets.mt_concat_dataset.create_concat_dataset


Module Contents
---------------

.. py:data:: T_co

.. py:class:: ConcatDataset(datasets, sampling: dict)

   Bases: :py:obj:`torch.utils.data.Dataset`\ [\ :py:obj:`T_co`\ ]


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: cumsum(sequence, sample_ratios)
      :staticmethod:



   .. py:attribute:: datasets
      :value: []



   .. py:attribute:: dataset_names
      :value: []



   .. py:attribute:: sample_ratios


   .. py:attribute:: cumulative_sizes


   .. py:attribute:: real_sizes


   .. py:method:: __len__()


   .. py:method:: __getitem__(idx)


   .. py:method:: _get_dataset_and_sample_index_list(sample_idxs: list)


   .. py:method:: _get_dataset_and_sample_index(idx: int)


   .. py:property:: updated_dataset_sizes


   .. py:method:: metadata_hasattr(attr) -> bool


   .. py:method:: get_metadata(attr, sample_idxs_to_get_metadata_for)


   .. py:method:: _dataset_sampling(dataset_sizes: list[int], dataset_names: list[str], sampling: dict) -> list[float]
      :staticmethod:


      Return expansion ratios for each dataset based on sampling strategy



.. py:function:: create_concat_dataset(dataset_configs: omegaconf.DictConfig, combined_dataset_config: dict) -> ConcatDataset

   Make a concat dataset with all the splits for each dataset. Keys will be {dataset}.{split}


