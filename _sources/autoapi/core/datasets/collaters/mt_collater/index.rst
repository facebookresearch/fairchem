core.datasets.collaters.mt_collater
===================================

.. py:module:: core.datasets.collaters.mt_collater

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.datasets.collaters.mt_collater.MTCollater


Module Contents
---------------

.. py:class:: MTCollater(task_config, exclude_keys, otf_graph: bool = False)

   .. py:attribute:: exclude_keys


   .. py:attribute:: task_config


   .. py:attribute:: dataset_task_map


   .. py:method:: __call__(data_list: list[fairchem.core.datasets.atomic_data.AtomicData]) -> fairchem.core.datasets.atomic_data.AtomicData


   .. py:method:: data_list_collater(data_list: list[fairchem.core.datasets.atomic_data.AtomicData], dataset_task_map: dict, exclude_keys: list) -> fairchem.core.datasets.atomic_data.AtomicData


   .. py:method:: _create_dataset_task_map(config)


   .. py:method:: _add_missing_attr(data_list, dataset_task_map)

      add missing data object attributes as inf



