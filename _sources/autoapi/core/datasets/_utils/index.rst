core.datasets._utils
====================

.. py:module:: core.datasets._utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.datasets._utils.rename_data_object_keys


Module Contents
---------------

.. py:function:: rename_data_object_keys(data_object: fairchem.core.datasets.atomic_data.AtomicData, key_mapping: dict[str, str | list[str]]) -> fairchem.core.datasets.atomic_data.AtomicData

   Rename data object keys

   :param data_object: data object
   :param key_mapping: dictionary specifying keys to rename and new names {prev_key: new_key}
   :param new_key can be a list of new keys:
   :param for example:
   :param :
   :param prev_key: energy
   :param new_key: [common_energy, oc20_energy]
   :param This is currently required when we use a single target/label for multiple tasks:


