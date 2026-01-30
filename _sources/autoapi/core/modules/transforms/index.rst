core.modules.transforms
=======================

.. py:module:: core.modules.transforms

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.modules.transforms.DataTransforms


Functions
---------

.. autoapisummary::

   core.modules.transforms._get_molecule_cell
   core.modules.transforms.common_transform
   core.modules.transforms.ensure_tensor
   core.modules.transforms.ani1x_transform
   core.modules.transforms.trans1x_transform
   core.modules.transforms.spice_transform
   core.modules.transforms.qmof_transform
   core.modules.transforms.qm9_transform
   core.modules.transforms.omol_transform
   core.modules.transforms.stress_reshape_transform
   core.modules.transforms.asedb_transform
   core.modules.transforms.decompose_tensor


Module Contents
---------------

.. py:function:: _get_molecule_cell(data_object: fairchem.core.datasets.atomic_data.AtomicData)

.. py:function:: common_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: ensure_tensor(data_object, keys)

.. py:function:: ani1x_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: trans1x_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: spice_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: qmof_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: qm9_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: omol_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: stress_reshape_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:function:: asedb_transform(data_object: fairchem.core.datasets.atomic_data.AtomicData, config) -> fairchem.core.datasets.atomic_data.AtomicData

.. py:class:: DataTransforms(config)

   .. py:attribute:: config


   .. py:method:: __call__(data_object)


.. py:function:: decompose_tensor(data_object, config) -> fairchem.core.datasets.atomic_data.AtomicData

