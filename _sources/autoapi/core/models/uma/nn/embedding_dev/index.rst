core.models.uma.nn.embedding_dev
================================

.. py:module:: core.models.uma.nn.embedding_dev

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.nn.embedding_dev.EdgeDegreeEmbedding
   core.models.uma.nn.embedding_dev.ChgSpinEmbedding
   core.models.uma.nn.embedding_dev.DatasetEmbedding


Module Contents
---------------

.. py:class:: EdgeDegreeEmbedding(sphere_channels: int, lmax: int, mmax: int, max_num_elements: int, edge_channels_list, rescale_factor, cutoff, mappingReduced, activation_checkpoint_chunk_size: int | None)

   Bases: :py:obj:`torch.nn.Module`


   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param lmax: degrees (l)
   :type lmax: int
   :param mmax: orders (m)
   :type mmax: int
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                    The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
   :param use_atom_edge_embedding: Whether to use atomic embedding along with relative distance for edge scalar features
   :type use_atom_edge_embedding: bool
   :param rescale_factor: Rescale the sum aggregation
   :type rescale_factor: float
   :param cutoff: Cutoff distance for the radial function
   :type cutoff: float
   :param mappingReduced: Class to convert l and m indices once node embedding is rotated
   :type mappingReduced: CoefficientMapping


   .. py:attribute:: sphere_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: mappingReduced


   .. py:attribute:: activation_checkpoint_chunk_size


   .. py:attribute:: m_0_num_coefficients
      :type:  int


   .. py:attribute:: m_all_num_coefficents
      :type:  int


   .. py:attribute:: max_num_elements


   .. py:attribute:: edge_channels_list


   .. py:attribute:: rad_func


   .. py:attribute:: rescale_factor


   .. py:attribute:: cutoff


   .. py:attribute:: envelope


   .. py:method:: forward_chunk(x, x_edge, edge_distance, edge_index, wigner_and_M_mapping_inv, node_offset=0)


   .. py:method:: forward(x, x_edge, edge_distance, edge_index, wigner_and_M_mapping_inv, node_offset=0)


.. py:class:: ChgSpinEmbedding(embedding_type, embedding_target, embedding_size, grad, scale=1.0)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: embedding_type


   .. py:attribute:: embedding_target


   .. py:method:: forward(x)


.. py:class:: DatasetEmbedding(embedding_size, grad, dataset_list)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: embedding_size


   .. py:attribute:: dataset_emb_dict


   .. py:method:: forward(dataset_list)


