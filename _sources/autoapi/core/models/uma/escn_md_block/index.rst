core.models.uma.escn_md_block
=============================

.. py:module:: core.models.uma.escn_md_block

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.escn_md_block.Edgewise
   core.models.uma.escn_md_block.SpectralAtomwise
   core.models.uma.escn_md_block.GridAtomwise
   core.models.uma.escn_md_block.eSCNMD_Block


Functions
---------

.. autoapisummary::

   core.models.uma.escn_md_block.set_mole_ac_start_index


Module Contents
---------------

.. py:function:: set_mole_ac_start_index(module: torch.nn.Module, index: int) -> None

.. py:class:: Edgewise(sphere_channels: int, hidden_channels: int, lmax: int, mmax: int, edge_channels_list: list[int], mappingReduced: fairchem.core.models.uma.common.so3.CoefficientMapping, SO3_grid: fairchem.core.models.uma.common.so3.SO3_Grid, cutoff: float, activation_checkpoint_chunk_size: int | None, act_type: typing_extensions.Literal[gate, s2] = 'gate')

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: activation_checkpoint_chunk_size


   .. py:attribute:: mappingReduced


   .. py:attribute:: SO3_grid


   .. py:attribute:: edge_channels_list


   .. py:attribute:: act_type


   .. py:attribute:: so2_conv_1


   .. py:attribute:: so2_conv_2


   .. py:attribute:: cutoff


   .. py:attribute:: envelope


   .. py:attribute:: out_mask


   .. py:method:: forward(x, x_edge, edge_distance, edge_index, wigner_and_M_mapping, wigner_and_M_mapping_inv, node_offset: int = 0)


   .. py:method:: forward_chunk(x, x_edge, edge_distance, edge_index, wigner_and_M_mapping, wigner_and_M_mapping_inv, node_offset: int = 0, ac_mole_start_idx: int = 0)


.. py:class:: SpectralAtomwise(sphere_channels: int, hidden_channels: int, lmax: int, mmax: int, SO3_grid: fairchem.core.models.uma.common.so3.SO3_Grid)

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: SO3_grid


   .. py:attribute:: scalar_mlp


   .. py:attribute:: so3_linear_1


   .. py:attribute:: act


   .. py:attribute:: so3_linear_2


   .. py:method:: forward(x)


.. py:class:: GridAtomwise(sphere_channels: int, hidden_channels: int, lmax: int, mmax: int, SO3_grid: fairchem.core.models.uma.common.so3.SO3_Grid)

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: SO3_grid


   .. py:attribute:: grid_mlp


   .. py:method:: forward(x)


.. py:class:: eSCNMD_Block(sphere_channels: int, hidden_channels: int, lmax: int, mmax: int, mappingReduced: fairchem.core.models.uma.common.so3.CoefficientMapping, SO3_grid: fairchem.core.models.uma.common.so3.SO3_Grid, edge_channels_list: list[int], cutoff: float, norm_type: typing_extensions.Literal[layer_norm, layer_norm_sh, rms_norm_sh], act_type: typing_extensions.Literal[gate, s2], ff_type: typing_extensions.Literal[spectral, grid], activation_checkpoint_chunk_size: int | None)

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: norm_1


   .. py:attribute:: edge_wise


   .. py:attribute:: norm_2


   .. py:method:: forward(x, x_edge, edge_distance, edge_index, wigner_and_M_mapping, wigner_and_M_mapping_inv, sys_node_embedding=None, node_offset: int = 0)


