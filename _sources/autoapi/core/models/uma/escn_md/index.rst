core.models.uma.escn_md
=======================

.. py:module:: core.models.uma.escn_md

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.models.uma.escn_md.ESCNMD_DEFAULT_EDGE_CHUNK_SIZE


Classes
-------

.. autoapisummary::

   core.models.uma.escn_md.eSCNMDBackbone
   core.models.uma.escn_md.MLP_EFS_Head
   core.models.uma.escn_md.MLP_Energy_Head
   core.models.uma.escn_md.Linear_Energy_Head
   core.models.uma.escn_md.Linear_Force_Head
   core.models.uma.escn_md.MLP_Stress_Head


Functions
---------

.. autoapisummary::

   core.models.uma.escn_md.compose_tensor


Module Contents
---------------

.. py:data:: ESCNMD_DEFAULT_EDGE_CHUNK_SIZE

.. py:class:: eSCNMDBackbone(max_num_elements: int = 100, sphere_channels: int = 128, lmax: int = 2, mmax: int = 2, grid_resolution: int | None = None, num_sphere_samples: int = 128, otf_graph: bool = False, max_neighbors: int = 300, use_pbc: bool = True, use_pbc_single: bool = True, cutoff: float = 5.0, edge_channels: int = 128, distance_function: str = 'gaussian', num_distance_basis: int = 512, direct_forces: bool = True, regress_forces: bool = True, regress_stress: bool = False, num_layers: int = 2, hidden_channels: int = 128, norm_type: str = 'rms_norm_sh', act_type: str = 'gate', ff_type: str = 'grid', activation_checkpointing: bool = False, chg_spin_emb_type: str = 'pos_emb', cs_emb_grad: bool = False, dataset_emb_grad: bool = False, dataset_list: list[str] | None = None, use_dataset_embedding: bool = True, use_cuda_graph_wigner: bool = False, radius_pbc_version: int = 1, always_use_pbc: bool = True)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.uma.nn.mole_utils.MOLEInterface`


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


   .. py:attribute:: max_num_elements


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: sphere_channels


   .. py:attribute:: grid_resolution


   .. py:attribute:: num_sphere_samples


   .. py:attribute:: always_use_pbc


   .. py:attribute:: regress_forces


   .. py:attribute:: direct_forces


   .. py:attribute:: regress_stress


   .. py:attribute:: otf_graph


   .. py:attribute:: max_neighbors


   .. py:attribute:: radius_pbc_version


   .. py:attribute:: enforce_max_neighbors_strictly
      :value: False



   .. py:attribute:: chg_spin_emb_type


   .. py:attribute:: cs_emb_grad


   .. py:attribute:: dataset_emb_grad


   .. py:attribute:: dataset_list


   .. py:attribute:: use_dataset_embedding


   .. py:attribute:: use_cuda_graph_wigner


   .. py:attribute:: sph_feature_size


   .. py:attribute:: mappingReduced


   .. py:attribute:: SO3_grid


   .. py:attribute:: sphere_embedding


   .. py:attribute:: charge_embedding


   .. py:attribute:: spin_embedding


   .. py:attribute:: cutoff


   .. py:attribute:: edge_channels


   .. py:attribute:: distance_function


   .. py:attribute:: num_distance_basis


   .. py:attribute:: source_embedding


   .. py:attribute:: target_embedding


   .. py:attribute:: edge_channels_list


   .. py:attribute:: edge_degree_embedding


   .. py:attribute:: num_layers


   .. py:attribute:: hidden_channels


   .. py:attribute:: norm_type


   .. py:attribute:: act_type


   .. py:attribute:: ff_type


   .. py:attribute:: blocks


   .. py:attribute:: norm


   .. py:attribute:: rot_mat_wigner_cuda
      :value: None



   .. py:method:: _get_rotmat_and_wigner(edge_distance_vecs: torch.Tensor, use_cuda_graph: bool)


   .. py:method:: _get_displacement_and_cell(data_dict)


   .. py:method:: csd_embedding(charge, spin, dataset)


   .. py:method:: _generate_graph(data_dict)


   .. py:method:: forward(data_dict) -> dict[str, torch.Tensor]


   .. py:method:: _init_gp_partitions(graph_dict, atomic_numbers_full)

      Graph Parallel
      This creates the required partial tensors for each rank given the full tensors.
      The tensors are split on the dimension along the node index using node_partition.



   .. py:property:: num_params


   .. py:method:: no_weight_decay() -> set


.. py:class:: MLP_EFS_Head(backbone, prefix=None, wrap_property=True)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


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


   .. py:attribute:: regress_stress


   .. py:attribute:: regress_forces


   .. py:attribute:: prefix


   .. py:attribute:: wrap_property


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: energy_block


   .. py:method:: forward(data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: AtomicData
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: MLP_Energy_Head(backbone, reduce: str = 'sum')

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


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


   .. py:attribute:: reduce


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: energy_block


   .. py:method:: forward(data_dict, emb: dict[str, torch.Tensor])

      Head forward.

      :param data: Atomic systems as input
      :type data: AtomicData
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: Linear_Energy_Head(backbone, reduce: str = 'sum')

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


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


   .. py:attribute:: reduce


   .. py:attribute:: energy_block


   .. py:method:: forward(data_dict, emb: dict[str, torch.Tensor])

      Head forward.

      :param data: Atomic systems as input
      :type data: AtomicData
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: Linear_Force_Head(backbone)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


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


   .. py:attribute:: linear


   .. py:method:: forward(data_dict, emb: dict[str, torch.Tensor])

      Head forward.

      :param data: Atomic systems as input
      :type data: AtomicData
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:function:: compose_tensor(trace: torch.Tensor, l2_symmetric: torch.Tensor) -> torch.Tensor

   Re-compose a tensor from its decomposition

   :param trace: a tensor with scalar part of the decomposition of r2 tensors in the batch
   :param l2_symmetric: tensor with the symmetric/traceless part of decomposition

   :returns: rank 2 tensor
   :rtype: tensor


.. py:class:: MLP_Stress_Head(backbone, reduce: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


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


   .. py:attribute:: reduce


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: scalar_block


   .. py:attribute:: l2_linear


   .. py:method:: forward(data_dict, emb: dict[str, torch.Tensor])

      Head forward.

      :param data: Atomic systems as input
      :type data: AtomicData
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



