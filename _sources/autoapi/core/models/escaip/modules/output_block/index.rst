core.models.escaip.modules.output_block
=======================================

.. py:module:: core.models.escaip.modules.output_block


Classes
-------

.. autoapisummary::

   core.models.escaip.modules.output_block.OutputProjection
   core.models.escaip.modules.output_block.OutputLayer


Module Contents
---------------

.. py:class:: OutputProjection(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

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


   .. py:attribute:: use_edge_readout


   .. py:attribute:: use_global_readout


   .. py:attribute:: node_projection


   .. py:attribute:: output_norm_node


   .. py:method:: forward(data, global_readouts, node_readouts, edge_readouts)


.. py:class:: OutputLayer(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs, output_type: Literal['Vector', 'Scalar'])

   Bases: :py:obj:`torch.nn.Module`


   Get the final prediction from the readouts (force or energy)


   .. py:attribute:: output_type


   .. py:attribute:: ffn


   .. py:attribute:: final_output


   .. py:method:: forward(features: torch.Tensor) -> torch.Tensor

      features: features from the backbone
      Shape ([num_nodes, hidden_size] or [num_nodes, max_neighbor, hidden_size])



