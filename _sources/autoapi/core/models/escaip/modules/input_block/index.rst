core.models.escaip.modules.input_block
======================================

.. py:module:: core.models.escaip.modules.input_block


Classes
-------

.. autoapisummary::

   core.models.escaip.modules.input_block.InputBlock
   core.models.escaip.modules.input_block.InputLayer


Module Contents
---------------

.. py:class:: InputBlock(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, molecular_graph_cfg: fairchem.core.models.escaip.configs.MolecularGraphConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

   Bases: :py:obj:`torch.nn.Module`


   Wrapper of InputLayer for adding normalization


   .. py:attribute:: backbone_dtype


   .. py:attribute:: input_layer


   .. py:attribute:: norm_node


   .. py:attribute:: norm_edge


   .. py:method:: forward(inputs: fairchem.core.models.escaip.custom_types.GraphAttentionData)


.. py:class:: InputLayer(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, molecular_graph_cfg: fairchem.core.models.escaip.configs.MolecularGraphConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

   Bases: :py:obj:`fairchem.core.models.escaip.modules.base_block.BaseGraphNeuralNetworkLayer`


   Base class for Graph Neural Network layers.
   Used in InputLayer and EfficientGraphAttention.


   .. py:attribute:: backbone_dtype


   .. py:attribute:: edge_attr_linear


   .. py:attribute:: edge_attr_norm


   .. py:attribute:: edge_ffn


   .. py:method:: forward(inputs: fairchem.core.models.escaip.custom_types.GraphAttentionData)


