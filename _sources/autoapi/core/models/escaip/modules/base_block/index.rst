core.models.escaip.modules.base_block
=====================================

.. py:module:: core.models.escaip.modules.base_block


Classes
-------

.. autoapisummary::

   core.models.escaip.modules.base_block.BaseGraphNeuralNetworkLayer


Module Contents
---------------

.. py:class:: BaseGraphNeuralNetworkLayer(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, molecular_graph_cfg: fairchem.core.models.escaip.configs.MolecularGraphConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

   Bases: :py:obj:`torch.nn.Module`


   Base class for Graph Neural Network layers.
   Used in InputLayer and EfficientGraphAttention.


   .. py:attribute:: source_atomic_embedding


   .. py:attribute:: target_atomic_embedding


   .. py:attribute:: source_direction_embedding


   .. py:attribute:: target_direction_embedding


   .. py:attribute:: edge_distance_embedding


   .. py:method:: get_edge_linear(gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)


   .. py:method:: get_node_linear(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)


   .. py:method:: get_edge_features(x: fairchem.core.models.escaip.custom_types.GraphAttentionData) -> torch.Tensor


   .. py:method:: get_node_features(node_features: torch.Tensor, neighbor_list: torch.Tensor) -> torch.Tensor


   .. py:method:: aggregate(edge_features, neighbor_mask)


   .. py:method:: forward()
      :abstractmethod:



