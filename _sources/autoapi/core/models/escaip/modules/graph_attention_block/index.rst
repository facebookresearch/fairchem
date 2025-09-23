core.models.escaip.modules.graph_attention_block
================================================

.. py:module:: core.models.escaip.modules.graph_attention_block


Classes
-------

.. autoapisummary::

   core.models.escaip.modules.graph_attention_block.EfficientGraphAttentionBlock
   core.models.escaip.modules.graph_attention_block.EfficientGraphAttention
   core.models.escaip.modules.graph_attention_block.FeedForwardNetwork


Module Contents
---------------

.. py:class:: EfficientGraphAttentionBlock(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, molecular_graph_cfg: fairchem.core.models.escaip.configs.MolecularGraphConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs, is_last: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Efficient Graph Attention Block module.
   Ref: swin transformer


   .. py:attribute:: backbone_dtype


   .. py:attribute:: graph_attention


   .. py:attribute:: feedforward


   .. py:attribute:: norm_attn_node


   .. py:attribute:: norm_attn_edge


   .. py:attribute:: norm_ffn_node


   .. py:attribute:: stochastic_depth_attn


   .. py:attribute:: stochastic_depth_ffn


   .. py:method:: forward(data: fairchem.core.models.escaip.custom_types.GraphAttentionData, node_features: torch.Tensor, edge_features: torch.Tensor)


.. py:class:: EfficientGraphAttention(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, molecular_graph_cfg: fairchem.core.models.escaip.configs.MolecularGraphConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

   Bases: :py:obj:`fairchem.core.models.escaip.modules.base_block.BaseGraphNeuralNetworkLayer`


   Efficient Graph Attention module.


   .. py:attribute:: backbone_dtype


   .. py:attribute:: repeating_dimensions_list


   .. py:attribute:: rep_dim_len


   .. py:attribute:: use_frequency_embedding


   .. py:attribute:: edge_attr_linear


   .. py:attribute:: edge_attr_norm


   .. py:attribute:: node_hidden_linear


   .. py:attribute:: edge_hidden_linear


   .. py:attribute:: message_norm


   .. py:attribute:: use_message_gate


   .. py:attribute:: attn_in_proj_q


   .. py:attribute:: attn_in_proj_k


   .. py:attribute:: attn_in_proj_v


   .. py:attribute:: attn_out_proj


   .. py:attribute:: attn_num_heads


   .. py:attribute:: attn_dropout


   .. py:attribute:: use_angle_embedding


   .. py:attribute:: use_graph_attention


   .. py:method:: forward(data: fairchem.core.models.escaip.custom_types.GraphAttentionData, node_features: torch.Tensor, edge_features: torch.Tensor)


   .. py:method:: multi_head_self_attention(input, attn_mask, frequency_vectors=None)


   .. py:method:: get_attn_bias(angle_embedding, edge_distance_expansion)


   .. py:method:: graph_attention_aggregate(edge_output, neighbor_mask)


.. py:class:: FeedForwardNetwork(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs, is_last: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Feed Forward Network module.


   .. py:attribute:: backbone_dtype


   .. py:attribute:: mlp_node


   .. py:method:: forward(node_features: torch.Tensor, edge_features: torch.Tensor)


