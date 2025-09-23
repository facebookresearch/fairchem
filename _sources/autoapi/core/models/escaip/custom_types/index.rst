core.models.escaip.custom_types
===============================

.. py:module:: core.models.escaip.custom_types


Classes
-------

.. autoapisummary::

   core.models.escaip.custom_types.GraphAttentionData


Functions
---------

.. autoapisummary::

   core.models.escaip.custom_types.map_graph_attention_data_to_device
   core.models.escaip.custom_types.flatten_graph_attention_data_with_spec


Module Contents
---------------

.. py:class:: GraphAttentionData

   Custom dataclass for storing graph data for Graph Attention Networks
   atomic_numbers: (N)
   edge_distance_expansion: (N, max_nei, edge_distance_expansion_size)
   edge_direction: (N, max_nei, 3)
   node_direction_expansion: (N, node_direction_expansion_size)
   attn_mask: (N * num_head, max_nei, max_nei) Attention mask with angle embeddings
   angle_embedding: (N * num_head, max_nei, max_nei) Angle embeddings (cosine)
   frequency_vectors: (N, max_nei, head_dim, 2l+1) Frequency embeddings
   neighbor_list: (N, max_nei)
   neighbor_mask: (N, max_nei)
   node_batch: (N)
   node_padding_mask: (N)
   graph_padding_mask: (num_graphs)


   .. py:attribute:: atomic_numbers
      :type:  torch.Tensor


   .. py:attribute:: edge_distance_expansion
      :type:  torch.Tensor


   .. py:attribute:: edge_direction
      :type:  torch.Tensor


   .. py:attribute:: node_direction_expansion
      :type:  torch.Tensor


   .. py:attribute:: attn_mask
      :type:  torch.Tensor


   .. py:attribute:: angle_embedding
      :type:  torch.Tensor | None


   .. py:attribute:: frequency_vectors
      :type:  torch.Tensor | None


   .. py:attribute:: neighbor_list
      :type:  torch.Tensor


   .. py:attribute:: neighbor_mask
      :type:  torch.Tensor


   .. py:attribute:: node_batch
      :type:  torch.Tensor


   .. py:attribute:: node_padding_mask
      :type:  torch.Tensor


   .. py:attribute:: graph_padding_mask
      :type:  torch.Tensor


.. py:function:: map_graph_attention_data_to_device(data: GraphAttentionData, device: torch.device | str) -> GraphAttentionData

   Map all tensor fields in GraphAttentionData to the specified device.


.. py:function:: flatten_graph_attention_data_with_spec(data, spec)

