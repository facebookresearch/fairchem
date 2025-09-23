core.models.escaip.utils.graph_utils
====================================

.. py:module:: core.models.escaip.utils.graph_utils


Functions
---------

.. autoapisummary::

   core.models.escaip.utils.graph_utils.get_node_direction_expansion_neighbor
   core.models.escaip.utils.graph_utils.map_sender_receiver_feature
   core.models.escaip.utils.graph_utils.legendre_polynomials
   core.models.escaip.utils.graph_utils.get_compact_frequency_vectors
   core.models.escaip.utils.graph_utils.get_attn_mask
   core.models.escaip.utils.graph_utils.get_attn_mask_env
   core.models.escaip.utils.graph_utils.pad_batch
   core.models.escaip.utils.graph_utils.unpad_results
   core.models.escaip.utils.graph_utils.patch_singleton_atom
   core.models.escaip.utils.graph_utils.compilable_scatter
   core.models.escaip.utils.graph_utils.get_displacement_and_cell


Module Contents
---------------

.. py:function:: get_node_direction_expansion_neighbor(direction_vec: torch.Tensor, neighbor_mask: torch.Tensor, lmax: int)

   Calculate Bond-Orientational Order (BOO) for each node in the graph.
   Ref: Steinhardt, et al. "Bond-orientational order in liquids and glasses." Physical Review B 28.2 (1983): 784.
   Input:
       direction_vec: (num_nodes, num_neighbors, 3)
       neighbor_mask: (num_nodes, num_neighbors)
   :returns: (num_nodes, num_neighbors, lmax + 1)
   :rtype: node_boo


.. py:function:: map_sender_receiver_feature(sender_feature, receiver_feature, neighbor_list)

   Map from node features to edge features.
   sender_feature, receiver_feature: (num_nodes, h)
   neighbor_list: (num_nodes, max_neighbors)
   return: sender_features, receiver_features (num_nodes, max_neighbors, h)


.. py:function:: legendre_polynomials(x: torch.Tensor, lmax: int) -> torch.Tensor

   Compute Legendre polynomials P_0..P_{lmax} for each element in x,
   using the standard recursion:
     P_0(x) = 1
     P_1(x) = x
     (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
   x can have any shape; output will have an extra dimension (lmax+1).


.. py:function:: get_compact_frequency_vectors(edge_direction: torch.Tensor, lmax: int, repeating_dimensions: torch.Tensor | list)

   Calculate a compact representation of frequency vectors.
   :param edge_direction: (N, k, 3) normalized direction vectors
   :param lmax: maximum l value for spherical harmonics
   :param repeating_dimensions: (lmax+1,) tensor or list with repeat counts for each l value

   :returns: (N, k, sum_{l=0..lmax} rep_l * (2l+1))
             flat tensor containing the spherical harmonics matched to repeating dimensions
   :rtype: frequency_vectors


.. py:function:: get_attn_mask(edge_direction: torch.Tensor, neighbor_mask: torch.Tensor, num_heads: int, lmax: int, use_angle_embedding: str)

   :param edge_direction: (num_nodes, max_neighbors, 3)
   :param neighbor_mask: (num_nodes, max_neighbors)
   :param num_heads: number of attention heads


.. py:function:: get_attn_mask_env(src_mask: torch.Tensor, num_heads: int)

   :param src_mask: (num_nodes, num_neighbors)
   :param num_heads: number of attention heads

   Output:
       attn_mask: (num_nodes * num_heads, num_neighbors, num_neighbors)


.. py:function:: pad_batch(max_atoms, max_batch_size, atomic_numbers, node_direction_expansion, edge_distance_expansion, edge_direction, neighbor_list, neighbor_mask, node_batch, num_graphs, src_mask=None)

   Pad the batch to have the same number of nodes in total.
   Needed for torch.compile

   Note: the sampler for multi-node training could sample batchs with different number of graphs.
   The number of sampled graphs could be smaller or larger than the batch size.
   This would cause the model to recompile or core dump.
   Temporarily, setting the max number of graphs to be twice the batch size to mitigate this issue.
   TODO: look into a better way to handle this.


.. py:function:: unpad_results(results, node_padding_mask, graph_padding_mask)

   Unpad the results to remove the padding.


.. py:function:: patch_singleton_atom(edge_direction, neighbor_list, neighbor_mask)

   Patch the singleton atoms in the neighbor_list and neighbor_mask.
   Add a self-loop to the singleton atom


.. py:function:: compilable_scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int = 0, reduce: str = 'sum') -> torch.Tensor

   torch_scatter scatter function with compile support.
   Modified from torch_geometric.utils.scatter_.


.. py:function:: get_displacement_and_cell(data, regress_stress, regress_forces, direct_forces)

   Get the displacement and cell from the data.
   For gradient-based forces/stress
   ref: https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/escn_md.py#L298


