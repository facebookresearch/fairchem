from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class GraphAttentionData:
    """
    Custom dataclass for storing graph data for Graph Attention Networks
    TODO: complete this
    """

    # attributes
    atomic_numbers: torch.Tensor
    charge: torch.Tensor
    spin: torch.Tensor
    edge_distance_expansion: torch.Tensor
    edge_direction: torch.Tensor
    edge_direction_expansion: torch.Tensor
    node_direction_expansion: torch.Tensor
    # neighbor self attention
    src_neighbor_attn_mask: torch.Tensor
    dst_neighbor_attn_mask: torch.Tensor
    src_index: torch.Tensor
    dst_index: torch.Tensor
    frequency_vectors: torch.Tensor | None
    # node self attention
    node_base_attn_mask: torch.Tensor | None
    node_sincx_matrix: torch.Tensor | None
    node_valid_mask: torch.Tensor | None
    # graph structure
    neighbor_index: torch.Tensor
    node_batch: torch.Tensor
    max_batch_size: int
    num_graphs: int
    max_num_nodes: int
    num_nodes: int


def flatten_graph_attention_data_with_spec(data, spec):
    # Flatten based on the in_spec structure
    flat_data = []
    for field_name in spec.context[0]:
        field_value = getattr(data, field_name)
        if isinstance(field_value, torch.Tensor):
            flat_data.append(field_value)
        elif field_value is None:
            flat_data.append(None)
        else:
            # Handle custom types like AttentionBias
            flat_data.extend(field_value.tree_flatten())
    return tuple(flat_data)


torch.export.register_dataclass(
    GraphAttentionData, serialized_type_name="GraphAttentionData"
)
torch.fx._pytree.register_pytree_flatten_spec(  # type: ignore
    GraphAttentionData, flatten_fn_spec=flatten_graph_attention_data_with_spec
)
