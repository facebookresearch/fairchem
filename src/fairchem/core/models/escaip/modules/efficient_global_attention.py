from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from ..configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from ..custom_types import GraphAttentionData

from fairchem.experimental.ericqu.src.modules.base_attention import (
    BaseAttention,
)
from fairchem.experimental.ericqu.src.utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
)


class EfficientGlobalCrossAttention(BaseAttention):
    """
    An efficient implementation of Cross Attention module for global features.
    It avoids materializing large tensors from `expand` operations.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        attn_type: Literal["global_queries_node", "node_queries_global"],
    ):
        super().__init__(global_cfg, gnn_cfg, reg_cfg)
        self.attn_type = attn_type
        normalization = NormalizationType(reg_cfg.normalization)
        self.node_norm = get_normalization_layer(normalization)(global_cfg.hidden_size)
        self.global_norm = get_normalization_layer(normalization)(
            global_cfg.hidden_size
        )
        # residual scaling
        if global_cfg.use_residual_scaling:
            self.res_scale = torch.nn.Parameter(
                torch.tensor(1 / global_cfg.num_layers), requires_grad=True
            )
        else:
            self.res_scale = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )

    def _get_qkv(self, q_in, k_in, v_in):
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)

        q = q.view(
            q.shape[0], q.shape[1], self.attn_num_heads, self.attn_head_dim
        ).transpose(1, 2)
        k = k.view(
            k.shape[0], k.shape[1], self.attn_num_heads, self.attn_head_dim
        ).transpose(1, 2)
        v = v.view(
            v.shape[0], v.shape[1], self.attn_num_heads, self.attn_head_dim
        ).transpose(1, 2)
        return q, k, v

    def global_queries_node_attention(
        self,
        data: GraphAttentionData,
        node_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        num_graphs, num_global_tokens, _ = global_reps.shape
        node_reps_normalized = self.node_norm(node_reps)
        global_reps_normalized = self.global_norm(global_reps)

        # q: (num_graphs, num_heads, num_global_tokens, head_dim)
        # k: (1, num_heads, num_nodes, head_dim)
        # v: (1, num_heads, num_nodes, head_dim)
        q, k, v = self._get_qkv(
            global_reps_normalized,
            node_reps_normalized.unsqueeze(0),
            node_reps_normalized.unsqueeze(0),
        )

        # attn_mask: (num_graphs, num_global_tokens, num_nodes) -> (num_graphs, 1, num_global_tokens, num_nodes)
        attn_mask = data.global_node_mask.view(
            num_graphs, num_global_tokens, -1
        ).unsqueeze(1)

        # attn_output: (num_graphs, num_heads, num_global_tokens, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).reshape(
            num_graphs, num_global_tokens, -1
        )
        attn_output = self.out_proj(attn_output)

        return self.res_scale * attn_output + global_reps

    def node_queries_global_attention(
        self,
        data: GraphAttentionData,
        node_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        num_graphs, num_global_tokens, _ = global_reps.shape
        node_reps_normalized = self.node_norm(node_reps)
        global_reps_normalized = self.global_norm(global_reps)

        # q: (1, num_heads, num_nodes, head_dim)
        # k: (num_graphs, num_heads, num_global_tokens, head_dim)
        # v: (num_graphs, num_heads, num_global_tokens, head_dim)
        q, k, v = self._get_qkv(
            node_reps_normalized.unsqueeze(0),
            global_reps_normalized,
            global_reps_normalized,
        )

        # attn_mask: (num_nodes, num_graphs, num_global_tokens) -> (num_graphs, 1, num_nodes, num_global_tokens)
        attn_mask = (
            data.node_global_mask.view(
                data.num_nodes, num_graphs, num_global_tokens
            )
            .permute(1, 0, 2)
            .unsqueeze(1)
        )

        # attn_output: (num_graphs, num_heads, num_nodes, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # Gather the results for each node from its corresponding graph
        attn_output = attn_output[data.node_batch]  # (num_nodes, num_heads, num_nodes, head_dim)
        # The query for each node is at the same index as the node in the sequence
        attn_output = torch.diagonal(attn_output, dim1=1, dim2=2).permute(0, 2, 1) # (num_nodes, head_dim, num_heads)
        attn_output = attn_output.reshape(data.num_nodes, -1) # (num_nodes, hidden_dim)

        attn_output = self.out_proj(attn_output)

        return self.res_scale * attn_output + node_reps

    def forward(
        self,
        data: GraphAttentionData,
        node_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        if self.attn_type == "global_queries_node":
            return self.global_queries_node_attention(data, node_reps, global_reps)
        elif self.attn_type == "node_queries_global":
            return self.node_queries_global_attention(data, node_reps, global_reps)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}") 