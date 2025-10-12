from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from ..configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        #MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from ..custom_types import GraphAttentionData

from .base_attention import (
    BaseAttention,
)
from ..utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
)


class GlobalCrossAttention(BaseAttention):
    """
    Cross Attention module for global features.
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

    def global_queries_node_attention(
        self,
        data: GraphAttentionData,
        node_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        num_graphs, num_global_tokens, hidden_dim = global_reps.shape
        # global_reps: (num_graphs, num_global_tokens, hidden_dim)
        # node_reps: (num_nodes, hidden_dim)
        node_reps_normalized = self.node_norm(node_reps)
        global_reps_normalized = self.global_norm(global_reps)

        # q (num_graphs * num_global_tokens, 1, hidden_dim)
        # k (num_graphs * num_global_tokens, num_nodes, hidden_dim)
        # v (num_graphs * num_global_tokens, num_nodes, hidden_dim)
        q, k, v = self.qkv_projection(
            global_reps_normalized.view(
                num_graphs * num_global_tokens, hidden_dim
            ).unsqueeze(1),
            node_reps_normalized.unsqueeze(0).expand(
                num_graphs * num_global_tokens, -1, -1
            ),
            node_reps_normalized.unsqueeze(0).expand(
                num_graphs * num_global_tokens, -1, -1
            ),
        )

        # get attention output (num_graphs * num_global_tokens, 1, hidden_dim)
        attn_output = self.scaled_dot_product_attention(
            q, k, v, data.global_node_mask[:, None, :, :]
        )

        # output shape: (num_graphs, num_global_tokens, hidden_dim)
        return (
            self.res_scale * attn_output.view(num_graphs, num_global_tokens, -1)
            + global_reps
        )

    def node_queries_global_attention(
        self,
        data: GraphAttentionData,
        node_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        # global_reps: (num_graphs, num_global_tokens, hidden_dim)
        # node_reps: (num_nodes, hidden_dim)
        node_reps_normalized = self.node_norm(node_reps)
        global_reps_normalized = self.global_norm(global_reps)

        # q (1, num_nodes, hidden_dim)
        # k (1, num_graphs * num_global_tokens, hidden_dim)
        # v (1, num_graphs * num_global_tokens, hidden_dim)
        q, k, v = self.qkv_projection(
            node_reps_normalized.unsqueeze(0),
            global_reps_normalized.view(-1, global_reps.shape[-1]).unsqueeze(0),
            global_reps_normalized.view(-1, global_reps.shape[-1]).unsqueeze(0),
        )

        # get attention output (1, num_nodes, hidden_dim)
        attn_output = self.scaled_dot_product_attention(
            q, k, v, data.node_global_mask[:, None, :, :].to(q.dtype)
        )

        return self.res_scale * attn_output.squeeze(0) + node_reps

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
