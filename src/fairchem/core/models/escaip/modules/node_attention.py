from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

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


class NodeAttention(BaseAttention):
    """
    Node Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__(global_cfg, gnn_cfg, reg_cfg)

        # Initialize Euclidean Rotary Encoding
        if gnn_cfg.use_sincx_mask:
            self.radial_weight = torch.nn.Parameter(
                torch.ones(gnn_cfg.atten_num_heads, gnn_cfg.attn_num_freq)
                / gnn_cfg.attn_num_freq,
                requires_grad=True,
            )
        else:
            self.radial_weight = None

        normalization = NormalizationType(reg_cfg.normalization)
        self.node_norm = get_normalization_layer(normalization)(global_cfg.hidden_size)

        # residual scaling
        if global_cfg.use_residual_scaling:
            self.node_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1 / global_cfg.num_layers), requires_grad=True
            )
        else:
            self.node_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )
        self.use_sincx_mask = gnn_cfg.use_sincx_mask
        self.num_heads = gnn_cfg.atten_num_heads

    def forward(self, data: GraphAttentionData, node_reps: torch.Tensor):
        # node_reps: (num_nodes, hidden_dim)
        node_reps_normalized = self.node_norm(node_reps)
        node_attn_mask = self.get_node_attention_mask(data, self.radial_weight)

        # get q, k, v (1, num_nodes, num_heads, hidden_dim)
        q, k, v = self.qkv_projection(
            node_reps_normalized.unsqueeze(0),
            node_reps_normalized.unsqueeze(0),
            node_reps_normalized.unsqueeze(0),
        )

        # get attention output
        attn_output = self.scaled_dot_product_attention(
            q, k, v, node_attn_mask[None, :, :, :].to(q.dtype)
        )

        # output shape: (num_nodes, hidden_dim)
        return self.node_attn_res_scale * attn_output.squeeze(0) + node_reps

    def get_node_attention_mask(
        self,
        data: GraphAttentionData,
        radial_weight: torch.Tensor,
        eps: float = 1e-6,
        normalize: bool = True,
    ):
        if self.use_sincx_mask:
            # mix frequency and radial weight
            # (num_heads, num_nodes, num_nodes)
            freq_weight = torch.einsum("ijk,hk->hij", data.node_sincx_matrix, radial_weight)

            # TODO: double check when there is only one graph with no padding
            freq_weight = freq_weight.masked_fill(~data.node_valid_mask.unsqueeze(0), 0)

            # Positive
            freq_weight = F.softplus(freq_weight) + eps

            # normalize
            if normalize:
                denom = freq_weight.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
                freq_weight = freq_weight * (data.node_valid_mask.sum() / denom)

            attn_bias = freq_weight.log()

            return attn_bias + data.node_base_attn_mask.expand(self.num_heads, -1, -1)

        return data.node_base_attn_mask.expand(self.num_heads, -1, -1)
