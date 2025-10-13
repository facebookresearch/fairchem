from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from fairchem.core.models.allscaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from fairchem.core.models.allscaip.custom_types import GraphAttentionData

from fairchem.core.models.allscaip.modules.neighborhood_attention import (
    NeighborhoodAttention,
)
from fairchem.core.models.allscaip.modules.node_attention import NodeAttention
from fairchem.core.models.allscaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_feedforward,
    get_normalization_layer,
)


class GraphAttentionBlock(nn.Module):
    """
    Graph Attention Block module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.use_node_path = global_cfg.use_node_path

        # Neighborhood attention
        self.neighborhood_attention = NeighborhoodAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Edge FFN
        self.edge_ffn = FeedForwardNetwork(global_cfg, gnn_cfg, reg_cfg)

        # Node attention
        if global_cfg.use_node_path:
            self.node_attention = NodeAttention(
                global_cfg=global_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg,
            )

        # Node ffn
        self.node_ffn = FeedForwardNetwork(global_cfg, gnn_cfg, reg_cfg)

    def forward(
        self,
        data: GraphAttentionData,
        neighbor_reps: torch.Tensor,
    ):
        # graph messages: (num_nodes, num_neighbors, hidden_dim)

        # 1. neighborhood self attention
        neighbor_reps = self.neighborhood_attention(data, neighbor_reps)

        # get node reps via self-loop
        node_reps = neighbor_reps[:, 0]

        # edge ffn
        edge_reps = self.edge_ffn(neighbor_reps[:, 1:])

        if self.use_node_path:
            # 3. node self attention
            node_reps = self.node_attention(data, node_reps)

        # 4. node ffn
        node_reps = self.node_ffn(node_reps)

        # restore neighbor reps
        neighbor_reps = torch.cat([node_reps.unsqueeze(1), edge_reps], dim=1)

        return neighbor_reps


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.ffn_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size)
        if global_cfg.use_residual_scaling:
            self.ffn_res_scale = torch.nn.Parameter(
                torch.tensor(1 / global_cfg.num_layers), requires_grad=True
            )
        else:
            self.ffn_res_scale = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )

    def forward(self, x: torch.Tensor):
        return self.ffn_res_scale * self.ffn(self.ffn_norm(x)) + x
