from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ..configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from ..custom_types import GraphAttentionData

from .global_attention import (
    GlobalCrossAttention,
)
from .neighborhood_attention import (
    NeighborhoodAttention,
)
from .node_attention import NodeAttention
from ..utils.nn_utils import (
    Activation,
    NormalizationType,
    get_feedforward,
    get_normalization_layer,
)


def print_debug(msg, data, neighbor_reps=None, global_reps=None, node_reps=None):
    print(msg)
    if global_reps is not None:
        print(
            "graph_globals: ",
            global_reps[: data.num_graphs].mean().item(),
            global_reps[: data.num_graphs].std().item(),
        )
    if neighbor_reps is not None:
        print(
            "neighbor_reps: ",
            neighbor_reps[: data.num_nodes].mean().item(),
            neighbor_reps[: data.num_nodes].std().item(),
        )
    if node_reps is not None:
        print(
            "node_reps: ",
            node_reps[: data.num_nodes].mean().item(),
            node_reps[: data.num_nodes].std().item(),
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

        self.use_global_path = global_cfg.use_global_path
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

        # node ffn
        self.node_ffn = FeedForwardNetwork(global_cfg, gnn_cfg, reg_cfg)

        if global_cfg.use_global_path:
            # Global cross attention
            self.node_queries_global_cross_attention = GlobalCrossAttention(
                global_cfg=global_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg,
                attn_type="node_queries_global",
            )
            self.global_queries_node_cross_attention = GlobalCrossAttention(
                global_cfg=global_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg,
                attn_type="global_queries_node",
            )
            # global ffn
            self.global_ffn = FeedForwardNetwork(global_cfg, gnn_cfg, reg_cfg)

    def forward(
        self,
        data: GraphAttentionData,
        neighbor_reps: torch.Tensor,
        global_reps: torch.Tensor,
    ):
        # graph messages: (num_nodes, num_neighbors, hidden_dim)
        # graph globals: (num_graphs, num_global_tokens, hidden_dim)

        # print("===================New Block========================")
        # print_debug("before nei attn", data, global_reps, None, neighbor_reps)
        # 1. neighborhood self attention
        neighbor_reps = self.neighborhood_attention(data, neighbor_reps)
        # print_debug("after nei attn", data, None, None, neighbor_reps)

        # get node reps
        node_reps = neighbor_reps[:, 0]

        # edge ffn
        edge_reps = self.edge_ffn(neighbor_reps[:, 1:])
        # print_debug("after edge ffn", data, None, None, edge_reps)

        if self.use_global_path:
            # 2. node to global cross attention (inject global information to node)
            node_reps = self.node_queries_global_cross_attention(
                data, node_reps, global_reps
            )
            # print_debug("after node to global cross attn", data, None, node_reps)

        if self.use_node_path:
            # 3. node self attention
            node_reps = self.node_attention(data, node_reps)
            # print_debug("after node self attn", data, None, node_reps)

        # 4. node ffn
        node_reps = self.node_ffn(node_reps)
        # print_debug("after node ffn", data, None, node_reps)
        #print("node reps shape:", node_reps.shape)
        #print("edge reps shape:", edge_reps.shape)
        
        # restore neighbor reps
        neighbor_reps = torch.cat([node_reps.unsqueeze(1), edge_reps], dim=1)
        # print_debug("after restore neighbor reps", data, None, None, neighbor_reps)

        if self.use_global_path:
            # 5. global to node cross attention (refresh global summary)
            global_reps = self.global_queries_node_cross_attention(
                data, node_reps, global_reps
            )

            # print_debug("after global to node cross attn", data, global_reps, None)
            # 6. global ffn
            global_reps = self.global_ffn(global_reps)
            # print_debug("after global ffn", data, global_reps, None)

        # print_debug("after block", data, global_reps, None, neighbor_reps)
        return neighbor_reps, global_reps


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
            dropout=reg_cfg.node_ffn_dropout,
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
