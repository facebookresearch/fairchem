from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from fairchem.core.models.escaip.custom_types import GraphAttentionData
from fairchem.core.models.escaip.modules.base_block import BaseGraphNeuralNetworkLayer
from fairchem.core.models.escaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_feedforward,
    get_normalization_layer,
)


class InputBlock(nn.Module):
    """
    Wrapper of InputLayer for adding normalization
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.backbone_dtype = (
            torch.float16 if global_cfg.use_fp16_backbone else torch.float32
        )

        self.input_layer = InputLayer(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        self.norm_node = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size, dtype=self.backbone_dtype)
        self.norm_edge = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size, dtype=self.backbone_dtype)

    def forward(self, inputs: GraphAttentionData):
        node_features, edge_features = self.input_layer(inputs)
        node_features, edge_features = (
            self.norm_node(node_features),
            self.norm_edge(edge_features),
        )
        return node_features, edge_features


class InputLayer(BaseGraphNeuralNetworkLayer):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        self.backbone_dtype = (
            torch.float16 if global_cfg.use_fp16_backbone else torch.float32
        )

        # Edge linear layer
        self.edge_attr_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)
        self.edge_attr_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size)

        # ffn for edge features
        self.edge_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            hidden_layer_multiplier=1,
            dropout=reg_cfg.edge_ffn_dropout,
            bias=True,
        ).to(self.backbone_dtype)

    def forward(self, inputs: GraphAttentionData):
        # Get edge features
        edge_features = self.get_edge_features(inputs)

        # Edge processing
        edge_hidden = self.edge_attr_linear(edge_features)
        edge_hidden = self.edge_attr_norm(edge_hidden)
        edge_hidden = edge_hidden.to(self.backbone_dtype)
        edge_output = edge_hidden + self.edge_ffn(edge_hidden)

        # Aggregation
        node_output = self.aggregate(edge_output, inputs.neighbor_mask)

        # Update inputs
        return node_output, edge_output
