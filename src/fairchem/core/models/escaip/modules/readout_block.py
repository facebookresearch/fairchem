from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        RegularizationConfigs,
    )
from fairchem.core.models.escaip.utils.graph_utils import (
    compilable_scatter,
)
from fairchem.core.models.escaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_feedforward,
    get_normalization_layer,
)


class ReadoutBlock(nn.Module):
    """
    Readout from each graph attention block for energy and force output
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.backbone_dtype = (
            torch.float16 if global_cfg.use_fp16_backbone else torch.float32
        )

        self.energy_reduce = gnn_cfg.energy_reduce
        self.use_edge_readout = global_cfg.regress_forces and global_cfg.direct_forces
        self.use_global_readout = gnn_cfg.use_global_readout

        # global read out
        if gnn_cfg.use_global_readout:
            self.global_ffn = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
                dropout=reg_cfg.mlp_dropout,
                bias=True,
            ).to(self.backbone_dtype)
            self.pre_global_norm = get_normalization_layer(
                NormalizationType(reg_cfg.normalization)
            )(global_cfg.hidden_size, dtype=self.backbone_dtype)
            # self.post_global_norm = get_normalization_layer(
            #     NormalizationType(reg_cfg.normalization)
            # )(global_cfg.hidden_size, dtype=self.backbone_dtype)

        # node read out
        self.node_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        ).to(self.backbone_dtype)
        self.pre_node_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size, dtype=self.backbone_dtype)
        # self.post_node_norm = get_normalization_layer(
        #     NormalizationType(reg_cfg.normalization)
        # )(global_cfg.hidden_size, dtype=self.backbone_dtype)

        # forces read out
        if self.use_edge_readout:
            self.edge_ffn = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
                dropout=reg_cfg.mlp_dropout,
                bias=True,
            ).to(self.backbone_dtype)
            self.pre_edge_norm = get_normalization_layer(
                NormalizationType(reg_cfg.normalization)
            )(global_cfg.hidden_size, dtype=self.backbone_dtype)
            # self.post_edge_norm = get_normalization_layer(
            #     NormalizationType(reg_cfg.normalization)
            # )(global_cfg.hidden_size, dtype=self.backbone_dtype)

    def forward(self, data, node_features, edge_features):
        """
        Output:
            Global Readout (G, H);
            Node Readout (N, H);
            Edge Readout (N, max_nei, H)
        """
        node_readout = node_features + self.node_ffn(self.pre_node_norm(node_features))
        # node_readout = self.post_node_norm(node_readout)

        if self.use_global_readout:
            global_features = compilable_scatter(
                src=node_features,
                index=data.node_batch,
                dim_size=data.graph_padding_mask.shape[0],
                dim=0,
                reduce=self.energy_reduce,
            )
            global_readout = global_features + self.global_ffn(
                self.pre_global_norm(global_features)
            )
            # global_readout = self.post_global_norm(global_readout)
        else:
            global_readout = torch.zeros_like(node_readout)

        if self.use_edge_readout:
            edge_readout = edge_features + self.edge_ffn(
                self.pre_edge_norm(edge_features)
            )
            # edge_readout = self.post_edge_norm(edge_readout)
        else:
            edge_readout = torch.zeros_like(node_readout)

        return global_readout, node_readout, edge_readout
