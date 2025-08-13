from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        RegularizationConfigs,
    )
from fairchem.core.models.escaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_feedforward,
    get_linear,
    get_normalization_layer,
)


class OutputProjection(nn.Module):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.use_edge_readout = global_cfg.regress_forces and global_cfg.direct_forces
        self.use_global_readout = gnn_cfg.use_global_readout

        # map concatenated readout features to hidden size
        self.node_projection = get_linear(
            in_features=global_cfg.hidden_size * (global_cfg.num_layers + 1),
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.output_norm_node = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size)

        if self.use_edge_readout:
            self.edge_projection = get_linear(
                in_features=global_cfg.hidden_size * (global_cfg.num_layers + 1),
                out_features=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                bias=True,
            )
            self.output_norm_edge = get_normalization_layer(
                NormalizationType(reg_cfg.normalization)
            )(global_cfg.hidden_size)

        if self.use_global_readout:
            self.global_projection = get_linear(
                in_features=global_cfg.hidden_size * (global_cfg.num_layers + 1),
                out_features=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                bias=True,
            )
            self.output_norm_global = get_normalization_layer(
                NormalizationType(reg_cfg.normalization)
            )(global_cfg.hidden_size)

    def forward(self, data, global_readouts, node_readouts, edge_readouts):
        node_features = self.node_projection(node_readouts)
        node_features = self.output_norm_node(node_features)

        if self.use_global_readout:
            global_features = self.global_projection(global_readouts)
            global_features = self.output_norm_global(global_features)
        else:
            global_features = None

        if self.use_edge_readout:
            edge_features = self.edge_projection(edge_readouts)
            edge_features = self.output_norm_edge(edge_features)
        else:
            edge_features = None
        return global_features, node_features, edge_features


class OutputLayer(nn.Module):
    """
    Get the final prediction from the readouts (force or energy)
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        output_type: Literal["Vector", "Scalar"],
    ):
        super().__init__()

        self.output_type = output_type
        output_type_dict = {
            "Vector": 3,
            "Scalar": 1,
        }
        assert output_type in output_type_dict, f"Invalid output type {output_type}"

        # mlp
        self.ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            hidden_layer_multiplier=gnn_cfg.output_hidden_layer_multiplier,
            dropout=reg_cfg.scalar_output_dropout
            if output_type == "Scalar"
            else reg_cfg.vector_output_dropout,
            bias=True,
        )

        # normalization
        # self.norm = get_normalization_layer(reg_cfg.normalization)(
        #     global_cfg.hidden_size
        # )

        # final output layer
        self.final_output = get_linear(
            in_features=global_cfg.hidden_size,
            out_features=output_type_dict[output_type],
            activation=None,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: features from the backbone
        Shape ([num_nodes, hidden_size] or [num_nodes, max_neighbor, hidden_size])
        """
        # mlp
        features = self.ffn(features)

        # final output layer
        return self.final_output(features)
