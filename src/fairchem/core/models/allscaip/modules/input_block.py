from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fairchem.core.models.allscaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
        RegularizationConfigs,
    )
    from fairchem.core.models.allscaip.custom_types import GraphAttentionData

from fairchem.core.models.allscaip.utils.nn_utils import (
    Activation,
    NormalizationType,
    get_linear,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.embedding_dev import ChgSpinEmbedding


class InputBlock(nn.Module):
    """
    Featurize the input data into edge and global embeddings.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Atomic number embeddings
        # ref: escn https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/escn/escn.py#L823
        self.atomic_embedding = nn.Embedding(
            molecular_graph_cfg.max_num_elements, global_cfg.hidden_size
        )
        nn.init.uniform_(self.atomic_embedding.weight.data, -0.001, 0.001)

        # Node direction embedding
        self.node_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.node_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.node_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size * 2)

        # Edge attribute linear
        self.edge_attr_linear = get_linear(
            in_features=gnn_cfg.edge_distance_expansion_size
            + (gnn_cfg.edge_direction_expansion_size) ** 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )
        self.edge_feature_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )

        # charge / spin embedding
        self.charge_embedding = ChgSpinEmbedding(
            "rand_emb",
            "charge",
            global_cfg.hidden_size,
            grad=True,
        )
        self.spin_embedding = ChgSpinEmbedding(
            "rand_emb",
            "spin",
            global_cfg.hidden_size,
            grad=True,
        )
        self.charge_spin_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
        )

    def forward(self, data: GraphAttentionData):
        # neighbor embeddings
        atomic_embedding = self.atomic_embedding(data.atomic_numbers)
        node_direction_embedding = self.node_direction_embedding(
            data.node_direction_expansion
        )
        node_embeddings = torch.cat(
            [atomic_embedding, node_direction_embedding], dim=-1
        )
        node_embeddings = self.node_linear(self.node_norm(node_embeddings))
        # node_embeddings: (num_nodes, hidden_dim)

        # charge / spin embedding
        charge_embedding = self.charge_embedding(data.charge)
        spin_embedding = self.spin_embedding(data.spin)
        charge_spin_embeddings = self.charge_spin_linear(
            torch.cat([charge_embedding, spin_embedding], dim=-1)
        )
        # charge_spin_embeddings: (num_graphs, hidden_dim)
        node_embeddings = node_embeddings + charge_spin_embeddings[data.node_batch]

        # neighbor embedding
        edge_attr = self.edge_attr_linear(
            torch.cat(
                [data.edge_distance_expansion, data.edge_direction_expansion], dim=-1
            )
        )
        neighbor_embeddings = self.edge_feature_linear(
            torch.cat([node_embeddings[data.neighbor_index[0]], edge_attr], dim=-1)
        )

        return neighbor_embeddings
