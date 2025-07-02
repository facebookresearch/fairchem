from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

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
    get_linear,
    get_normalization_layer,
)
from fairchem.core.models.escaip.utils.stochastic_depth import (
    SkipStochasticDepth,
    StochasticDepth,
)


class EfficientGraphAttentionBlock(nn.Module):
    """
    Efficient Graph Attention Block module.
    Ref: swin transformer
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        is_last: bool = False,
    ):
        super().__init__()

        self.backbone_dtype = (
            torch.float16 if global_cfg.use_fp16_backbone else torch.float32
        )

        # Graph attention
        self.graph_attention = EfficientGraphAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Feed forward network
        self.feedforward = FeedForwardNetwork(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
            is_last=is_last,
        )

        # Normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.norm_attn_node = get_normalization_layer(normalization)(
            global_cfg.hidden_size, dtype=self.backbone_dtype
        )
        self.norm_attn_edge = get_normalization_layer(normalization)(
            global_cfg.hidden_size, dtype=self.backbone_dtype
        )
        self.norm_ffn_node = get_normalization_layer(normalization)(
            global_cfg.hidden_size, dtype=self.backbone_dtype
        )
        if not (
            (is_last) and (global_cfg.regress_forces and not global_cfg.direct_forces)
        ):
            self.norm_ffn_edge = get_normalization_layer(normalization)(
                global_cfg.hidden_size, dtype=self.backbone_dtype
            )
        else:
            self.norm_ffn_edge = nn.Identity()

        # Stochastic depth
        self.stochastic_depth_attn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )
        self.stochastic_depth_ffn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L452
        # x = x + self.stochastic_depth(self.graph_attention(self.norm_attn(x)))
        # x = x + self.stochastic_depth(self.feedforward(self.norm_ffn(x)))

        # attention
        node_hidden, edge_hidden = (
            self.norm_attn_node(node_features),
            self.norm_attn_edge(edge_features),
        )
        node_hidden, edge_hidden = self.graph_attention(data, node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_attn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )

        # feedforward
        node_hidden, edge_hidden = (
            self.norm_ffn_node(node_features),
            self.norm_ffn_edge(edge_features),
        )
        node_hidden, edge_hidden = self.feedforward(node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_ffn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )
        return node_features, edge_features


class EfficientGraphAttention(BaseGraphNeuralNetworkLayer):
    """
    Efficient Graph Attention module.
    """

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

        # Convert repeating_dimensions to Python list for compile-time constants
        # This helps torch.compile recognize these as static values
        self.repeating_dimensions_list = gnn_cfg.freequency_list

        # Also store the length as a constant for looping
        self.rep_dim_len = len(self.repeating_dimensions_list)

        # Register buffer for use in non-compiled contexts
        repeating_dimensions = torch.tensor(gnn_cfg.freequency_list, dtype=torch.long)
        self.register_buffer(
            "repeating_dimensions", repeating_dimensions, persistent=False
        )

        # Pre-calculate the padding size needed for memory-efficient attention
        if gnn_cfg.use_frequency_embedding:
            # Calculate the total dimension of the expanded frequency vectors
            freq_dim = 0
            for _l, rep_count in enumerate(gnn_cfg.freequency_list):
                if rep_count > 0:
                    freq_dim += rep_count * (2 * _l + 1)

            # Calculate padding needed to make it divisible by 8
            padding_size = (8 - freq_dim % 8) % 8
            self.padding_size = padding_size
            self.register_buffer(
                "padding_size_tensor",
                torch.tensor(padding_size, dtype=torch.long),
                persistent=False,
            )

        # Store use_frequency_embedding flag
        self.use_frequency_embedding = gnn_cfg.use_frequency_embedding

        # Edge linear layer
        self.edge_attr_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)
        self.edge_attr_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size)

        # Node hidden layer
        self.node_hidden_linear = self.get_node_linear(global_cfg, reg_cfg).to(
            self.backbone_dtype
        )

        # Edge hidden layer
        self.edge_hidden_linear = get_linear(
            in_features=global_cfg.hidden_size,
            out_features=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        ).to(self.backbone_dtype)

        # message normalization
        self.message_norm = get_normalization_layer(
            NormalizationType(reg_cfg.normalization)
        )(global_cfg.hidden_size, dtype=self.backbone_dtype)

        # message linear
        self.use_message_gate = gnn_cfg.use_message_gate
        if self.use_message_gate:
            self.gate_linear = get_linear(
                in_features=global_cfg.hidden_size * 2,
                out_features=global_cfg.hidden_size,
                activation=None,
                bias=True,
            ).to(self.backbone_dtype)
            self.candidate_linear = get_linear(
                in_features=global_cfg.hidden_size * 2,
                out_features=global_cfg.hidden_size,
                activation=None,
                bias=True,
            ).to(self.backbone_dtype)
        else:
            self.message_linear = get_linear(
                in_features=global_cfg.hidden_size * 3,
                out_features=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                bias=True,
            ).to(self.backbone_dtype)

        # Multi-head attention
        # self.multi_head_attention = nn.MultiheadAttention(
        #     embed_dim=global_cfg.hidden_size,
        #     num_heads=gnn_cfg.atten_num_heads,
        #     dropout=reg_cfg.atten_dropout,
        #     bias=True,
        #     batch_first=True,
        #     dtype=self.backbone_dtype,
        # )
        self.attn_in_proj_q = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        ).to(self.backbone_dtype)
        self.attn_in_proj_k = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        ).to(self.backbone_dtype)
        self.attn_in_proj_v = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        ).to(self.backbone_dtype)
        self.attn_out_proj = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        ).to(self.backbone_dtype)
        self.attn_num_heads = gnn_cfg.atten_num_heads
        self.attn_dropout = reg_cfg.atten_dropout

        # scalar for attention bias
        self.use_angle_embedding = gnn_cfg.use_angle_embedding
        if self.use_angle_embedding == "scalar":
            self.attn_scalar = nn.Parameter(
                torch.ones(gnn_cfg.atten_num_heads, dtype=self.backbone_dtype),
                requires_grad=True,
            )
        elif self.use_angle_embedding == "bias":
            self.attn_distance_embedding = get_linear(
                in_features=gnn_cfg.edge_distance_expansion_size,
                out_features=gnn_cfg.angle_embedding_size,
                activation=Activation(global_cfg.activation),
                bias=True,
                dropout=reg_cfg.mlp_dropout,
            )
            self.attn_angle_embedding = get_linear(
                in_features=gnn_cfg.angle_expansion_size + 1,
                out_features=gnn_cfg.angle_embedding_size,
                activation=Activation(global_cfg.activation),
                bias=True,
                dropout=reg_cfg.mlp_dropout,
            )
            self.attn_bias_projection = get_linear(
                in_features=gnn_cfg.angle_embedding_size**3,
                out_features=gnn_cfg.atten_num_heads,
                activation=None,
                bias=True,
                dropout=reg_cfg.mlp_dropout,
            )
        elif self.use_angle_embedding == "none":
            self.attn_scalar = torch.tensor(1.0, dtype=self.backbone_dtype)
        else:
            raise ValueError(
                f"Invalid use_angle_embedding {gnn_cfg.use_angle_embedding}. Must be one of ['scalar', 'bias', 'none']"
            )

        # Graph attention for aggregation
        # ref: "How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>
        self.use_graph_attention = gnn_cfg.use_graph_attention
        if self.use_graph_attention:
            self.attn_weight = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    gnn_cfg.atten_num_heads,
                    global_cfg.hidden_size // gnn_cfg.atten_num_heads,
                    dtype=self.backbone_dtype,
                ),
                requires_grad=True,
            )
            # glorot initialization
            stdv = math.sqrt(
                6.0 / (self.attn_weight.shape[-2] + self.attn_weight.shape[-1])
            )
            self.attn_weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # Get edge attributes
        edge_attr = self.get_edge_features(data)

        edge_attr = self.edge_attr_linear(edge_attr)
        edge_attr = self.edge_attr_norm(edge_attr)
        edge_attr = edge_attr.to(self.backbone_dtype)

        # Get node features
        node_features = self.get_node_features(node_features, data.neighbor_list)
        node_hidden = self.node_hidden_linear(node_features)

        # Get edge faetures
        edge_hidden = self.edge_hidden_linear(edge_features)

        # Concatenate edge and node features (num_nodes, num_neighbors, hidden_size)
        if self.use_message_gate:
            message = torch.cat([edge_attr, node_hidden], dim=-1)
            update_gate = torch.sigmoid(self.gate_linear(message))
            candidate = torch.tanh(self.candidate_linear(message))
            message = update_gate * candidate + (1 - update_gate) * edge_hidden
        else:
            message = self.message_linear(
                torch.cat([edge_attr, edge_hidden, node_hidden], dim=-1)
            )
        message = self.message_norm(message)

        # Multi-head self-attention
        if self.use_angle_embedding == "bias":
            attn_mask = data.attn_mask + self.get_attn_bias(
                data.angle_embedding, data.edge_distance_expansion
            )
        elif self.use_angle_embedding == "scalar" and data.angle_embedding is not None:
            angle_embedding = data.angle_embedding.reshape(
                -1,
                self.attn_scalar.shape[0],
                data.angle_embedding.shape[-2],
                data.angle_embedding.shape[-1],
            ) * self.attn_scalar.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            angle_embedding = angle_embedding.reshape(
                -1, data.angle_embedding.shape[-2], data.angle_embedding.shape[-1]
            )
            attn_mask = data.attn_mask + angle_embedding
        else:
            attn_mask = data.attn_mask
        # edge_output = self.multi_head_attention(
        #     query=message,
        #     key=message,
        #     value=message,
        #     # key_padding_mask=~data.neighbor_mask,
        #     attn_mask=attn_mask,
        #     need_weights=False,
        # )[0]
        edge_output = self.multi_head_self_attention(
            input=message,
            attn_mask=attn_mask,
            frequency_vectors=data.frequency_vectors,
        )

        # Aggregation
        if self.use_graph_attention:
            node_output, edge_output = self.graph_attention_aggregate(
                edge_output, data.neighbor_mask
            )
        else:
            node_output = self.aggregate(
                edge_output, data.neighbor_mask
            )  ##### TODO: use masked with envolope

        return node_output, edge_output

    def multi_head_self_attention(self, input, attn_mask, frequency_vectors=None):
        # input (num_nodes, num_neighbors, hidden_size)
        # attn_mask (num_nodes * num_heads, num_neighbors, num_neighbors)
        # frequency_vectors: (num_nodes, num_neighbors, sum_{l=0..lmax} rep_l * (2l+1))
        num_nodes, num_neighbors, hidden_dim = input.shape
        head_dim = hidden_dim // self.attn_num_heads
        q = (
            self.attn_in_proj_q(input)
            .reshape(num_nodes, num_neighbors, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.attn_in_proj_k(input)
            .reshape(num_nodes, num_neighbors, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.attn_in_proj_v(input)
            .reshape(num_nodes, num_neighbors, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        # q,k,v (num_nodes, num_heads, num_neighbors, head_dim)

        # Apply frequency embedding if enabled
        if self.use_frequency_embedding and frequency_vectors is not None:
            # Add head dimension to frequency vectors
            # (num_nodes, num_neighbors, sum_{l=0..lmax} rep_l * (2l+1)) ->
            # (num_nodes, 1, num_neighbors, sum_{l=0..lmax} rep_l * (2l+1))
            freq_vecs = frequency_vectors.unsqueeze(1)

            # Create expanded q and k by repeating sections according to repeating_dimensions
            # For each l-value, we repeat the corresponding section 2*l+1 times

            # Lists to collect expanded sections
            q_expanded_sections = []
            k_expanded_sections = []

            # Current position in the head_dim
            curr_pos = 0

            # For each l value - use Python constant for loop range
            for _l in range(self.rep_dim_len):
                # Get repeat count from the Python list - not a tensor
                rep_count = self.repeating_dimensions_list[_l]

                # Skip zero repeats - this is now a static check during compilation
                if rep_count == 0:
                    continue

                # Skip if we've reached the end of the head_dim
                if curr_pos >= head_dim:
                    break

                # Calculate repetition factor for this l: 2*l+1
                sh_dim = 2 * _l + 1

                # End position for this segment
                end_pos = min(curr_pos + rep_count, head_dim)

                # Get the corresponding section from q and k
                q_section = q[
                    ..., curr_pos:end_pos
                ]  # (num_nodes, num_heads, num_neighbors, rep_count)
                k_section = k[
                    ..., curr_pos:end_pos
                ]  # (num_nodes, num_heads, num_neighbors, rep_count)

                # Reshape to prepare for repeating each dimension
                # (num_nodes, num_heads, num_neighbors, rep_count) -> (num_nodes, num_heads, num_neighbors, rep_count, 1)
                q_section = q_section.unsqueeze(-1)
                k_section = k_section.unsqueeze(-1)

                # Repeat each dimension 2*l+1 times
                # (num_nodes, num_heads, num_neighbors, rep_count, 1) -> (num_nodes, num_heads, num_neighbors, rep_count, 2*l+1)
                q_expanded = q_section.expand(-1, -1, -1, -1, sh_dim)
                k_expanded = k_section.expand(-1, -1, -1, -1, sh_dim)

                # Reshape to flatten the last two dimensions
                # (num_nodes, num_heads, num_neighbors, rep_count, 2*l+1) -> (num_nodes, num_heads, num_neighbors, rep_count*(2*l+1))
                q_expanded = q_expanded.reshape(
                    num_nodes, self.attn_num_heads, num_neighbors, -1
                )
                k_expanded = k_expanded.reshape(
                    num_nodes, self.attn_num_heads, num_neighbors, -1
                )

                # Add to our collection
                q_expanded_sections.append(q_expanded)
                k_expanded_sections.append(k_expanded)

                # Move to the next position
                curr_pos = end_pos

            # Only process if we have expanded sections
            if q_expanded_sections:
                # Concatenate the expanded sections
                # [(num_nodes, num_heads, num_neighbors, rep_0*(2*0+1)), (num_nodes, num_heads, num_neighbors, rep_1*(2*1+1)), ...]
                # -> (num_nodes, num_heads, num_neighbors, sum_l rep_l*(2*l+1))
                q = torch.cat(q_expanded_sections, dim=-1)
                k = torch.cat(k_expanded_sections, dim=-1)

                # Apply frequency vectors
                q = q * freq_vecs
                k = k * freq_vecs

                # Pad q and k to make their last dimension divisible by 8 for memory-efficient attention
                # Using Python constant instead of tensor
                if self.padding_size > 0:
                    # Pad the last dimension with zeros
                    q = F.pad(q, (0, self.padding_size))
                    k = F.pad(k, (0, self.padding_size))
                    # Also pad the frequency vectors to match
                    freq_vecs = F.pad(freq_vecs, (0, self.padding_size))

                # Scale appropriately
                # constant = math.sqrt(q.shape[-1] / head_dim)
                # q = q * constant
                # k = k * constant

        # View attention mask
        attn_mask = attn_mask.view(
            num_nodes, self.attn_num_heads, num_neighbors, num_neighbors
        )

        dropout_p = self.attn_dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            num_nodes, num_neighbors, hidden_dim
        )

        attn_output = self.attn_out_proj(attn_output)

        return attn_output

    def get_attn_bias(self, angle_embedding, edge_distance_expansion):
        # legendre_embedding (num_nodes, num_neighbors, num_neighbors, angle_embedding_size)
        legendre_embedding = self.attn_angle_embedding(angle_embedding)
        # edge_distance_expansion (num_nodes, num_neighbors, angle_embedding_size)
        edge_distance_embedding = self.attn_distance_embedding(edge_distance_expansion)
        # edge_dist_outer (num_nodes, num_neighbors, num_neighbors, angle_embedding_size)
        edge_dist_outer = edge_distance_embedding.unsqueeze(
            2
        ) * edge_distance_embedding.unsqueeze(1)
        # edge_dist_outer (num_nodes, num_neighbors, num_neighbors, angle_embedding_size, angle_embedding_size)
        edge_dist_outer = edge_dist_outer.unsqueeze(-1) * edge_dist_outer.unsqueeze(-2)
        # angle_embedding (num_nodes, num_neighbors, num_neighbors, angle_embedding_size, angle_embedding_size, angle_embedding_size)
        angle_embedding = legendre_embedding.unsqueeze(-1).unsqueeze(
            -1
        ) * edge_dist_outer.unsqueeze(-3)
        # angle_embedding (num_nodes, num_neighbors, num_neighbors, angle_embedding_size ** 3)
        angle_embedding = angle_embedding.view(
            angle_embedding.shape[0],
            angle_embedding.shape[1],
            angle_embedding.shape[2],
            -1,
        )
        # attn_bias (num_nodes, num_neighbors, num_neighbors, num_heads)
        attn_bias = self.attn_bias_projection(angle_embedding)
        # attn_bias (num_nodes * num_heads, max_neighbors, max_neighbors)
        attn_bias = attn_bias.permute(0, 3, 1, 2).reshape(
            -1, attn_bias.shape[1], attn_bias.shape[2]
        )
        return attn_bias

    def graph_attention_aggregate(self, edge_output, neighbor_mask):
        # Graph attention for aggregation
        # ref: "How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>

        num_nodes, num_neighbors, _ = edge_output.shape
        _, _, num_heads, head_dim = self.attn_weight.shape

        edge_output = edge_output.view(num_nodes, num_neighbors, num_heads, head_dim)
        # alpha (num_nodes, num_neighbors, num_heads)
        alpha = (edge_output * self.attn_weight).sum(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = alpha.masked_fill(neighbor_mask.unsqueeze(-1) == 0, float("-inf"))
        alpha = F.softmax(alpha, dim=1)
        node_output = (alpha.unsqueeze(-1) * edge_output).sum(1)
        node_output = node_output.view(num_nodes, -1)
        edge_output = edge_output.view(num_nodes, num_neighbors, -1)
        return node_output, edge_output


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        is_last: bool = False,
    ):
        super().__init__()
        self.backbone_dtype = (
            torch.float16 if global_cfg.use_fp16_backbone else torch.float32
        )
        self.mlp_node = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=Activation(global_cfg.activation),
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.node_ffn_dropout,
        ).to(self.backbone_dtype)
        if not (
            (is_last) and (global_cfg.regress_forces and not global_cfg.direct_forces)
        ):
            self.mlp_edge = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=Activation(global_cfg.activation),
                hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
                bias=True,
                dropout=reg_cfg.edge_ffn_dropout,
            ).to(self.backbone_dtype)
        else:
            self.mlp_edge = nn.Identity()

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        return self.mlp_node(node_features), self.mlp_edge(edge_features)
