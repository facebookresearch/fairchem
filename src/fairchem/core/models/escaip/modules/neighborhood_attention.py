from __future__ import annotations

from typing import TYPE_CHECKING

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

from .base_attention import (
    BaseAttention,
)
from ..utils.nn_utils import (
    #Activation,
    NormalizationType,
    #get_linear,
    get_normalization_layer,
)


class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.attn_num_heads = gnn_cfg.atten_num_heads
        self.src_attn = BaseAttention(global_cfg, gnn_cfg, reg_cfg)
        self.dst_attn = BaseAttention(global_cfg, gnn_cfg, reg_cfg)

        # Initialize Frequency embedding
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
        # Calculate the total dimension of the expanded frequency vectors
        freq_dim = 0
        for _l, rep_count in enumerate(gnn_cfg.freequency_list):
            if rep_count > 0:
                freq_dim += rep_count * (2 * _l + 1)

        # Calculate padding needed to make it divisible by 8
        # Required for memory-efficient attention
        padding_size = (8 - freq_dim % 8) % 8
        self.padding_size = padding_size
        self.register_buffer(
            "padding_size_tensor",
            torch.tensor(padding_size, dtype=torch.long),
            persistent=False,
        )

        # normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.src_attn_norm = get_normalization_layer(normalization)(
            global_cfg.hidden_size
        )
        self.dst_attn_norm = get_normalization_layer(normalization)(
            global_cfg.hidden_size
        )

        # residual scaling
        if global_cfg.use_residual_scaling:
            self.src_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1 / global_cfg.num_layers), requires_grad=True
            )
            self.dst_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1 / global_cfg.num_layers), requires_grad=True
            )
        else:
            self.src_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )
            self.dst_attn_res_scale = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )

        self.use_freq_mask = gnn_cfg.use_freq_mask


    def forward(
        self,
        data: GraphAttentionData,
        neighbor_reps: torch.Tensor,
    ):
        """
        source and destination neighborhoodattention
        attn_mask: (num_nodes, num_neighbors, num_neighbors)
        frequency_vectors: (num_nodes, num_neighbors, sum_{l=0..lmax} rep_l * (2l+1))
        neighbor_reps: (num_nodes, num_neighbors, hidden_dim)
        """

        # Source neighborhood attention
        neighbor_reps = (
            neighbor_reps
            + self.src_attn_res_scale
            * self.multi_head_self_attention(
                attn_module=self.src_attn,
                input=self.src_attn_norm(neighbor_reps),
                attn_mask=data.src_neighbor_attn_mask[:, None, None, :],
                frequency_vectors=data.frequency_vectors,
            )
        )

        # Change Index
        neighbor_reps = neighbor_reps[data.dst_index[0], data.dst_index[1]]
        frequency_vectors_dst = data.frequency_vectors[
            data.dst_index[0], data.dst_index[1]
        ]

        # Destination neighborhood attention
        neighbor_reps = (
            neighbor_reps
            + self.dst_attn_res_scale
            * self.multi_head_self_attention(
                attn_module=self.dst_attn,
                input=self.dst_attn_norm(neighbor_reps),
                attn_mask=data.dst_neighbor_attn_mask[:, None, None, :],
                frequency_vectors=frequency_vectors_dst,
            )
        )

        # Change Index
        neighbor_reps = neighbor_reps[data.src_index[0], data.src_index[1]]

        return neighbor_reps

    def multi_head_self_attention(
        self,
        attn_module: BaseAttention,
        input: torch.Tensor,
        attn_mask: torch.Tensor,
        frequency_vectors: torch.Tensor,
    ):
        # input (num_nodes, num_neighbors, hidden_size)
        # attn_mask (num_nodes * num_heads, num_neighbors, num_neighbors)
        # frequency_vectors: (num_nodes, num_neighbors, sum_{l=0..lmax} rep_l * (2l+1))

        # q,k,v (num_nodes, num_neighbors, num_heads, head_dim)
        q, k, v = attn_module.qkv_projection(input, input, input)

        # apply frequency embedding
        if self.use_freq_mask:
            q, k = self.apply_frequency_embedding(q, k, frequency_vectors)

        # get attention output
        attn_output = attn_module.scaled_dot_product_attention(q, k, v, attn_mask)

        # output shape: (num_nodes, num_neighbors, hidden_dim)
        return attn_output

    def apply_frequency_embedding(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frequency_vectors: torch.Tensor,
    ):
        num_nodes, _, num_neighbors, head_dim = q.shape

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
        return q, k
