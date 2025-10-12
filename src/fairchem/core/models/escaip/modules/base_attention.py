from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        RegularizationConfigs,
    )


class BaseAttention(nn.Module):
    """
    Base Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Multi-head attention
        self.attn_in_proj_q = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        )
        self.attn_in_proj_k = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        )
        self.attn_in_proj_v = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        )
        self.attn_out_proj = nn.Linear(
            global_cfg.hidden_size,
            global_cfg.hidden_size,
            bias=True,
        )
        self.attn_num_heads = gnn_cfg.atten_num_heads
        self.attn_dropout = reg_cfg.atten_dropout

    def qkv_projection(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, q_seq_len, hidden_dim = q.shape
        vk_seq_len = k.shape[1]
        head_dim = hidden_dim // self.attn_num_heads
        q = (
            self.attn_in_proj_q(q)
            .reshape(batch_size, q_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.attn_in_proj_k(k)
            .reshape(batch_size, vk_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.attn_in_proj_v(v)
            .reshape(batch_size, vk_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        return q, k, v

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_heads, _, head_dim = v.shape
        q_seq_len = q.shape[2]
        hidden_dim = num_heads * head_dim

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=1 / math.sqrt(v.size(-1)),
        )

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, q_seq_len, hidden_dim
        )

        attn_output = self.attn_out_proj(attn_output)

        return attn_output
