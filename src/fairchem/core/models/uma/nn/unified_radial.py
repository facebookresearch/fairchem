"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unified Radial MLP: Computes all layers' radial functions in a single
batched operation.

Instead of running 8 separate RadialMLP forward passes:
    for layer in layers:
        radial_out = layer.so2_conv_1.rad_func(x_edge)  # Sequential

We run one batched computation:
    all_radial_outs = unified_radial_mlp(x_edge)  # list of [E, out]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .radial import RadialMLP

__all__ = ["UnifiedRadialMLP", "create_unified_radial_mlp"]


class UnifiedRadialMLP(nn.Module):
    """
    Unified radial MLP that batches computation across multiple layers.

    Takes N RadialMLP modules with identical architecture but different
    weights, and computes all N outputs in a single batched forward pass.

    Architecture (from RadialMLP after MOLE merge):
        Linear(in, hidden) -> LayerNorm -> SiLU
        -> Linear(hidden, hidden) -> LayerNorm -> SiLU
        -> Linear(hidden, out)

    First layer uses concatenated weights for single GEMM since all N
    layers share same input. Layers 2-3 are processed per-layer after
    LayerNorm/SiLU diverge.
    """

    def __init__(self, radial_mlps: list[RadialMLP]) -> None:
        """
        Initialize from a list of RadialMLP modules.

        Args:
            radial_mlps: List of RadialMLP modules with identical
                architecture.
        """
        super().__init__()

        self.num_layers = len(radial_mlps)
        assert self.num_layers > 0, "Need at least one RadialMLP"

        # Validate architecture (must be 7-module / 3-layer MLP)
        first_mlp = radial_mlps[0]
        if len(first_mlp.net) != 7:
            raise ValueError(
                f"Expected 7 modules in RadialMLP, got {len(first_mlp.net)}"
            )

        # Structure: [0]Linear -> [1]LN -> [2]SiLU ->
        #            [3]Linear -> [4]LN -> [5]SiLU -> [6]Linear
        self.in_features = first_mlp.net[0].in_features
        self.hidden_features = first_mlp.net[0].out_features
        self.out_features = first_mlp.net[6].out_features

        # Layer 1: Concatenate all weights into one big linear
        # (since input is shared)
        # W1_cat: [N*hidden, in], b1_cat: [N*hidden]
        W1_list = [mlp.net[0].weight.data for mlp in radial_mlps]
        b1_list = [mlp.net[0].bias.data for mlp in radial_mlps]
        self.register_buffer("W1_cat", torch.cat(W1_list, dim=0))
        self.register_buffer("b1_cat", torch.cat(b1_list, dim=0))

        # LayerNorm 1 params (per-layer)
        self.register_buffer(
            "gamma1",
            torch.stack([mlp.net[1].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "beta1",
            torch.stack([mlp.net[1].bias.data for mlp in radial_mlps], dim=0),
        )

        # Layer 2
        self.register_buffer(
            "W2",
            torch.stack([mlp.net[3].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "b2",
            torch.stack([mlp.net[3].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "gamma2",
            torch.stack([mlp.net[4].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "beta2",
            torch.stack([mlp.net[4].bias.data for mlp in radial_mlps], dim=0),
        )

        # Layer 3 (output)
        self.register_buffer(
            "W3",
            torch.stack([mlp.net[6].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "b3",
            torch.stack([mlp.net[6].bias.data for mlp in radial_mlps], dim=0),
        )

        self.ln_eps = first_mlp.net[1].eps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute all N radial outputs in parallel.

        Args:
            x: Input tensor of shape [E, in_features]

        Returns:
            List of N tensors, each of shape [E, out_features]
        """
        N = self.num_layers
        H = self.hidden_features

        # Layer 1: Single big linear (all N layers share same input x)
        # x: [E, in], W1_cat: [N*hidden, in], b1_cat: [N*hidden]
        # Result: [E, N*hidden] -> reshape to [E, N, hidden]
        h_all = torch.nn.functional.linear(x, self.W1_cat, self.b1_cat)
        h_all = h_all.view(h_all.shape[0], N, H)

        # Split into tuple of [E, hidden] tensors, one per layer
        h_per_layer = h_all.unbind(dim=1)

        # Now process each layer independently for layers 2-3
        outputs = []
        for i in range(N):
            h = h_per_layer[i]  # [E, hidden]

            # LayerNorm 1 + SiLU
            h = torch.nn.functional.layer_norm(
                h, (H,), self.gamma1[i], self.beta1[i], self.ln_eps
            )
            h = torch.nn.functional.silu(h)

            # Layer 2: Linear + LayerNorm + SiLU
            h = torch.nn.functional.linear(h, self.W2[i], self.b2[i])
            h = torch.nn.functional.layer_norm(
                h, (H,), self.gamma2[i], self.beta2[i], self.ln_eps
            )
            h = torch.nn.functional.silu(h)

            # Layer 3: Linear (output)
            h = torch.nn.functional.linear(h, self.W3[i], self.b3[i])
            outputs.append(h)

        return outputs


def create_unified_radial_mlp(
    rad_funcs: list[RadialMLP],
) -> UnifiedRadialMLP:
    """
    Factory function to create a unified radial MLP.

    Args:
        rad_funcs: List of RadialMLP modules to unify.

    Returns:
        Unified radial MLP module.
    """
    return UnifiedRadialMLP(rad_funcs)
