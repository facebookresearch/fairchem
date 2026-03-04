"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unified Radial MLP: Computes all layers' radial functions in a single
batched operation.

Instead of running N separate RadialMLP forward passes:
    for layer in layers:
        radial_out = layer.so2_conv_1.rad_func(x_edge)  # Sequential

We run one batched first layer, then each tail:
    all_radial_outs = unified_radial_mlp(x_edge)  # list of [E, out]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .radial import RadialMLP

__all__ = ["UnifiedRadialMLP", "create_unified_radial_mlp"]

# Expected structure of RadialMLP.net Sequential
_EXPECTED_NET_STRUCTURE = (
    nn.Linear,  # 0: first linear
    nn.LayerNorm,  # 1
    nn.SiLU,  # 2
    nn.Linear,  # 3: second linear
    nn.LayerNorm,  # 4
    nn.SiLU,  # 5
    nn.Linear,  # 6: third linear
)


def _validate_radial_mlp(mlp: RadialMLP, idx: int, reference: RadialMLP | None) -> None:
    """
    Validate a single RadialMLP has expected structure and matches reference.

    Args:
        mlp: The RadialMLP to validate.
        idx: Index in the list (for error messages).
        reference: First RadialMLP to compare dimensions against (None for first).
    """
    # Check layer count
    if len(mlp.net) != 7:
        raise ValueError(f"RadialMLP[{idx}]: expected 7 layers, got {len(mlp.net)}")

    # Check layer types
    for j, expected_type in enumerate(_EXPECTED_NET_STRUCTURE):
        if not isinstance(mlp.net[j], expected_type):
            raise TypeError(
                f"RadialMLP[{idx}].net[{j}]: expected {expected_type.__name__}, "
                f"got {type(mlp.net[j]).__name__}"
            )

    # Check feature dimensions match reference (all MLPs must be identical)
    if reference is not None:
        for j in (0, 3, 6):  # Linear layers
            if mlp.net[j].in_features != reference.net[j].in_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: in_features mismatch "
                    f"({mlp.net[j].in_features} vs {reference.net[j].in_features})"
                )
            if mlp.net[j].out_features != reference.net[j].out_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: out_features mismatch "
                    f"({mlp.net[j].out_features} vs {reference.net[j].out_features})"
                )


class UnifiedRadialMLP(nn.Module):
    """
    Unified radial MLP that batches the first linear layer across all RadialMLPs.

    Includes edge_degree_embedding.rad_func and all layer rad_funcs in a single
    batched GEMM for the first layer. The first layer uses concatenated weights
    (all share the same input). Layers 2+ use indexed functional calls.

    forward() returns [edge_degree_out, layer_0_out, ..., layer_N-1_out].
    """

    def __init__(
        self,
        edge_degree_mlp: RadialMLP,
        layer_radial_mlps: list[RadialMLP],
    ) -> None:
        """
        Initialize from edge_degree and layer RadialMLP modules.

        Args:
            edge_degree_mlp: RadialMLP for edge_degree_embedding.
            layer_radial_mlps: List of RadialMLP modules for each layer.
        """
        super().__init__()

        assert len(layer_radial_mlps) > 0, "Need at least one layer RadialMLP"

        # Validate edge_degree MLP has expected structure
        _validate_radial_mlp(edge_degree_mlp, idx=-1, reference=None)

        # Validate all layer MLPs have expected structure and match each other
        for i, mlp in enumerate(layer_radial_mlps):
            _validate_radial_mlp(mlp, i, layer_radial_mlps[0] if i > 0 else None)

        self.num_layers = len(layer_radial_mlps)
        self.hidden_features = layer_radial_mlps[0].net[0].out_features
        self.edge_hidden_features = edge_degree_mlp.net[0].out_features
        self.ln_eps = layer_radial_mlps[0].net[1].eps
        self.edge_ln_eps = edge_degree_mlp.net[1].eps

        # First layer: concatenated [edge_degree, layer_0, ...] for single GEMM
        self.register_buffer(
            "W1_cat",
            torch.cat(
                [edge_degree_mlp.net[0].weight.data]
                + [mlp.net[0].weight.data for mlp in layer_radial_mlps],
                dim=0,
            ),
        )
        self.register_buffer(
            "b1_cat",
            torch.cat(
                [edge_degree_mlp.net[0].bias.data]
                + [mlp.net[0].bias.data for mlp in layer_radial_mlps],
                dim=0,
            ),
        )

        # Edge degree tail buffers (separate from layers)
        self.register_buffer("edge_ln1_weight", edge_degree_mlp.net[1].weight.data)
        self.register_buffer("edge_ln1_bias", edge_degree_mlp.net[1].bias.data)
        self.register_buffer("edge_fc2_weight", edge_degree_mlp.net[3].weight.data)
        self.register_buffer("edge_fc2_bias", edge_degree_mlp.net[3].bias.data)
        self.register_buffer("edge_ln2_weight", edge_degree_mlp.net[4].weight.data)
        self.register_buffer("edge_ln2_bias", edge_degree_mlp.net[4].bias.data)
        self.register_buffer("edge_fc3_weight", edge_degree_mlp.net[6].weight.data)
        self.register_buffer("edge_fc3_bias", edge_degree_mlp.net[6].bias.data)

        # Layer tail buffers: stacked [N, ...] for indexed access
        self.register_buffer(
            "ln1_weight",
            torch.stack([mlp.net[1].weight.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln1_bias",
            torch.stack([mlp.net[1].bias.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_weight",
            torch.stack([mlp.net[3].weight.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_bias",
            torch.stack([mlp.net[3].bias.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_weight",
            torch.stack([mlp.net[4].weight.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_bias",
            torch.stack([mlp.net[4].bias.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_weight",
            torch.stack([mlp.net[6].weight.data for mlp in layer_radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_bias",
            torch.stack([mlp.net[6].bias.data for mlp in layer_radial_mlps], dim=0),
        )

    def edge_degree_tail(self, h: torch.Tensor) -> torch.Tensor:
        """Apply layers 2+ for edge_degree MLP (LN -> SiLU -> Linear -> ...)."""
        H = self.edge_hidden_features
        h = torch.nn.functional.layer_norm(
            h, (H,), self.edge_ln1_weight, self.edge_ln1_bias, self.edge_ln_eps
        )
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, self.edge_fc2_weight, self.edge_fc2_bias)
        h = torch.nn.functional.layer_norm(
            h, (H,), self.edge_ln2_weight, self.edge_ln2_bias, self.edge_ln_eps
        )
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, self.edge_fc3_weight, self.edge_fc3_bias)

    def layer_radial_tail(self, h: torch.Tensor, i: int) -> torch.Tensor:
        """Apply layers 2+ for layer i (LN -> SiLU -> Linear -> ...)."""
        H = self.hidden_features
        h = torch.nn.functional.layer_norm(
            h, (H,), self.ln1_weight[i], self.ln1_bias[i], self.ln_eps
        )
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, self.fc2_weight[i], self.fc2_bias[i])
        h = torch.nn.functional.layer_norm(
            h, (H,), self.ln2_weight[i], self.ln2_bias[i], self.ln_eps
        )
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, self.fc3_weight[i], self.fc3_bias[i])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute all radial outputs: edge_degree + layer radials.

        Args:
            x: Input tensor of shape [E, in_features]

        Returns:
            List [edge_degree_out, layer_0_out, ..., layer_N-1_out]
        """
        # Single batched GEMM for ALL first layers (edge_degree + N layers)
        h_all = torch.nn.functional.linear(x, self.W1_cat, self.b1_cat)

        # Split: edge_degree chunk first, then layer chunks
        splits = [self.edge_hidden_features] + [self.hidden_features] * self.num_layers
        h_chunks = h_all.split(splits, dim=1)

        # Process edge_degree tail (chunk 0), then layer tails (chunks 1..N)
        results = [self.edge_degree_tail(h_chunks[0])]
        results.extend(
            [self.layer_radial_tail(h_chunks[i + 1], i) for i in range(self.num_layers)]
        )
        return results


def create_unified_radial_mlp(
    edge_degree_mlp: RadialMLP,
    layer_radial_mlps: list[RadialMLP],
) -> UnifiedRadialMLP:
    """
    Factory function to create a UnifiedRadialMLP.

    Args:
        edge_degree_mlp: RadialMLP for edge_degree_embedding.
        layer_radial_mlps: List of RadialMLP modules for each layer.

    Returns:
        UnifiedRadialMLP instance with concatenated first layer weights.
    """
    return UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)
