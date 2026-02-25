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

__all__ = ["UnifiedRadialMLP"]

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
    Unified radial MLP that batches the first linear layer across N RadialMLPs.

    The first layer uses concatenated weights for a single GEMM (all N layers
    share the same input). Layers 2+ use stacked weight buffers for fast
    indexed functional calls.
    """

    def __init__(self, radial_mlps: list[RadialMLP]) -> None:
        """
        Initialize from a list of RadialMLP modules.

        Args:
            radial_mlps: List of RadialMLP modules with identical architecture.
        """
        super().__init__()

        assert len(radial_mlps) > 0, "Need at least one RadialMLP"

        # Validate all MLPs have expected structure and match each other
        for i, mlp in enumerate(radial_mlps):
            _validate_radial_mlp(mlp, i, radial_mlps[0] if i > 0 else None)

        self.num_layers = len(radial_mlps)
        self.hidden_features = radial_mlps[0].net[0].out_features
        self.ln_eps = radial_mlps[0].net[1].eps

        # First layer: concatenated for single GEMM
        self.register_buffer(
            "W1_cat",
            torch.cat([mlp.net[0].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "b1_cat",
            torch.cat([mlp.net[0].bias.data for mlp in radial_mlps], dim=0),
        )

        # Remaining layers: stacked [N, ...] for indexed access
        self.register_buffer(
            "ln1_weight",
            torch.stack([mlp.net[1].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln1_bias",
            torch.stack([mlp.net[1].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_weight",
            torch.stack([mlp.net[3].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_bias",
            torch.stack([mlp.net[3].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_weight",
            torch.stack([mlp.net[4].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_bias",
            torch.stack([mlp.net[4].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_weight",
            torch.stack([mlp.net[6].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_bias",
            torch.stack([mlp.net[6].bias.data for mlp in radial_mlps], dim=0),
        )

    def umas_radial_mlp(self, h: torch.Tensor, i: int) -> torch.Tensor:
        """Apply layers 2+ (LN -> SiLU -> Linear -> LN -> SiLU -> Linear)."""
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
        Compute all N radial outputs.

        Args:
            x: Input tensor of shape [E, in_features]

        Returns:
            List of N tensors, each of shape [E, out_features]
        """
        # Single batched GEMM for first layer, then split into per-layer chunks
        h_all = torch.nn.functional.linear(x, self.W1_cat, self.b1_cat)
        h_per_layer = h_all.split(self.hidden_features, dim=1)
        return [self.umas_radial_mlp(h_per_layer[i], i) for i in range(self.num_layers)]
