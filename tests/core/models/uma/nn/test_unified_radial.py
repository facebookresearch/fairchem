"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for UnifiedRadialMLP.
"""

from __future__ import annotations

import pytest
import torch


class TestUnifiedRadialMLP:
    """Tests for UnifiedRadialMLP functionality."""

    @pytest.fixture()
    def radial_mlp_list(self):
        """Create a list of RadialMLP modules with different weights."""
        from fairchem.core.models.uma.nn.radial import RadialMLP

        torch.manual_seed(42)
        num_layers = 8
        edge_channels = [64, 128, 128, 256]

        rad_funcs = []
        for _ in range(num_layers):
            mlp = RadialMLP(edge_channels)
            # Randomize weights to ensure each layer is different
            for param in mlp.parameters():
                param.data = torch.randn_like(param.data)
            rad_funcs.append(mlp)

        return rad_funcs

    def test_unified_matches_per_layer(self, radial_mlp_list):
        """Test that unified radial outputs match per-layer RadialMLP outputs."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        # Create test input
        torch.manual_seed(123)
        E = 500  # number of edges
        x_edge = torch.randn(E, 64)  # edge features

        # Compute per-layer outputs
        per_layer_outputs = [mlp(x_edge) for mlp in radial_mlp_list]

        # Compute unified output
        unified_outputs = unified(x_edge)

        # Compare
        assert len(unified_outputs) == len(per_layer_outputs)
        for i, (unified_out, per_layer_out) in enumerate(
            zip(unified_outputs, per_layer_outputs)
        ):
            torch.testing.assert_close(
                unified_out,
                per_layer_out,
                atol=1e-6,
                rtol=1e-6,
                msg=f"Layer {i} output mismatch",
            )

    def test_unified_output_shapes(self, radial_mlp_list):
        """Test that unified radial outputs have correct shapes."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        E = 300
        x_edge = torch.randn(E, 64)

        outputs = unified(x_edge)

        # Check list length
        assert len(outputs) == len(radial_mlp_list)

        # Check each tensor shape - should match RadialMLP output
        expected_out_features = radial_mlp_list[0].net[-1].out_features
        for i, out in enumerate(outputs):
            assert out.shape == (E, expected_out_features), (
                f"Layer {i}: expected shape ({E}, {expected_out_features}), "
                f"got {out.shape}"
            )

    def test_unified_is_inference_only(self, radial_mlp_list):
        """Test that unified radial has no learnable parameters (inference-only)."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        # All weights should be buffers, not parameters
        params = list(unified.parameters())
        assert len(params) == 0, (
            f"Expected 0 learnable parameters, got {len(params)}. "
            "UnifiedRadialMLP should be inference-only."
        )

        # But should have buffers
        buffers = list(unified.buffers())
        assert len(buffers) > 0, "Expected buffers but got none"

    def test_unified_different_batch_sizes(self, radial_mlp_list):
        """Test that unified radial works with different batch sizes."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        for E in [1, 10, 100, 1000]:
            x_edge = torch.randn(E, 64)
            outputs = unified(x_edge)

            assert len(outputs) == len(radial_mlp_list)
            for out in outputs:
                assert out.shape[0] == E

    def test_unified_preserves_dtype(self, radial_mlp_list):
        """Test that unified radial preserves input dtype."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        E = 100
        for dtype in [torch.float32, torch.float64]:
            # Convert unified to dtype
            unified_typed = unified.to(dtype)
            x_edge = torch.randn(E, 64, dtype=dtype)

            outputs = unified_typed(x_edge)

            for out in outputs:
                assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"

    def test_unified_gradient_flow(self, radial_mlp_list):
        """Test that gradients flow through unified radial."""
        from fairchem.core.models.uma.nn.unified_radial import (
            create_unified_radial_mlp,
        )

        unified = create_unified_radial_mlp(radial_mlp_list)

        E = 100
        x_edge = torch.randn(E, 64, requires_grad=True)

        outputs = unified(x_edge)

        # Sum all outputs and backprop
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        # Input should have gradients
        assert x_edge.grad is not None
        assert x_edge.grad.abs().sum() > 0
