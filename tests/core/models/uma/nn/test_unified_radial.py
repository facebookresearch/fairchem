"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for UnifiedRadialMLP.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestUnifiedRadialMLP:
    """Tests for UnifiedRadialMLP functionality."""

    @pytest.fixture
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
                assert out.dtype == dtype, (
                    f"Expected dtype {dtype}, got {out.dtype}"
                )

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


class TestPrecomputedRadialIntegration:
    """Tests for precomputed_radial parameter integration."""

    def test_precomputed_radial_none_is_noop(self):
        """Test that precomputed_radial=None produces identical output to omitting it."""
        from fairchem.core.models.uma.nn.radial import RadialMLP
        from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
        from fairchem.core.models.uma.common.so3 import CoefficientMapping

        torch.manual_seed(42)

        # Create SO2_Convolution with rad_func
        lmax, mmax = 2, 2
        sphere_channels = 64
        m_output_channels = 64

        mappingReduced = CoefficientMapping(lmax, mmax)

        conv = SO2_Convolution(
            sphere_channels=sphere_channels,
            m_output_channels=m_output_channels,
            lmax=lmax,
            mmax=mmax,
            mappingReduced=mappingReduced,
            internal_weights=False,
            edge_channels_list=[128, 256, 256],
            extra_m0_output_channels=sphere_channels,
        )

        # Create inputs
        E = 100
        num_coeffs = mappingReduced.res_size
        x = torch.randn(E, num_coeffs, sphere_channels)
        x_edge = torch.randn(E, 128)

        # Forward without precomputed_radial
        out1, gate1 = conv(x, x_edge)

        # Forward with precomputed_radial=None (should be identical)
        out2, gate2 = conv(x, x_edge, precomputed_radial=None)

        torch.testing.assert_close(out1, out2, atol=0, rtol=0)
        torch.testing.assert_close(gate1, gate2, atol=0, rtol=0)

    def test_precomputed_radial_matches_computed(self):
        """Test that precomputed radial produces same output as computed radial."""
        from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
        from fairchem.core.models.uma.common.so3 import CoefficientMapping

        torch.manual_seed(42)

        lmax, mmax = 2, 2
        sphere_channels = 64
        m_output_channels = 64

        mappingReduced = CoefficientMapping(lmax, mmax)

        conv = SO2_Convolution(
            sphere_channels=sphere_channels,
            m_output_channels=m_output_channels,
            lmax=lmax,
            mmax=mmax,
            mappingReduced=mappingReduced,
            internal_weights=False,
            edge_channels_list=[128, 256, 256],
            extra_m0_output_channels=sphere_channels,
        )

        E = 100
        num_coeffs = mappingReduced.res_size
        x = torch.randn(E, num_coeffs, sphere_channels)
        x_edge = torch.randn(E, 128)

        # Compute radial manually
        precomputed = conv.rad_func(x_edge)

        # Forward with computed radial
        out1, gate1 = conv(x, x_edge)

        # Forward with precomputed radial
        out2, gate2 = conv(x, x_edge, precomputed_radial=precomputed)

        torch.testing.assert_close(
            out1, out2, atol=1e-6, rtol=1e-6,
            msg="Output mismatch with precomputed radial"
        )
        torch.testing.assert_close(
            gate1, gate2, atol=1e-6, rtol=1e-6,
            msg="Gate mismatch with precomputed radial"
        )
