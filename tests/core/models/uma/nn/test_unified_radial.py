"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for UnifiedRadialMLP.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.radial import RadialMLP
from fairchem.core.models.uma.nn.unified_radial import (
    UnifiedRadialMLP,
    create_unified_radial_mlp,
)


class TestUnifiedRadialMLP:
    """Tests for UnifiedRadialMLP with edge_degree + layer radials."""

    @pytest.fixture
    def edge_degree_mlp(self):
        """Create edge_degree RadialMLP."""
        return RadialMLP([768, 128, 128, 32])

    @pytest.fixture
    def layer_radial_mlps(self):
        """Create layer RadialMLPs."""
        return [RadialMLP([768, 128, 128, 32]) for _ in range(4)]

    def test_output_shape(self, edge_degree_mlp, layer_radial_mlps):
        """Verify output list has correct length and tensor shapes."""
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)
        x = torch.randn(100, 768)

        outputs = unified(x)

        # Output: [edge_degree_out, layer_0_out, ..., layer_N-1_out]
        assert len(outputs) == 1 + len(layer_radial_mlps)
        for out in outputs:
            assert out.shape == (100, 32)

    def test_buffer_shapes(self, edge_degree_mlp, layer_radial_mlps):
        """Verify internal buffers have expected shapes."""
        num_layers = len(layer_radial_mlps)
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)

        # First layer: concatenated [(edge_degree+N*layer)*H, in]
        total_hidden = 128 + num_layers * 128
        assert unified.W1_cat.shape == (total_hidden, 768)
        assert unified.b1_cat.shape == (total_hidden,)

        # Layer norm weights for layers (stacked)
        assert unified.ln1_weight.shape == (num_layers, 128)
        assert unified.fc2_weight.shape == (num_layers, 128, 128)
        assert unified.fc3_weight.shape == (num_layers, 32, 128)

    def test_attributes(self, edge_degree_mlp, layer_radial_mlps):
        """Verify module attributes are set correctly."""
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)

        assert unified.num_layers == len(layer_radial_mlps)
        assert unified.hidden_features == 128
        assert unified.edge_hidden_features == 128

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 8, 12])
    def test_different_num_layers(self, num_layers):
        """Test with various numbers of layers."""
        edge_degree_mlp = RadialMLP([768, 128, 128, 32])
        layer_radial_mlps = [
            RadialMLP([768, 128, 128, 32]) for _ in range(num_layers)
        ]
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)
        x = torch.randn(50, 768)

        outputs = unified(x)

        assert len(outputs) == 1 + num_layers
        assert unified.num_layers == num_layers


class TestUnifiedRadialMLPBackwardCompatibility:
    """Tests ensuring unified output matches original RadialMLP outputs."""

    @pytest.fixture
    def edge_degree_mlp(self):
        """Create edge_degree RadialMLP with fixed seed."""
        torch.manual_seed(42)
        return RadialMLP([768, 128, 128, 32])

    @pytest.fixture
    def layer_radial_mlps(self):
        """Create layer RadialMLPs with fixed seed."""
        torch.manual_seed(43)
        return [RadialMLP([768, 128, 128, 32]) for _ in range(4)]

    def test_matches_original_radial_mlps(self, edge_degree_mlp, layer_radial_mlps):
        """Verify unified output matches individual RadialMLP forward passes."""
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)

        torch.manual_seed(123)
        x = torch.randn(100, 768)

        # Get unified outputs
        unified_outputs = unified(x)

        # Compare edge_degree output (index 0)
        expected_edge = edge_degree_mlp(x)
        torch.testing.assert_close(
            unified_outputs[0],
            expected_edge,
            rtol=1e-5,
            atol=1e-5,
            msg="Mismatch at edge_degree",
        )

        # Compare layer outputs (indices 1..N)
        for i, original_mlp in enumerate(layer_radial_mlps):
            expected = original_mlp(x)
            torch.testing.assert_close(
                unified_outputs[i + 1],
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch at layer {i}",
            )

    def test_gradient_flow(self, edge_degree_mlp, layer_radial_mlps):
        """Verify gradients flow correctly through the unified implementation."""
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)

        x = torch.randn(50, 768, requires_grad=True)
        outputs = unified(x)
        loss = sum(out.sum() for out in outputs)

        # This should not raise
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 500])
    def test_different_batch_sizes(
        self, edge_degree_mlp, layer_radial_mlps, batch_size
    ):
        """Test numerical equivalence across different batch sizes."""
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps)
        x = torch.randn(batch_size, 768)

        unified_outputs = unified(x)

        # Check edge_degree
        expected_edge = edge_degree_mlp(x)
        torch.testing.assert_close(
            unified_outputs[0], expected_edge, rtol=1e-5, atol=1e-5
        )

        # Check layer outputs
        for i, original_mlp in enumerate(layer_radial_mlps):
            expected = original_mlp(x)
            torch.testing.assert_close(
                unified_outputs[i + 1], expected, rtol=1e-5, atol=1e-5
            )


class TestCreateUnifiedRadialMLP:
    """Tests for the factory function."""

    @pytest.fixture
    def edge_degree_mlp(self):
        """Create edge_degree RadialMLP."""
        return RadialMLP([768, 128, 128, 32])

    @pytest.fixture
    def layer_radial_mlps(self):
        """Create layer RadialMLPs."""
        return [RadialMLP([768, 128, 128, 32]) for _ in range(3)]

    def test_factory_creates_instance(self, edge_degree_mlp, layer_radial_mlps):
        """Verify factory returns UnifiedRadialMLP instance."""
        result = create_unified_radial_mlp(edge_degree_mlp, layer_radial_mlps)

        assert isinstance(result, UnifiedRadialMLP)
        assert result.num_layers == 3

    def test_factory_empty_layers_raises(self, edge_degree_mlp):
        """Verify factory raises on empty layer list."""
        with pytest.raises(AssertionError):
            create_unified_radial_mlp(edge_degree_mlp, [])

    def test_unified_is_inference_only(self, edge_degree_mlp, layer_radial_mlps):
        """Test that unified radial has no learnable parameters (inference-only)."""
        unified = create_unified_radial_mlp(edge_degree_mlp, layer_radial_mlps)

        # All weights should be registered as buffers, not parameters
        params = list(unified.parameters())
        assert len(params) == 0, f"Expected no parameters, got {len(params)}"

        # But should have buffers
        buffers = list(unified.buffers())
        assert len(buffers) > 0, "Expected buffers for weights"


@pytest.mark.gpu
class TestUnifiedRadialMLPGPU:
    """GPU-specific tests."""

    def test_cuda_execution(self):
        """Verify module works on CUDA."""
        edge_degree_mlp = RadialMLP([768, 128, 128, 32]).cuda()
        layer_radial_mlps = [RadialMLP([768, 128, 128, 32]).cuda() for _ in range(4)]
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps).cuda()
        x = torch.randn(100, 768, device="cuda")

        outputs = unified(x)

        assert all(out.device.type == "cuda" for out in outputs)

    def test_matches_original_on_gpu(self):
        """Verify GPU results match original RadialMLPs."""
        torch.manual_seed(42)
        edge_degree_mlp = RadialMLP([768, 128, 128, 32]).cuda()
        torch.manual_seed(43)
        layer_radial_mlps = [RadialMLP([768, 128, 128, 32]).cuda() for _ in range(4)]
        unified = UnifiedRadialMLP(edge_degree_mlp, layer_radial_mlps).cuda()

        x = torch.randn(100, 768, device="cuda")
        unified_outputs = unified(x)

        # Check edge_degree
        expected_edge = edge_degree_mlp(x)
        torch.testing.assert_close(
            unified_outputs[0], expected_edge, rtol=1e-4, atol=1e-4
        )

        # Check layers
        for i, original_mlp in enumerate(layer_radial_mlps):
            expected = original_mlp(x)
            torch.testing.assert_close(
                unified_outputs[i + 1], expected, rtol=1e-4, atol=1e-4
            )
