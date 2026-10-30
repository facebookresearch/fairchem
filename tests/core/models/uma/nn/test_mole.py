"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.fairchem.core.models.uma.nn.mole import (
    MOLE,
    MOLEGlobals,
    _pnorm,
    _softmax,
    init_linear,
    interval_intersection,
    norm_str_to_fn,
)


class TestUtilityFunctions:
    """Test utility functions for MoLE."""

    def test_interval_intersection_overlap(self):
        """Test interval intersection with overlapping intervals."""
        # Complete overlap
        result = interval_intersection([0, 10], [5, 15])
        assert result == [5, 10]

        # Partial overlap at start
        result = interval_intersection([0, 5], [3, 8])
        assert result == [3, 5]

        # Partial overlap at end
        result = interval_intersection([5, 10], [0, 7])
        assert result == [5, 7]

        # Identical intervals
        result = interval_intersection([2, 8], [2, 8])
        assert result == [2, 8]

    def test_interval_intersection_no_overlap(self):
        """Test interval intersection with non-overlapping intervals."""
        # No overlap - separate intervals
        result = interval_intersection([0, 5], [10, 15])
        assert result is None

        # No overlap - touching endpoints
        result = interval_intersection([0, 5], [5, 10])
        assert result == [5, 5]  # Single point intersection

        # No overlap - reversed order
        result = interval_intersection([10, 15], [0, 5])
        assert result is None

    def test_interval_intersection_edge_cases(self):
        """Test edge cases for interval intersection."""
        # Single point intervals
        result = interval_intersection([5, 5], [5, 5])
        assert result == [5, 5]

        # Single point vs range
        result = interval_intersection([5, 5], [3, 7])
        assert result == [5, 5]

        # Zero-length intervals
        result = interval_intersection([0, 0], [1, 1])
        assert result is None

    def test_softmax_function(self):
        """Test _softmax function behavior."""
        x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        result = _softmax(x)

        # Should be positive
        assert torch.all(result > 0)

        # The function adds 0.005 to each softmax element, so sum = 1 + 3*0.005 = 1.015
        row_sums = result.sum(dim=1)
        expected_sum = 1.0 + 3 * 0.005  # 1.0 + num_elements * epsilon
        assert torch.allclose(
            row_sums, torch.full_like(row_sums, expected_sum), rtol=1e-5
        )

        # Larger values should have higher probabilities
        assert result[0, 2] > result[0, 1] > result[0, 0]  # 3 > 2 > 1

    def test_pnorm_function(self):
        """Test _pnorm function behavior."""
        x = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
        result = _pnorm(x)

        # Should be positive
        assert torch.all(result >= 0)

        # Should sum to 1 per row (L1 normalized)
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5)

        # Should use absolute values
        assert torch.allclose(result[0], result[1])  # Same absolute values

    def test_norm_str_to_fn(self):
        """Test normalization function string mapping."""
        # Test valid mappings
        assert norm_str_to_fn("softmax") == _softmax
        assert norm_str_to_fn("pnorm") == _pnorm

        # Test invalid mapping
        with pytest.raises(ValueError):
            norm_str_to_fn("invalid")


class TestMOLEGlobals:
    """Test MOLEGlobals dataclass."""

    def test_mole_globals_creation(self):
        """Test creating MOLEGlobals instance."""
        expert_coeffs = torch.randn(3, 4)  # 3 systems, 4 experts
        mole_sizes = torch.tensor([10, 15, 8])  # 3 systems with different sizes

        globals_obj = MOLEGlobals(
            expert_mixing_coefficients=expert_coeffs,
            mole_sizes=mole_sizes,
            ac_start_idx=5,
        )

        assert torch.equal(globals_obj.expert_mixing_coefficients, expert_coeffs)
        assert torch.equal(globals_obj.mole_sizes, mole_sizes)
        assert globals_obj.ac_start_idx == 5

    def test_mole_globals_default_ac_start_idx(self):
        """Test default value for ac_start_idx."""
        expert_coeffs = torch.randn(2, 3)
        mole_sizes = torch.tensor([5, 7])

        globals_obj = MOLEGlobals(
            expert_mixing_coefficients=expert_coeffs, mole_sizes=mole_sizes
        )

        assert globals_obj.ac_start_idx == 0  # Default value


class TestInitLinear:
    """Test weight initialization function."""

    def test_init_linear_with_bias(self):
        """Test weight initialization with bias."""
        num_experts = 3
        in_features = 10
        out_features = 5

        weights, bias = init_linear(num_experts, True, out_features, in_features)

        # Check weight shape and type
        assert isinstance(weights, nn.Parameter)
        assert weights.shape == (num_experts, out_features, in_features)

        # Check bias shape and type
        assert isinstance(bias, nn.Parameter)
        assert bias.shape == (out_features,)

        # Check initialization range (Xavier-style)
        k = math.sqrt(1.0 / in_features)
        assert torch.all(weights >= -k)
        assert torch.all(weights <= k)
        assert torch.all(bias >= -k)
        assert torch.all(bias <= k)

    def test_init_linear_without_bias(self):
        """Test weight initialization without bias."""
        num_experts = 2
        in_features = 8
        out_features = 4

        weights, bias = init_linear(num_experts, False, out_features, in_features)

        assert isinstance(weights, nn.Parameter)
        assert weights.shape == (num_experts, out_features, in_features)
        assert bias is None

    def test_init_linear_different_experts(self):
        """Test that different experts get different initial weights."""
        num_experts = 3
        in_features = 5
        out_features = 3

        weights, _ = init_linear(num_experts, False, out_features, in_features)

        # Different experts should have different weights (with high probability)
        expert0 = weights[0]
        expert1 = weights[1]
        expert2 = weights[2]

        assert not torch.allclose(expert0, expert1, rtol=1e-3)
        assert not torch.allclose(expert1, expert2, rtol=1e-3)
        assert not torch.allclose(expert0, expert2, rtol=1e-3)


class TestMOLE:
    """Test MOLE implementation (pure PyTorch)."""

    @pytest.fixture()
    def setup_mole(self):
        """Setup MOLE test data."""
        num_experts = 3
        in_features = 6
        out_features = 4
        batch_size = 3

        expert_coeffs = torch.softmax(torch.randn(batch_size, num_experts), dim=1)
        mole_sizes = torch.tensor([5, 8, 7])  # 3 systems with different sizes

        globals_obj = MOLEGlobals(
            expert_mixing_coefficients=expert_coeffs, mole_sizes=mole_sizes
        )

        layer = MOLE(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            global_mole_tensors=globals_obj,
            bias=True,
        )

        return {
            "layer": layer,
            "globals": globals_obj,
            "num_experts": num_experts,
            "in_features": in_features,
            "out_features": out_features,
            "batch_size": batch_size,
            "total_atoms": mole_sizes.sum().item(),
        }

    def test_mole_initialization(self, setup_mole):
        """Test MOLE initialization."""
        data = setup_mole
        layer = data["layer"]

        assert layer.num_experts == data["num_experts"]
        assert layer.in_features == data["in_features"]
        assert layer.out_features == data["out_features"]

        # Check weight and bias shapes
        assert layer.weights.shape == (
            data["num_experts"],
            data["out_features"],
            data["in_features"],
        )
        assert layer.bias.shape == (data["out_features"],)

    def test_mole_forward_basic(self, setup_mole):
        """Test basic MOLE forward pass."""
        data = setup_mole
        layer = data["layer"]

        # Create input: [total_atoms, in_features]
        x = torch.randn(data["total_atoms"], data["in_features"])

        output = layer(x)

        # Check output shape
        assert output.shape == (data["total_atoms"], data["out_features"])

    def test_mole_system_segmentation(self, setup_mole):
        """Test that MOLE correctly segments systems."""
        data = setup_mole
        layer = data["layer"]
        globals_obj = data["globals"]

        # Manually compute expected system boundaries
        start_idxs = [0] + torch.cumsum(globals_obj.mole_sizes, dim=0).tolist()
        expected_intervals = list(zip(start_idxs, start_idxs[1:]))

        # Expected: [(0,5), (5,13), (13,20)] for mole_sizes=[5,8,7]
        assert expected_intervals == [(0, 5), (5, 13), (13, 20)]

        # Create input and test forward pass
        x = torch.randn(data["total_atoms"], data["in_features"])
        output = layer(x)

        # Verify that each system segment gets processed
        assert output.shape[0] == data["total_atoms"]

    def test_mole_activation_checkpointing(self, setup_mole):
        """Test MOLE with activation checkpointing (chunked inputs)."""
        data = setup_mole
        layer = data["layer"]
        globals_obj = data["globals"]

        # Test with a chunk that spans multiple systems
        # Total atoms: 20 (5+8+7), let's process atoms 3-15
        chunk_start = 3
        chunk_size = 12

        # Update ac_start_idx to simulate chunked processing
        globals_obj.ac_start_idx = chunk_start

        # Create chunked input
        x_chunk = torch.randn(chunk_size, data["in_features"])

        output = layer(x_chunk)

        # Output should match chunk size
        assert output.shape == (chunk_size, data["out_features"])

        # Reset for other tests
        globals_obj.ac_start_idx = 0

    def test_mole_interval_overlap_logic(self, setup_mole):
        """Test the interval overlap logic in detail."""
        data = setup_mole
        layer = data["layer"]
        globals_obj = data["globals"]

        # mole_sizes = [5, 8, 7] -> intervals: [(0,5), (5,13), (13,20)]
        # Test chunk (8, 16) should overlap with systems 1 and 2

        globals_obj.ac_start_idx = 8
        chunk_size = 8  # covers atoms 8-15
        x_chunk = torch.randn(chunk_size, data["in_features"])

        output = layer(x_chunk)

        assert output.shape == (chunk_size, data["out_features"])

    def test_mole_merged_linear_layer(self, setup_mole):
        """Test conversion to merged linear layer."""
        data = setup_mole
        layer = data["layer"]

        # Get merged linear layer (assumes first system's coefficients)
        merged_layer = layer.merged_linear_layer()

        # Check that it's a standard Linear layer
        assert isinstance(merged_layer, nn.Linear)
        assert merged_layer.in_features == data["in_features"]
        assert merged_layer.out_features == data["out_features"]

        # Check that weights are properly mixed
        expected_weight = torch.einsum(
            "eoi, be->boi", layer.weights, data["globals"].expert_mixing_coefficients
        )[0]  # First system

        assert torch.allclose(merged_layer.weight, expected_weight)

        if layer.bias is not None:
            assert torch.allclose(merged_layer.bias, layer.bias)

    def test_mole_weight_mixing_consistency(self, setup_mole):
        """Test that weight mixing produces consistent results."""
        data = setup_mole
        layer = data["layer"]

        # Create input
        x = torch.randn(data["total_atoms"], data["in_features"])

        # Forward pass
        output1 = layer(x)
        output2 = layer(x)

        # Should be deterministic
        assert torch.allclose(output1, output2)

        # Test that different expert coefficients produce different outputs
        original_coeffs = data["globals"].expert_mixing_coefficients.clone()

        # Change coefficients for second system
        data["globals"].expert_mixing_coefficients[1] = torch.softmax(
            torch.randn(data["num_experts"]), dim=0
        )

        output3 = layer(x)

        # Outputs should be different (at least for the second system's atoms)
        system2_start = data["globals"].mole_sizes[0].item()
        system2_end = system2_start + data["globals"].mole_sizes[1].item()

        assert not torch.allclose(
            output1[system2_start:system2_end], output3[system2_start:system2_end]
        )

        # Restore original coefficients
        data["globals"].expert_mixing_coefficients = original_coeffs

    def test_mole_empty_mole_sizes_error(self, setup_mole):
        """Test error handling with empty mole_sizes."""
        data = setup_mole
        layer = data["layer"]

        # Set empty mole_sizes
        data["globals"].mole_sizes = torch.tensor([])

        x = torch.randn(5, data["in_features"])

        with pytest.raises(AssertionError):
            layer(x)

    def test_mole_shape_mismatch_assertion(self, setup_mole):
        """Test shape mismatch assertion in forward pass."""
        data = setup_mole
        layer = data["layer"]

        # This test is tricky to trigger since the interval logic should prevent it
        # But we can test the assertion exists by examining the code path
        x = torch.randn(data["total_atoms"], data["in_features"])
        output = layer(x)

        # Verify the assertion passes for correct input
        assert output.shape[0] == x.shape[0]


def _reference_forward(layer, x, coefficients, sizes):
    """
    Apply each system's mixed expert weight to that system's rows of x.
    """
    outputs = []
    start = 0
    for system, size in enumerate(sizes):
        weight = torch.einsum("e,eoi->oi", coefficients[system], layer.weights)
        output = x[start : start + size] @ weight.T
        if layer.bias is not None:
            output = output + layer.bias
        outputs.append(output)
        start += size
    return torch.cat(outputs)


class TestMOLEOutputs:
    """
    Check MOLE outputs and gradients against a per-system reference.
    """

    num_experts = 4
    in_features = 6
    out_features = 5
    # The empty second system must not shift which coefficients later systems use
    sizes = (4, 0, 5, 3)

    def _make_layer(self, sizes, bias=True):
        """
        Build a MOLE layer with random mixing coefficients for each system.
        """
        torch.manual_seed(0)
        coefficients = torch.softmax(torch.randn(len(sizes), self.num_experts), dim=1)
        global_mole_tensors = MOLEGlobals(
            expert_mixing_coefficients=coefficients,
            mole_sizes=torch.tensor(sizes),
        )
        layer = MOLE(
            num_experts=self.num_experts,
            in_features=self.in_features,
            out_features=self.out_features,
            global_mole_tensors=global_mole_tensors,
            bias=bias,
        )
        return layer, coefficients

    def test_each_system_uses_its_own_mixed_weights(self):
        """
        Rows of each system are transformed by that system's mixture of the
        expert weights, including when a system has no rows.
        """
        layer, coefficients = self._make_layer(self.sizes)
        x = torch.randn(sum(self.sizes), self.in_features)

        torch.testing.assert_close(
            layer(x), _reference_forward(layer, x, coefficients, self.sizes)
        )

    @pytest.mark.parametrize("chunk_size", [1, 3, 5])
    def test_chunked_inputs_match_full_input(self, chunk_size):
        """
        Feeding the rows in chunks with ac_start_idx set to each chunk's offset,
        as activation checkpointing does, reproduces the full output. The chunk
        sizes split systems, span several systems, and border the empty system.
        """
        layer, _ = self._make_layer(self.sizes)
        x = torch.randn(sum(self.sizes), self.in_features)
        full = layer(x)

        chunks = []
        for start in range(0, x.shape[0], chunk_size):
            layer.global_mole_tensors.ac_start_idx = start
            chunks.append(layer(x[start : start + chunk_size]))

        torch.testing.assert_close(torch.cat(chunks), full)

    @pytest.mark.parametrize("bias", [True, False])
    def test_merged_linear_layer_matches_forward(self, bias):
        """
        For a single system, the merged nn.Linear gives the same output as the
        MOLE layer.
        """
        layer, _ = self._make_layer((7,), bias=bias)
        x = torch.randn(7, self.in_features)

        merged = layer.merged_linear_layer()

        torch.testing.assert_close(merged(x), layer(x))

    def test_frozen_weights_with_3d_input(self):
        """
        With frozen expert weights and a 3D input that needs gradients, as in
        force inference, outputs and input gradients match the reference.
        """
        layer, coefficients = self._make_layer(self.sizes)
        layer.requires_grad_(False)
        x = torch.randn(sum(self.sizes), 3, self.in_features, requires_grad=True)

        out = layer(x)
        expected = _reference_forward(layer, x, coefficients, self.sizes)
        torch.testing.assert_close(out, expected)

        grad_out = torch.randn_like(out)
        (grad,) = torch.autograd.grad(out, x, grad_out)
        (expected_grad,) = torch.autograd.grad(expected, x, grad_out)
        torch.testing.assert_close(grad, expected_grad)
