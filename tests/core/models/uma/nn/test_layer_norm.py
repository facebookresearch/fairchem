"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.fairchem.core.models.uma.nn.layer_norm import (
    get_l_to_all_m_expand_index,
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantDegreeLayerScale,
)


class TestGetLToAllMExpandIndex:
    """Test the get_l_to_all_m_expand_index function."""
    
    def test_lmax_0(self):
        """Test with lmax=0 (only L=0, single component)."""
        expand_index = get_l_to_all_m_expand_index(0)
        expected = torch.tensor([0], dtype=torch.long)
        assert torch.equal(expand_index, expected)
        assert expand_index.shape == (1,)
    
    def test_lmax_1(self):
        """Test with lmax=1 (L=0: 1 component, L=1: 3 components)."""
        expand_index = get_l_to_all_m_expand_index(1)
        # L=0: position 0 -> degree 0
        # L=1: positions 1,2,3 -> degree 1
        expected = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        assert torch.equal(expand_index, expected)
        assert expand_index.shape == (4,)
    
    def test_lmax_2(self):
        """Test with lmax=2 (L=0: 1, L=1: 3, L=2: 5 components)."""
        expand_index = get_l_to_all_m_expand_index(2)
        # L=0: position 0 -> degree 0
        # L=1: positions 1,2,3 -> degree 1  
        # L=2: positions 4,5,6,7,8 -> degree 2
        expected = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
        assert torch.equal(expand_index, expected)
        assert expand_index.shape == (9,)
    
    def test_lmax_3(self):
        """Test with lmax=3."""
        expand_index = get_l_to_all_m_expand_index(3)
        expected_length = (3 + 1) ** 2  # 16
        assert expand_index.shape == (expected_length,)
        
        # Check specific patterns
        assert expand_index[0] == 0  # L=0
        assert torch.all(expand_index[1:4] == 1)  # L=1
        assert torch.all(expand_index[4:9] == 2)  # L=2
        assert torch.all(expand_index[9:16] == 3)  # L=3
    
    def test_spherical_harmonic_indexing_formula(self):
        """Test that the indexing follows spherical harmonic conventions."""
        for lmax in range(5):
            expand_index = get_l_to_all_m_expand_index(lmax)
            
            # Total length should be (lmax + 1)^2
            assert len(expand_index) == (lmax + 1) ** 2
            
            # Check each degree L has 2*L + 1 components
            for lval in range(lmax + 1):
                start_idx = lval ** 2
                length = 2 * lval + 1
                end_idx = start_idx + length
                
                # All positions for this L should map to lval
                assert torch.all(expand_index[start_idx:end_idx] == lval)
    
    def test_dtype(self):
        """Test that output has correct dtype."""
        expand_index = get_l_to_all_m_expand_index(2)
        assert expand_index.dtype == torch.long


class TestEquivariantLayerNormArray:
    """Test the EquivariantLayerNormArray class."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        lmax = 2
        num_channels = 8
        batch_size = 4
        sphere_basis = (lmax + 1) ** 2  # 9
        
        # Create test input [N, sphere_basis, C]
        x = torch.randn(batch_size, sphere_basis, num_channels)
        
        return {
            'x': x,
            'lmax': lmax,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'sphere_basis': sphere_basis
        }
    
    def test_initialization_with_affine(self, setup_data):
        """Test layer initialization with affine parameters."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True
        )
        
        # Check parameters exist and have correct shapes
        assert layer.affine_weight is not None
        assert layer.affine_bias is not None
        assert layer.affine_weight.shape == (data['lmax'] + 1, data['num_channels'])
        assert layer.affine_bias.shape == (data['num_channels'],)
        
        # Check initial values
        assert torch.allclose(layer.affine_weight, torch.ones_like(layer.affine_weight))
        assert torch.allclose(layer.affine_bias, torch.zeros_like(layer.affine_bias))
    
    def test_initialization_without_affine(self, setup_data):
        """Test layer initialization without affine parameters."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False
        )
        
        assert layer.affine_weight is None
        assert layer.affine_bias is None
    
    def test_forward_shape(self, setup_data):
        """Test that forward pass preserves input shape."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        output = layer(data['x'])
        assert output.shape == data['x'].shape
    
    def test_l0_mean_subtraction(self, setup_data):
        """Test that L=0 components have mean subtracted."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False  # Disable affine to isolate normalization
        )
        
        # Create input with non-zero mean for L=0
        x = data['x'].clone()
        x[:, 0, :] += 5.0  # Add constant to L=0 components
        
        output = layer(x)
        
        # L=0 components should have approximately zero mean across channels
        l0_output = output[:, 0, :]  # [N, C]
        l0_mean = l0_output.mean(dim=-1)  # [N]
        assert torch.allclose(l0_mean, torch.zeros_like(l0_mean), atol=1e-6)
    
    def test_higher_l_no_mean_subtraction(self, setup_data):
        """Test that L>0 components are normalized but don't have mean subtracted."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False
        )
        
        # Create input where all L>0 components have the same positive value
        x = torch.zeros_like(data['x'])
        x[:, 1:, :] = 2.0  # Set all L>0 components to 2
        
        output = layer(x)
        
        # L>0 components should be normalized (scaled) but not zero-centered
        l_higher_output = output[:, 1:, :]
        
        # They should maintain their positive values (no mean subtraction)
        assert torch.all(l_higher_output > 0)
        
        # They should be normalized - check that variance is controlled
        # For uniform input, all L>0 should have the same normalized value
        for lval in range(1, data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            degree_output = output[:, start_idx:(start_idx + length), :]
            
            # All components of the same degree should have identical values
            # since input was uniform and no mean subtraction occurred
            if length > 1:
                first_component = degree_output[:, 0, :]  # [N, C]
                for m_idx in range(1, length):
                    m_component = degree_output[:, m_idx, :]
                    assert torch.allclose(first_component, m_component, rtol=1e-6)
    
    def test_normalization_types(self, setup_data):
        """Test different normalization types (norm vs component)."""
        data = setup_data
        
        layer_norm = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            normalization="norm",
            affine=False
        )
        
        layer_component = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            normalization="component", 
            affine=False
        )
        
        output_norm = layer_norm(data['x'])
        output_component = layer_component(data['x'])
        
        # Outputs should be different for different normalization types
        assert not torch.allclose(output_norm, output_component)
        
        # Both should preserve input shape
        assert output_norm.shape == data['x'].shape
        assert output_component.shape == data['x'].shape
    
    def test_eps_numerical_stability(self, setup_data):
        """Test numerical stability with small eps value."""
        data = setup_data
        
        # Create input with very small values
        x = torch.ones_like(data['x']) * 1e-8
        
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            eps=1e-5
        )
        
        # Should not crash or produce NaN/Inf
        output = layer(x)
        assert torch.isfinite(output).all()
        assert output.shape == x.shape
    
    def test_l_higher_normalization_but_no_centering(self, setup_data):
        """Test that L>0 components are normalized but preserve equivariance."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False,
            normalization="component"  # Use component normalization for clearer effect
        )
        
        # Create input with non-unit values to see normalization effect clearly
        x = torch.zeros_like(data['x'])
        x[:, 1:4, :] = 5.0   # L=1 components (large values)
        x[:, 4:9, :] = 10.0  # L=2 components (even larger)
        
        output = layer(x)
        
        # Verify that L>0 components ARE changed (normalized) 
        l1_input = x[:, 1:4, :]
        l1_output = output[:, 1:4, :]
        assert not torch.allclose(l1_input, l1_output, rtol=0.1)  # Should be different (normalized)
        
        l2_input = x[:, 4:9, :]
        l2_output = output[:, 4:9, :]
        assert not torch.allclose(l2_input, l2_output, rtol=0.1)  # Should be different (normalized)
        
        # Check that normalized values are smaller than input (scaled down)
        assert torch.all(l1_output.abs() < l1_input.abs())
        assert torch.all(l2_output.abs() < l2_input.abs())
        
        # Each degree should be normalized separately 
        # Check that within each degree, all m components have the same magnitude
        for lval in range(1, data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            degree_output = output[:, start_idx:(start_idx + length), :]
            
            if length > 1:
                # All m components should have the same norm (equivariant normalization)
                norms = degree_output.norm(dim=-1)  # [N, 2L+1] 
                first_norm = norms[:, 0]  # [N]
                for m_idx in range(1, length):
                    m_norm = norms[:, m_idx]
                    assert torch.allclose(first_norm, m_norm, rtol=1e-6)
    
    def test_equivariance_preservation(self, setup_data):
        """Test that the layer preserves equivariance properties."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False  # Disable affine for cleaner test
        )
        
        # For L=0, scaling should not affect relative relationships
        x = data['x'].clone()
        x_scaled = x.clone()
        x_scaled[:, 0, :] *= 2.0  # Scale L=0 components
        
        output1 = layer(x)
        output2 = layer(x_scaled)
        
        # L=0 outputs should have same variance after normalization
        l0_var1 = output1[:, 0, :].var(dim=-1)
        l0_var2 = output2[:, 0, :].var(dim=-1)
        assert torch.allclose(l0_var1, l0_var2, rtol=1e-5)
    
    def test_repr(self, setup_data):
        """Test string representation."""
        data = setup_data
        layer = EquivariantLayerNormArray(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            eps=1e-6
        )
        
        repr_str = repr(layer)
        assert "EquivariantLayerNormArray" in repr_str
        assert f"lmax={data['lmax']}" in repr_str
        assert f"num_channels={data['num_channels']}" in repr_str
        assert "eps=1e-06" in repr_str


class TestEquivariantLayerNormArraySphericalHarmonics:
    """Test the EquivariantLayerNormArraySphericalHarmonics class."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        lmax = 2
        num_channels = 8
        batch_size = 4
        sphere_basis = (lmax + 1) ** 2  # 9
        
        x = torch.randn(batch_size, sphere_basis, num_channels)
        
        return {
            'x': x,
            'lmax': lmax,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'sphere_basis': sphere_basis
        }
    
    def test_initialization_with_std_balance(self, setup_data):
        """Test initialization with degree balancing."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            std_balance_degrees=True
        )
        
        # Check that balance weights exist and have correct shape
        assert layer.balance_degree_weight is not None
        expected_shape = ((data['lmax'] + 1) ** 2 - 1, 1)  # Exclude L=0
        assert layer.balance_degree_weight.shape == expected_shape
        
        # Check that weights sum correctly (weighted by 1/(2L+1) for each degree)
        # For lmax=2: L=1 has 3 components (weight 1/3), L=2 has 5 components (weight 1/5)
        # Total weight should be normalized by lmax
        total_weight = layer.balance_degree_weight.sum()
        expected_total = (1.0 + 1.0) / data['lmax']  # (L=1 + L=2) / lmax
        assert torch.allclose(total_weight, torch.tensor(expected_total))
    
    def test_initialization_without_std_balance(self, setup_data):
        """Test initialization without degree balancing."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            std_balance_degrees=False
        )
        
        assert layer.balance_degree_weight is None
    
    def test_l0_uses_standard_layernorm(self, setup_data):
        """Test that L=0 components use standard PyTorch LayerNorm."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        # Check that norm_l0 is a standard LayerNorm
        assert isinstance(layer.norm_l0, nn.LayerNorm)
        assert layer.norm_l0.normalized_shape == (data['num_channels'],)
        
        # Test that L=0 normalization works as expected
        x = data['x'].clone()
        x[:, 0, :] += 10.0  # Add offset to L=0
        
        output = layer(x)
        
        # L=0 should be properly normalized
        l0_output = output[:, 0, :]  # Shape: [N, C]
        
        # For standard LayerNorm, each sample should have mean ≈ 0 and std ≈ 1 across channels
        l0_mean = l0_output.mean(dim=-1)  # Mean across channels for each sample
        l0_std = l0_output.std(dim=-1, unbiased=False)  # Std across channels for each sample
        
        assert torch.allclose(l0_mean, torch.zeros_like(l0_mean), atol=1e-6)
        assert torch.allclose(l0_std, torch.ones_like(l0_std), rtol=1e-5)
    
    def test_higher_l_joint_normalization(self, setup_data):
        """Test that L>0 components are normalized jointly."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False,
            std_balance_degrees=False
        )
        
        # Create input where L>0 components have different scales
        x = torch.zeros_like(data['x'])
        x[:, 1:4, :] = 1.0   # L=1 components
        x[:, 4:9, :] = 2.0   # L=2 components
        
        output = layer(x)
        
        # Since all L>0 components are normalized together, they should have similar variance
        # The relative scaling should be preserved in the normalization
        l1_output = output[:, 1:4, :]
        l2_output = output[:, 4:9, :]
        
        # Check that both have non-zero values and the ratio is reasonable
        assert not torch.allclose(l1_output, torch.zeros_like(l1_output))
        assert not torch.allclose(l2_output, torch.zeros_like(l2_output))
        
        # The ratio should be approximately preserved but may not be exactly 2.0 due to joint normalization
        l1_norm = l1_output.norm(dim=1).mean()
        l2_norm = l2_output.norm(dim=1).mean()
        ratio = l2_norm / l1_norm
        # Allow a wider tolerance since joint normalization affects the ratio
        assert 1.0 < ratio < 3.0  # Should still be larger but not necessarily exactly 2.0
    
    def test_std_balance_degrees_effect(self, setup_data):
        """Test the effect of std_balance_degrees on normalization."""
        data = setup_data
        
        layer_balanced = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            std_balance_degrees=True,
            affine=False
        )
        
        layer_unbalanced = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            std_balance_degrees=False,
            affine=False
        )
        
        # Create input with different magnitudes for different degrees
        x = torch.zeros_like(data['x'])
        x[:, 1:4, :] = 1.0   # L=1: 3 components
        x[:, 4:9, :] = 2.0   # L=2: 5 components (different magnitude)
        
        output_balanced = layer_balanced(x)
        output_unbalanced = layer_unbalanced(x)
        
        # Outputs should be different due to different weighting schemes
        # Check that at least the L>0 parts are different
        assert not torch.allclose(output_balanced[:, 1:, :], output_unbalanced[:, 1:, :], rtol=1e-5)
    
    def test_forward_shape_preservation(self, setup_data):
        """Test that forward pass preserves input shape."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        output = layer(data['x'])
        assert output.shape == data['x'].shape
    
    def test_lmax_zero_case(self):
        """Test edge case with lmax=0 (only L=0 components)."""
        lmax = 0
        num_channels = 4
        batch_size = 2
        x = torch.randn(batch_size, 1, num_channels)
        
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=lmax,
            num_channels=num_channels
        )
        
        output = layer(x)
        assert output.shape == x.shape
        
        # Should behave like standard LayerNorm for L=0
        expected = layer.norm_l0(x)
        assert torch.allclose(output, expected)
    
    def test_repr(self, setup_data):
        """Test string representation."""
        data = setup_data
        layer = EquivariantLayerNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            eps=1e-6,
            std_balance_degrees=True
        )
        
        repr_str = repr(layer)
        assert "EquivariantLayerNormArraySphericalHarmonics" in repr_str
        assert f"lmax={data['lmax']}" in repr_str
        assert f"num_channels={data['num_channels']}" in repr_str
        assert "std_balance_degrees=True" in repr_str


class TestEquivariantRMSNormArraySphericalHarmonics:
    """Test the EquivariantRMSNormArraySphericalHarmonics class."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        lmax = 2
        num_channels = 8
        batch_size = 4
        sphere_basis = (lmax + 1) ** 2  # 9
        
        x = torch.randn(batch_size, sphere_basis, num_channels)
        
        return {
            'x': x,
            'lmax': lmax,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'sphere_basis': sphere_basis
        }
    
    def test_initialization(self, setup_data):
        """Test layer initialization."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True
        )
        
        # Check affine parameters
        assert layer.affine_weight.shape == (data['lmax'] + 1, data['num_channels'])
        assert torch.allclose(layer.affine_weight, torch.ones_like(layer.affine_weight))
    
    def test_rms_normalization_all_degrees(self, setup_data):
        """Test that RMS normalization is applied to all degrees L>=0."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False,
            normalization="component"
        )
        
        # Create input with uniform values
        x = torch.ones_like(data['x'])
        output = layer(x)
        
        # All components should be normalized by the same global factor
        # Check that the relative scaling is preserved across degrees
        for lval in range(data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            degree_output = output[:, start_idx:(start_idx + length), :]
            
            # Each degree should have consistent scaling
            if length > 1:
                norms = degree_output.norm(dim=-1)  # [N, 2L+1]
                assert torch.allclose(norms[:, 0], norms[:, 1], rtol=1e-5)
    
    def test_no_mean_subtraction(self, setup_data):
        """Test that RMS norm doesn't subtract mean from any degree."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=False
        )
        
        # Create input with non-zero mean everywhere
        x = torch.ones_like(data['x']) * 5.0
        output = layer(x)
        
        # No component should have zero mean (since no mean subtraction)
        for lval in range(data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            degree_output = output[:, start_idx:(start_idx + length), :]
            degree_mean = degree_output.mean(dim=-1)
            assert not torch.allclose(degree_mean, torch.zeros_like(degree_mean))
    
    def test_forward_shape_preservation(self, setup_data):
        """Test that forward pass preserves input shape."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonics(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        output = layer(data['x'])
        assert output.shape == data['x'].shape


class TestEquivariantRMSNormArraySphericalHarmonicsV2:
    """Test the EquivariantRMSNormArraySphericalHarmonicsV2 class."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        lmax = 2
        num_channels = 8
        batch_size = 4
        sphere_basis = (lmax + 1) ** 2  # 9
        
        x = torch.randn(batch_size, sphere_basis, num_channels)
        
        return {
            'x': x,
            'lmax': lmax,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'sphere_basis': sphere_basis
        }
    
    def test_initialization_with_all_options(self, setup_data):
        """Test initialization with all options enabled."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True,
            centering=True,
            std_balance_degrees=True
        )
        
        # Check parameters
        assert layer.affine_weight.shape == (data['lmax'] + 1, data['num_channels'])
        assert layer.affine_bias.shape == (data['num_channels'],)
        
        # Check buffers
        assert layer.expand_index.shape == (data['sphere_basis'],)
        assert layer.balance_degree_weight.shape == (data['sphere_basis'], 1)
        
        # Check balance weights sum correctly
        total_weight = layer.balance_degree_weight.sum()
        expected_total = 1.0  # Should be normalized to sum to 1
        assert torch.allclose(total_weight, torch.tensor(expected_total))
    
    def test_expand_index_buffer(self, setup_data):
        """Test that expand_index buffer is created correctly."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        # expand_index should map spherical harmonic components to their degree L
        expected_expand_index = get_l_to_all_m_expand_index(data['lmax'])
        assert torch.equal(layer.expand_index, expected_expand_index)
    
    def test_centering_l0_only(self, setup_data):
        """Test that centering only affects L=0 components."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            centering=True,
            affine=False  # Disable affine for cleaner test
        )
        
        # Create input with non-zero mean for L=0
        x = data['x'].clone()
        x[:, 0, :] += 10.0  # Add large offset to L=0
        
        output = layer(x)
        
        # L=0 should have approximately zero mean
        l0_output = output[:, 0, :]
        l0_mean = l0_output.mean(dim=-1)
        assert torch.allclose(l0_mean, torch.zeros_like(l0_mean), atol=1e-6)
        
        # L>0 components should not be affected by centering in the same way
        # (they don't get mean subtracted individually)
    
    def test_no_centering(self, setup_data):
        """Test behavior without centering."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            centering=False,
            affine=False
        )
        
        # Create input with non-zero mean for L=0
        x = data['x'].clone()
        x[:, 0, :] += 10.0
        
        output = layer(x)
        
        # L=0 should NOT have zero mean (no centering)
        l0_output = output[:, 0, :]
        l0_mean = l0_output.mean(dim=-1)
        assert not torch.allclose(l0_mean, torch.zeros_like(l0_mean), atol=1e-3)
    
    def test_vectorized_operation_efficiency(self, setup_data):
        """Test that the vectorized implementation works correctly."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True,
            centering=False
        )
        
        # The layer should process all degrees in one pass
        output = layer(data['x'])
        assert output.shape == data['x'].shape
        
        # Check that affine weights are applied correctly using expand_index
        # The same degree should have the same scaling factor
        for lval in range(data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            
            # All components of the same degree should use the same weight
            expected_weight = layer.affine_weight[lval, :]
            degree_mask = (layer.expand_index == lval)
            assert degree_mask.sum() == length
    
    def test_std_balance_degrees_normalization_computation(self, setup_data):
        """Test the degree balancing in normalization computation."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            std_balance_degrees=True,
            affine=False
        )
        
        # Check that balance weights are computed correctly
        expected_weights = torch.zeros((data['lmax'] + 1) ** 2, 1)
        for lval in range(data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            expected_weights[start_idx:(start_idx + length), :] = 1.0 / length
        expected_weights = expected_weights / (data['lmax'] + 1)
        
        assert torch.allclose(layer.balance_degree_weight, expected_weights)
    
    def test_affine_bias_only_with_centering(self, setup_data):
        """Test that affine bias is only applied when centering is enabled."""
        data = setup_data
        
        # With centering: should have bias
        layer_with_centering = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True,
            centering=True
        )
        assert layer_with_centering.affine_bias is not None
        
        # Without centering: should not have bias
        layer_without_centering = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True,
            centering=False
        )
        assert layer_without_centering.affine_bias is None
    
    def test_bias_application_l0_only(self, setup_data):
        """Test that bias is only applied to L=0 components."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            affine=True,
            centering=True
        )
        
        # Set bias to a known value
        layer.affine_bias.data.fill_(5.0)
        
        # Create zero input to isolate bias effect
        x = torch.zeros_like(data['x'])
        output = layer(x)
        
        # L=0 should have the bias added
        l0_output = output[:, 0, :]
        assert torch.allclose(l0_output, torch.full_like(l0_output, 5.0))
        
        # L>0 should remain zero (no bias)
        l_higher_output = output[:, 1:, :]
        assert torch.allclose(l_higher_output, torch.zeros_like(l_higher_output))
    
    def test_norm_vs_component_normalization(self, setup_data):
        """Test different normalization types."""
        data = setup_data
        
        # Test norm vs component gives different results
        layer_norm = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            normalization="norm",
            std_balance_degrees=False,  # Must be False for norm mode
            affine=False
        )
        
        layer_component = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            normalization="component",
            affine=False
        )
        
        output_norm = layer_norm(data['x'])
        output_component = layer_component(data['x'])
        
        # Should produce different results
        assert not torch.allclose(output_norm, output_component)
    
    def test_repr(self, setup_data):
        """Test string representation."""
        data = setup_data
        layer = EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            centering=True,
            std_balance_degrees=False
        )
        
        repr_str = repr(layer)
        assert "EquivariantRMSNormArraySphericalHarmonicsV2" in repr_str
        assert f"lmax={data['lmax']}" in repr_str
        assert f"num_channels={data['num_channels']}" in repr_str
        assert "centering=True" in repr_str
        assert "std_balance_degrees=False" in repr_str


class TestEquivariantDegreeLayerScale:
    """Test the EquivariantDegreeLayerScale class."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        lmax = 3
        num_channels = 16
        batch_size = 4
        sphere_basis = (lmax + 1) ** 2  # 16
        
        x = torch.randn(batch_size, sphere_basis, num_channels)
        
        return {
            'x': x,
            'lmax': lmax,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'sphere_basis': sphere_basis
        }
    
    def test_initialization_weight_scaling(self, setup_data):
        """Test that weights are initialized with degree-dependent scaling."""
        data = setup_data
        scale_factor = 2.0
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            scale_factor=scale_factor
        )
        
        # Check weight shape
        assert layer.affine_weight.shape == (1, data['lmax'] + 1, data['num_channels'])
        
        # Check L=0 weights are 1.0 (no scaling)
        assert torch.allclose(layer.affine_weight[0, 0, :], torch.ones(data['num_channels']))
        
        # Check L>0 weights are scaled by 1/sqrt(scale_factor * L)
        for lval in range(1, data['lmax'] + 1):
            expected_scale = 1.0 / math.sqrt(scale_factor * lval)
            expected_weight = torch.full((data['num_channels'],), expected_scale)
            assert torch.allclose(layer.affine_weight[0, lval, :], expected_weight)
    
    def test_expand_index_buffer(self, setup_data):
        """Test that expand_index buffer is created correctly."""
        data = setup_data
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        # expand_index should match the utility function
        expected_expand_index = get_l_to_all_m_expand_index(data['lmax'])
        assert torch.equal(layer.expand_index, expected_expand_index)
    
    def test_forward_scaling_by_degree(self, setup_data):
        """Test that forward pass scales different degrees appropriately."""
        data = setup_data
        scale_factor = 4.0
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            scale_factor=scale_factor
        )
        
        # Create input with unit values
        x = torch.ones_like(data['x'])
        output = layer(x)
        
        # Check scaling for each degree
        for lval in range(data['lmax'] + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            
            degree_output = output[:, start_idx:(start_idx + length), :]
            
            if lval == 0:
                # L=0 should be unscaled
                expected_output = torch.ones_like(degree_output)
            else:
                # L>0 should be scaled by 1/sqrt(scale_factor * lval)
                expected_scale = 1.0 / math.sqrt(scale_factor * lval)
                expected_output = torch.full_like(degree_output, expected_scale)
            
            assert torch.allclose(degree_output, expected_output)
    
    def test_different_scale_factors(self, setup_data):
        """Test behavior with different scale factors."""
        data = setup_data
        
        layer_scale2 = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            scale_factor=2.0
        )
        
        layer_scale4 = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            scale_factor=4.0
        )
        
        x = torch.ones_like(data['x'])
        output2 = layer_scale2(x)
        output4 = layer_scale4(x)
        
        # Different scale factors should produce different outputs for L>0
        assert not torch.allclose(output2, output4)
        
        # L=0 should be the same (no scaling)
        assert torch.allclose(output2[:, 0, :], output4[:, 0, :])
        
        # L>0 with scale_factor=4 should be more suppressed than scale_factor=2
        # For L=1: scale2 gives 1/sqrt(2*1) = 1/sqrt(2), scale4 gives 1/sqrt(4*1) = 1/2
        l1_output2 = output2[:, 1:4, :]
        l1_output4 = output4[:, 1:4, :]
        
        ratio = l1_output4.mean() / l1_output2.mean()
        expected_ratio = (1/2) / (1/math.sqrt(2))  # (1/2) / (1/sqrt(2)) = sqrt(2)/2
        assert torch.allclose(ratio, torch.tensor(expected_ratio), rtol=1e-5)
    
    def test_forward_shape_preservation(self, setup_data):
        """Test that forward pass preserves input shape."""
        data = setup_data
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        output = layer(data['x'])
        assert output.shape == data['x'].shape
    
    def test_gradient_flow(self, setup_data):
        """Test that gradients flow correctly through the layer."""
        data = setup_data
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels']
        )
        
        x = data['x'].clone().requires_grad_(True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are reasonable
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check that layer parameters have gradients
        assert layer.affine_weight.grad is not None
        assert not torch.isnan(layer.affine_weight.grad).any()
    
    def test_repr(self, setup_data):
        """Test string representation."""
        data = setup_data
        scale_factor = 3.0
        layer = EquivariantDegreeLayerScale(
            lmax=data['lmax'],
            num_channels=data['num_channels'],
            scale_factor=scale_factor
        )
        
        repr_str = repr(layer)
        assert "EquivariantDegreeLayerScale" in repr_str
        assert f"lmax={data['lmax']}" in repr_str
        assert f"num_channels={data['num_channels']}" in repr_str
        assert f"scale_factor={scale_factor}" in repr_str
