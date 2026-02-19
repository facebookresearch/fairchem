"""
Tests for SphericalSmearing class.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fairchem.core.models.utils.basis import SphericalSmearing


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in all tests."""
    torch.manual_seed(42)
    np.random.seed(42)


def test_spherical_smearing_comprehensive():
    """Comprehensive test covering basic functionality, options, and mathematical correctness."""
    # Test all three options with max_n=2 for speed
    for option in ["all", "sine", "cosine"]:
        smearing = SphericalSmearing(max_n=2, option=option)

        # Test basic shapes and dimensions
        xyz = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        output = smearing(xyz)

        assert output.shape == (3, smearing.out_dim)
        expected_out_dim = int(np.sum(smearing.m == 0) + 2 * np.sum(smearing.m != 0))
        assert smearing.out_dim == expected_out_dim
        assert len(smearing.m) == len(smearing.n)
        assert torch.isfinite(output).all()

        # Test normalization: different magnitudes, same direction should give same output
        xyz_normalized = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        xyz_scaled = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
        assert torch.allclose(smearing(xyz_normalized), smearing(xyz_scaled), atol=1e-6)

        # Test m/n pairing: m must satisfy 0 <= m <= n
        for n_val, m_val in zip(smearing.n, smearing.m):
            assert 0 <= m_val <= n_val, f"Invalid pairing in {option} mode: m={m_val}, n={n_val}"

    # Test option filtering logic
    all_sm = SphericalSmearing(max_n=2, option="all")
    sine_sm = SphericalSmearing(max_n=2, option="sine")
    cosine_sm = SphericalSmearing(max_n=2, option="cosine")

    assert all(n % 2 == 1 for n in sine_sm.n), f"Sine mode has even n values: {sine_sm.n}"
    assert all(n % 2 == 0 for n in cosine_sm.n), f"Cosine mode has odd n values: {cosine_sm.n}"
    assert len(sine_sm.n) + len(cosine_sm.n) == len(all_sm.n)

    # Test Y_00 constant: Y_00 = 1/sqrt(4π) ≈ 0.282 for any normalized vector
    y00_expected = 1.0 / np.sqrt(4 * np.pi)
    assert torch.allclose(output[:, 0], torch.tensor(y00_expected, dtype=torch.float32), atol=1e-3)


def test_spherical_smearing_edge_cases():
    """Test edge cases: axis-aligned vectors and batch processing."""
    smearing = SphericalSmearing(max_n=2, option="all")

    # Test axis-aligned vectors (including negative directions)
    xyz = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=torch.float32)
    output = smearing(xyz)

    assert output.shape == (6, smearing.out_dim)
    assert torch.isfinite(output).all()

    # Test batch processing with random vectors
    xyz_random = torch.randn(5, 3, dtype=torch.float32)
    xyz_random = xyz_random / xyz_random.norm(dim=-1, keepdim=True)
    output_random = smearing(xyz_random)

    assert output_random.shape == (5, smearing.out_dim)
    assert torch.isfinite(output_random).all()
