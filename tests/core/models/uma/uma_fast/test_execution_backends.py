"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Validation tests for execution backends.

Tests that backend validation correctly accepts/rejects model configurations.
E2E accuracy tests are done via run_benchmarks.sh and compare_forces.py scripts.
"""

from __future__ import annotations

import pytest

from fairchem.core.models.uma.triton import HAS_TRITON

# =============================================================================
# Tests: Validation Errors
# =============================================================================


class MockEdgeDegreeEmbedding:
    """Mock edge_degree_embedding with activation_checkpoint_chunk_size=None."""

    activation_checkpoint_chunk_size = None


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_requires_correct_lmax():
    """
    Verify that umas_fast_gpu raises ValueError for incorrect lmax.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    # Create a mock model with wrong lmax
    class MockModelWrongLmax:
        lmax = 3  # Wrong - should be 2
        mmax = 2
        sphere_channels = 128
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFastGPUBackend.validate(MockModelWrongLmax())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_requires_correct_mmax():
    """
    Verify that umas_fast_gpu raises ValueError for incorrect mmax.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    # Create a mock model with wrong mmax
    class MockModelWrongMmax:
        lmax = 2
        mmax = 1  # Wrong - should be 2
        sphere_channels = 128
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFastGPUBackend.validate(MockModelWrongMmax())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_requires_sphere_channels_divisible_by_128():
    """
    Verify that umas_fast_gpu raises ValueError for incorrect sphere_channels.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    # Create a mock model with wrong sphere_channels
    class MockModelWrongChannels:
        lmax = 2
        mmax = 2
        sphere_channels = 100  # Wrong - should be divisible by 128
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    with pytest.raises(ValueError, match="divisible by 128"):
        UMASFastGPUBackend.validate(MockModelWrongChannels())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_accepts_correct_config():
    """
    Verify that umas_fast_gpu validation passes for correct model config.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    # Create a mock model with correct parameters
    class MockModel:
        lmax = 2
        mmax = 2
        sphere_channels = 128
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    # Should not raise
    UMASFastGPUBackend.validate(MockModel())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_accepts_512_channels():
    """
    Verify that umas_fast_gpu validation passes for sphere_channels=512.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    # Create a mock model with 512 channels (like UMA-S)
    class MockModel:
        lmax = 2
        mmax = 2
        sphere_channels = 512
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    # Should not raise
    UMASFastGPUBackend.validate(MockModel())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_umas_fast_gpu_validation_requires_merge_mole():
    """
    Verify that umas_fast_gpu raises ValueError when merge_mole=False.
    """
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    class MockModel:
        lmax = 2
        mmax = 2
        sphere_channels = 128
        edge_degree_embedding = MockEdgeDegreeEmbedding()

    class MockSettings:
        activation_checkpointing = False
        merge_mole = False  # Wrong - should be True

    with pytest.raises(ValueError, match="merge_mole=True"):
        UMASFastGPUBackend.validate(MockModel(), MockSettings())
