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

# =============================================================================
# Tests: Validation Errors
# =============================================================================


class MockEdgeDegreeEmbedding:
    """Mock edge_degree_embedding with activation_checkpoint_chunk_size=None."""

    activation_checkpoint_chunk_size = None


@pytest.mark.gpu()
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


@pytest.mark.gpu()
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


@pytest.mark.gpu()
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


@pytest.mark.gpu()
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


@pytest.mark.gpu()
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


@pytest.mark.gpu()
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


# =============================================================================
# Tests: E2E Force Correctness
# =============================================================================


@pytest.mark.gpu()
def test_umas_fast_pytorch_forces_match_baseline_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_pytorch produces forces matching general backend.

    Uses PBC system from fake_uma_dataset (oc20, 5-20 atoms).
    """
    import os

    import torch

    from fairchem.core.datasets.ase_datasets import AseDBDataset
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater
    from fairchem.core.units.mlip_unit import MLIPPredictUnit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    atoms = db.get_atoms(0)  # PBC system

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=baseline_settings
    )

    # Test (umas_fast_pytorch backend)
    test_settings = InferenceSettings(
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_pytorch",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-2, atol=5e-2
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=1e-2, atol=1e-2
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"


@pytest.mark.gpu()
def test_umas_fast_pytorch_forces_match_baseline_no_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_pytorch produces forces matching general backend.

    Uses non-PBC system from fake_uma_dataset (omol, 2-5 atoms).
    """
    import os

    import torch

    from fairchem.core.datasets.ase_datasets import AseDBDataset
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater
    from fairchem.core.units.mlip_unit import MLIPPredictUnit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "omol")})
    atoms = db.get_atoms(0)  # Non-PBC molecule
    atoms.pbc = [False, False, False]

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "omol"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=baseline_settings
    )

    # Test (umas_fast_pytorch backend)
    test_settings = InferenceSettings(
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_pytorch",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-2, atol=5e-2
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=1e-2, atol=1e-2
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"


# =============================================================================
# Tests: Triton Autograd Gradcheck
# =============================================================================


@pytest.mark.gpu()
def test_node_to_edge_wigner_permute_gradcheck():
    """
    Verify NodeToEdgeWignerPermuteFunction backward pass is correct via gradcheck.

    Uses fast_mode=True for statistical gradient validation (random projections)
    instead of full Jacobian computation to avoid OOM.
    """
    import torch

    from fairchem.core.models.uma.triton.node_to_edge_wigner_permute import (
        NodeToEdgeWignerPermuteFunction,
    )

    torch.manual_seed(42)
    device = "cuda"
    num_nodes = 8
    num_edges = 16
    sphere_channels = 128  # Minimum for kernel block size

    # Create test inputs
    x = torch.randn(
        num_nodes, 9, sphere_channels, device=device, dtype=torch.float64
    ).requires_grad_(True)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_tgt = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)

    # Gradcheck with fast_mode to avoid full Jacobian OOM
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: NodeToEdgeWignerPermuteFunction.apply(
            x_in, edge_index, w_in
        ),
        (x, wigner),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )


@pytest.mark.gpu()
def test_permute_wigner_inv_edge_to_node_gradcheck():
    """
    Verify PermuteWignerInvEdgeToNodeFunction backward pass is correct via gradcheck.

    Uses fast_mode=True for statistical gradient validation (random projections)
    instead of full Jacobian computation to avoid OOM.
    """
    import torch

    from fairchem.core.models.uma.triton.permute_wigner_inv_edge_to_node import (
        PermuteWignerInvEdgeToNodeFunction,
    )

    torch.manual_seed(42)
    device = "cuda"
    num_edges = 16
    sphere_channels = 128  # Minimum for kernel block size

    # Create test inputs
    x = torch.randn(
        num_edges, 9, sphere_channels, device=device, dtype=torch.float64
    ).requires_grad_(True)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)

    # Gradcheck with fast_mode to avoid full Jacobian OOM
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: PermuteWignerInvEdgeToNodeFunction.apply(x_in, w_in),
        (x, wigner),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )
