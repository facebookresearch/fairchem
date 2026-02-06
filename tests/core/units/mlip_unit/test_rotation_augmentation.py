"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for rotation augmentation in MLIPPredictUnit to avoid Wigner singularities.
"""

from __future__ import annotations

import os
from functools import partial

import pytest
import torch

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit
from fairchem.core.units.mlip_unit.predict import (
    _generate_rotation_matrix,
    _inverse_rotate_output,
)


# =============================================================================
# Unit Tests for Rotation Functions (No Model Required)
# =============================================================================


class TestRotationMatrixGeneration:
    """Tests for _generate_rotation_matrix function"""

    def test_rotation_matrix_is_orthogonal(self):
        """Verify R @ R.T = I (orthogonality)"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        identity = torch.eye(3, dtype=torch.float32)
        assert torch.allclose(R @ R.T, identity, atol=1e-6)
        assert torch.allclose(R.T @ R, identity, atol=1e-6)

    def test_rotation_matrix_is_proper_rotation(self):
        """Verify det(R) = +1 (proper rotation, not reflection)"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        det = torch.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-6)

    def test_rotation_is_deterministic(self):
        """Same seed should produce identical rotation matrix"""
        R1 = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        R2 = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        assert torch.allclose(R1, R2)

    def test_different_seeds_produce_different_rotations(self):
        """Different seeds should produce different rotations"""
        R1 = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        R2 = _generate_rotation_matrix(seed=123, device=torch.device("cpu"), dtype=torch.float32)
        assert not torch.allclose(R1, R2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rotation_matrix_dtype(self, dtype):
        """Rotation matrix should have correct dtype"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=dtype)
        assert R.dtype == dtype


class TestForceRotationRoundtrip:
    """Tests for force vector rotation and inverse rotation"""

    def test_force_roundtrip(self):
        """f -> rotate -> inverse_rotate -> f should be identity"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        forces_orig = torch.randn(10, 3)

        # Forward rotation: f' = f @ R.T
        forces_rotated = forces_orig @ R.T
        # Inverse rotation via _inverse_rotate_output
        forces_back = _inverse_rotate_output(forces_rotated, "forces", R)

        assert torch.allclose(forces_orig, forces_back, atol=1e-6)

    def test_force_rotation_preserves_magnitude(self):
        """Rotation should preserve force magnitudes"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        forces_orig = torch.randn(10, 3)

        # Forward rotation
        forces_rotated = forces_orig @ R.T

        # Magnitudes should be identical
        mag_orig = torch.norm(forces_orig, dim=1)
        mag_rotated = torch.norm(forces_rotated, dim=1)
        assert torch.allclose(mag_orig, mag_rotated, atol=1e-6)


class TestStressRotationRoundtrip:
    """Tests for stress tensor rotation and inverse rotation"""

    def test_stress_roundtrip(self):
        """σ -> rotate -> inverse_rotate -> σ should be identity"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)

        # Create symmetric stress tensor (physical stress is symmetric)
        stress_3x3 = torch.randn(3, 3)
        stress_3x3 = (stress_3x3 + stress_3x3.T) / 2
        stress_flat = stress_3x3.view(1, 9)

        # Forward rotation: σ' = R @ σ @ R.T
        stress_rotated_3x3 = R @ stress_3x3 @ R.T
        stress_rotated_flat = stress_rotated_3x3.view(1, 9)

        # Inverse rotation via _inverse_rotate_output
        stress_back = _inverse_rotate_output(stress_rotated_flat, "stress", R)

        assert torch.allclose(stress_flat, stress_back, atol=1e-6)

    def test_stress_symmetry_preserved(self):
        """Rotation should preserve symmetry of stress tensor"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)

        # Create symmetric stress tensor
        stress_3x3 = torch.randn(3, 3)
        stress_3x3 = (stress_3x3 + stress_3x3.T) / 2

        # Rotate
        stress_rotated = R @ stress_3x3 @ R.T

        # Should still be symmetric
        assert torch.allclose(stress_rotated, stress_rotated.T, atol=1e-6)

    def test_stress_trace_preserved(self):
        """Rotation should preserve trace (invariant of rank-2 tensor)"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)

        stress_3x3 = torch.randn(3, 3)
        stress_3x3 = (stress_3x3 + stress_3x3.T) / 2

        trace_orig = torch.trace(stress_3x3)
        stress_rotated = R @ stress_3x3 @ R.T
        trace_rotated = torch.trace(stress_rotated)

        assert torch.allclose(trace_orig, trace_rotated, atol=1e-6)

    def test_stress_batch_roundtrip(self):
        """Test rotation roundtrip for batched stress tensors"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        batch_size = 5

        # Create batch of symmetric stress tensors
        stress_batch = torch.randn(batch_size, 3, 3)
        stress_batch = (stress_batch + stress_batch.transpose(-1, -2)) / 2
        stress_flat = stress_batch.view(batch_size, 9)

        # Forward rotation for batch
        stress_rotated = R @ stress_batch @ R.T
        stress_rotated_flat = stress_rotated.view(batch_size, 9)

        # Inverse rotation
        stress_back = _inverse_rotate_output(stress_rotated_flat, "stress", R)

        assert torch.allclose(stress_flat, stress_back, atol=1e-6)


class TestEnergyInvariance:
    """Test that energy (scalar) is unchanged by rotation functions"""

    def test_energy_unchanged_by_inverse_rotate(self):
        """Energy should be returned unchanged"""
        R = _generate_rotation_matrix(seed=42, device=torch.device("cpu"), dtype=torch.float32)
        energy = torch.tensor([1.5, 2.3, -0.7])

        energy_result = _inverse_rotate_output(energy, "energy", R)

        assert torch.allclose(energy, energy_result)


# =============================================================================
# Integration Tests (Require Model Fixtures)
# =============================================================================


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_energy_invariance_under_rotation(
    seed, conserving_mole_checkpoint, fake_uma_dataset
):
    """Energy predictions should be identical with and without rotation augmentation"""
    checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    # Predictor without rotation
    settings_no_rot = InferenceSettings(rotation_seed=None)
    predictor_no_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_no_rot)

    # Predictor with rotation
    settings_rot = InferenceSettings(rotation_seed=seed)
    predictor_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_rot)

    for sample_idx in range(3):
        sample = a2g(db.get_atoms(sample_idx), task_name="oc20")
        batch = data_list_collater([sample], otf_graph=True)

        E_no_rot = predictor_no_rot.predict(batch)["energy"]
        E_rot = predictor_rot.predict(batch)["energy"]

        assert torch.allclose(E_no_rot, E_rot, rtol=1e-4, atol=1e-6), \
            f"Energy mismatch: no_rot={E_no_rot.item()}, rot={E_rot.item()}"


@pytest.mark.parametrize("seed", [42, 123])
def test_forces_equivariance_under_rotation(
    seed, conserving_mole_checkpoint, fake_uma_dataset
):
    """Force predictions should match after internal back-rotation"""
    checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    # Predictor without rotation
    settings_no_rot = InferenceSettings(rotation_seed=None)
    predictor_no_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_no_rot)

    # Predictor with rotation
    settings_rot = InferenceSettings(rotation_seed=seed)
    predictor_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_rot)

    for sample_idx in range(3):
        sample = a2g(db.get_atoms(sample_idx), task_name="oc20")
        batch = data_list_collater([sample], otf_graph=True)

        F_no_rot = predictor_no_rot.predict(batch)["forces"]
        F_rot = predictor_rot.predict(batch)["forces"]

        # Forces should match because back-rotation is applied internally
        assert torch.allclose(F_no_rot, F_rot, rtol=1e-4, atol=1e-5), \
            f"Force mismatch at sample {sample_idx}, max diff: {(F_no_rot - F_rot).abs().max()}"


def test_multiple_seeds_give_consistent_results(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """Different rotation seeds should all give same physical result"""
    checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater([sample], otf_graph=True)

    seeds = [42, 123, 456, 789, 12345]
    energies = []
    forces_list = []

    for seed in seeds:
        settings = InferenceSettings(rotation_seed=seed)
        predictor = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings)
        result = predictor.predict(batch)
        energies.append(result["energy"])
        forces_list.append(result["forces"])

    # All energies should match
    for E in energies[1:]:
        assert torch.allclose(energies[0], E, rtol=1e-4, atol=1e-6)

    # All forces should match
    for F in forces_list[1:]:
        assert torch.allclose(forces_list[0], F, rtol=1e-4, atol=1e-5)


def test_rotation_none_is_identity(conserving_mole_checkpoint, fake_uma_dataset):
    """rotation_seed=None should give identical results on repeated calls"""
    checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    settings = InferenceSettings(rotation_seed=None)
    predictor = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings)

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater([sample], otf_graph=True)

    # Multiple predictions should be numerically identical
    results = [predictor.predict(batch) for _ in range(3)]

    for r in results[1:]:
        assert torch.allclose(results[0]["energy"], r["energy"])
        assert torch.allclose(results[0]["forces"], r["forces"])


# =============================================================================
# Direct Force Model Tests
# =============================================================================


@pytest.mark.parametrize("seed", [42])
def test_direct_forces_with_rotation(seed, direct_checkpoint, fake_uma_dataset):
    """Direct force models should also work with rotation augmentation"""
    checkpoint_path, _ = direct_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    # Predictor without rotation
    settings_no_rot = InferenceSettings(rotation_seed=None)
    predictor_no_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_no_rot)

    # Predictor with rotation
    settings_rot = InferenceSettings(rotation_seed=seed)
    predictor_rot = MLIPPredictUnit(checkpoint_path, device="cpu", inference_settings=settings_rot)

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater([sample], otf_graph=True)

    result_no_rot = predictor_no_rot.predict(batch)
    result_rot = predictor_rot.predict(batch)

    # Energy should match
    assert torch.allclose(result_no_rot["energy"], result_rot["energy"], rtol=1e-4, atol=1e-6)

    # Forces should match (direct forces also need back-rotation)
    assert torch.allclose(result_no_rot["forces"], result_rot["forces"], rtol=1e-4, atol=1e-5)
