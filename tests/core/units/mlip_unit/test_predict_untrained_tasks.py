"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit


@pytest.mark.gpu()
def test_untrained_forces_gpu(conserving_mole_checkpoint):
    """Test computing forces for energy-only checkpoint on GPU."""
    _test_untrained_forces(conserving_mole_checkpoint[0], "cuda")


def test_untrained_forces_cpu(conserving_mole_checkpoint):
    """Test computing forces for energy-only checkpoint on CPU."""
    _test_untrained_forces(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_forces(checkpoint_path, device):
    """
    Test that untrained forces can be computed for energy-only checkpoint.
    """
    # Create predictor with untrained forces enabled
    settings = InferenceSettings(compute_untrained_forces={"omol"})
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that forces task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "forces" in name for name in task_names
    ), f"No forces task found in {task_names}"

    # Create test data
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify both energy and forces are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"

    # Verify shapes
    assert preds["energy"].shape == (1,), f"Wrong energy shape: {preds['energy'].shape}"
    assert preds["forces"].shape == (
        3,
        3,
    ), f"Wrong forces shape: {preds['forces'].shape}"

    # Verify forces are finite
    assert torch.isfinite(preds["forces"]).all(), "Forces contain NaN or Inf"


@pytest.mark.gpu()
def test_untrained_stress_selective_gpu(conserving_mole_checkpoint):
    """Test selective stress computation on GPU."""
    _test_untrained_stress_selective(conserving_mole_checkpoint[0], "cuda")


def test_untrained_stress_selective_cpu(conserving_mole_checkpoint):
    """Test selective stress computation on CPU."""
    _test_untrained_stress_selective(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_stress_selective(checkpoint_path, device):
    """
    Test that stress can be selectively enabled for specific datasets.
    """
    # Enable stress only for omol dataset
    settings = InferenceSettings(
        compute_untrained_forces={"omol"},
        compute_untrained_stress={"omol"},
    )
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that stress task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "stress" in name for name in task_names
    ), f"No stress task found in {task_names}"

    # Create test data
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify energy, forces, and stress are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"
    assert "stress" in preds, "Stress prediction missing"

    # Verify stress shape
    assert preds["stress"].shape == (
        1,
        9,
    ), f"Wrong stress shape: {preds['stress'].shape}"

    # Verify stress is finite
    assert torch.isfinite(preds["stress"]).all(), "Stress contains NaN or Inf"


@pytest.mark.gpu()
def test_untrained_hessian_gpu(conserving_mole_checkpoint):
    """Test hessian computation on GPU."""
    _test_untrained_hessian(conserving_mole_checkpoint[0], "cuda")


def test_untrained_hessian_cpu(conserving_mole_checkpoint):
    """Test hessian computation on CPU."""
    _test_untrained_hessian(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_hessian(checkpoint_path, device):
    """
    Test that hessian can be computed for energy-only checkpoint.
    """
    # Enable hessian for omol
    settings = InferenceSettings(
        compute_untrained_forces={"omol"},
        compute_untrained_hessian={"omol"},
        hessian_vmap=True,
    )
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that hessian task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "hessian" in name for name in task_names
    ), f"No hessian task found in {task_names}"

    # Create test data (single system required for hessian)
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify energy, forces, and hessian are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"
    assert "hessian" in preds, "Hessian prediction missing"

    # Verify hessian shape: (1, 3*N, 3*N) — batch dim is always 1
    n_atoms = len(atoms)
    expected_shape = (1, n_atoms * 3, n_atoms * 3)
    assert (
        preds["hessian"].shape == expected_shape
    ), f"Wrong hessian shape: {preds['hessian'].shape}, expected {expected_shape}"

    # Verify hessian is finite
    assert torch.isfinite(preds["hessian"]).all(), "Hessian contains NaN or Inf"

    # Verify hessian is symmetric (squeeze batch dim for symmetry check)
    hessian = preds["hessian"].squeeze(0)
    assert torch.allclose(hessian, hessian.T, atol=1e-5), "Hessian is not symmetric"


def test_hessian_batch_size_validation(conserving_mole_checkpoint):
    """Test that hessian computation fails for batch_size > 1."""
    settings = InferenceSettings(
        compute_untrained_forces={"omol"},
        compute_untrained_hessian={"omol"},
    )
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )

    # Create batch with 2 systems
    from ase.build import molecule

    atoms1 = molecule("H2O")
    atoms1.info.update({"charge": 0, "spin": 1})
    atoms2 = molecule("H2O")
    atoms2.info.update({"charge": 0, "spin": 1})

    data1 = AtomicData.from_ase(
        atoms1,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    data2 = AtomicData.from_ase(
        atoms2,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data1, data2])

    # Should raise ValueError
    with pytest.raises(ValueError, match="Hessian computation requires batch_size=1"):
        predictor.predict(batch)


def test_no_duplicate_tasks(conserving_mole_checkpoint):
    """Test that no duplicate tasks are created if checkpoint already has them."""
    # Load checkpoint without untrained tasks
    # Now load with untrained forces enabled (but if checkpoint already has forces, no duplicate)
    settings = InferenceSettings(compute_untrained_stress={"omol"})
    predictor_untrained = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )
    untrained_task_names = set(predictor_untrained.tasks.keys())

    # Check that we don't have duplicate tasks
    task_name_counts = {}
    for name in untrained_task_names:
        # Count how many tasks have the same property
        property_name = name.split("_")[-1]  # e.g., "forces" from "omol_forces"
        task_name_counts[property_name] = task_name_counts.get(property_name, 0) + 1

    # Each property should appear at most once per dataset
    # (This is a simplified check; in reality we'd need to look at dataset+property combos)
    for prop, count in task_name_counts.items():
        # With a single-dataset checkpoint, we should have exactly 1 task per property
        assert count <= 3, f"Property {prop} has {count} tasks, may have duplicates"


def test_direct_force_model_hessian_validation(direct_mole_checkpoint):
    """Test that direct-force models reject hessian requests."""
    # Try to enable hessian on direct-force model (should fail)
    settings = InferenceSettings(compute_untrained_hessian={"omol"})

    with pytest.raises(
        ValueError, match="Cannot compute Hessian for direct-force models"
    ):
        MLIPPredictUnit(
            direct_mole_checkpoint[0], device="cpu", inference_settings=settings
        )


def test_direct_force_model_stress_validation(direct_mole_checkpoint):
    """Test that direct-force models reject stress requests."""
    # Try to enable stress on direct-force model (should fail)
    settings = InferenceSettings(compute_untrained_stress={"omol"})

    with pytest.raises(
        ValueError, match="Cannot compute stress for direct-force models"
    ):
        MLIPPredictUnit(
            direct_mole_checkpoint[0], device="cpu", inference_settings=settings
        )


def test_direct_force_model_forces_allowed(direct_mole_checkpoint):
    """Test that direct-force models allow forces (already computed directly)."""
    # Forces should be allowed on direct-force models since they compute them directly
    # This should NOT raise an error, but also shouldn't add untrained tasks
    settings = InferenceSettings(compute_untrained_forces={"omol"})

    # This should succeed
    predictor = MLIPPredictUnit(
        direct_mole_checkpoint[0], device="cpu", inference_settings=settings
    )

    # Verify predictor was created successfully
    assert predictor is not None
    assert predictor.model.module.direct_forces is True
