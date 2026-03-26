"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy.testing as npt
import pytest
import ray
import torch
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate._batch import InferenceBatcher
from fairchem.core.datasets.atomic_data import AtomicData

# mark all tests in this module as serial (Ray needs serial execution due to large number of subprocesses)
pytestmark = pytest.mark.serial


@pytest.fixture(scope="module")
def uma_predict_unit():
    """Get a UMA predict unit for testing."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0])


@pytest.fixture(scope="module")
def uma_predict_unit_alt():
    """Get a different UMA predict unit for testing checkpoint swaps."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    # Require at least 2 UMA models for checkpoint swap testing
    if len(uma_models) < 2:
        raise RuntimeError(
            f"At least 2 different UMA models are required for checkpoint swap tests. "
            f"Found only {len(uma_models)}: {uma_models}"
        )
    return pretrained_mlip.get_predict_unit(uma_models[1])


def setup_ray():
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")

    # Use a unique namespace for this worker to avoid interference
    # This is especially important for parallel test execution (pytest -n 8)
    import os
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    namespace = f"test-{worker_id}-{int(time.time() * 1000000)}"

    if ray.is_initialized():
        with contextlib.suppress(Exception):
            serve.shutdown()
        with contextlib.suppress(Exception):
            ray.shutdown()
        # Give Ray time to fully shut down before reinitializing
        time.sleep(0.5)

    ray.init(
        ignore_reinit_error=True,
        namespace=namespace,
        num_cpus=16,  # Increased to support default ray_actor_options num_cpus=8
        num_gpus=1 if torch.cuda.is_available() else 0,
        logging_level="ERROR",  # Reduce noise in test output
    )


def cleanup_ray():
    """Cleanup Ray resources safely without affecting other test workers."""
    if not ray.is_initialized():
        return
        
    # CRITICAL: Must shutdown serve BEFORE ray to avoid dead actor errors
    try:
        serve.shutdown()
    except Exception as e:
        print(f"Warning: Error during serve shutdown: {e}")
    
    try:
        ray.shutdown()
    except Exception as e:
        print(f"Warning: Error during ray shutdown: {e}")


@pytest.fixture()
def inference_batcher(uma_predict_unit):
    setup_ray()
    batcher = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        concurrency_backend="threads",
        concurrency_backend_options={"max_workers": 4},
    )

    yield batcher

    cleanup_ray()


def test_initialization_with_custom_concurrency_options(uma_predict_unit):
    try:
        max_workers = 8
        batcher = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            concurrency_backend="threads",
            concurrency_backend_options={"max_workers": max_workers},
        )

        assert isinstance(batcher.executor, ThreadPoolExecutor)
    finally:
        cleanup_ray()


def test_initialization_with_ray_actor_options(uma_predict_unit):
    try:
        batcher = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            ray_actor_options={"num_cpus": 2},
        )

        assert hasattr(batcher, "predict_server_handle")
    finally:
        cleanup_ray()


def test_context_manager_enter_exit(uma_predict_unit):
    try:
        with InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
        ) as batcher:
            assert hasattr(batcher, "executor")
            assert hasattr(batcher, "predict_server_handle")
            executor = batcher.executor

        assert executor is not None

        with pytest.raises(
            RuntimeError, match="cannot schedule new futures after shutdown"
        ):
            executor.submit(time.sleep, 1)
    finally:
        cleanup_ray()


def test_batched_atomic_data_predictions(inference_batcher):
    """Test batched predictions using AtomicData directly."""
    atoms_list = [bulk("Cu"), bulk("Al"), bulk("Fe")]
    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
    ]

    with ThreadPoolExecutor(max_workers=len(atoms_list)) as executor:
        futures = [
            executor.submit(inference_batcher.batch_predict_unit.predict, data)
            for data in atomic_data_list
        ]
        results = [future.result() for future in futures]

    assert len(results) == len(atoms_list)
    for i, preds in enumerate(results):
        assert "energy" in preds
        assert "forces" in preds
        assert preds["energy"].shape == (1,)
        assert preds["forces"].shape == (len(atoms_list[i]), 3)


def test_batch_vs_serial_consistency(inference_batcher, uma_predict_unit):
    """Test that batched and serial calculations produce consistent results."""
    atoms_list = [
        bulk("Cu"),
        bulk("Al"),
        bulk("Fe"),
        bulk("Ni"),
    ]

    def calculate_properties(atoms, predict_unit):
        atoms.calc = FAIRChemCalculator(predict_unit, task_name="omat")
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
        }

    results_batched = list(
        inference_batcher.executor.map(
            partial(
                calculate_properties,
                predict_unit=inference_batcher.batch_predict_unit,
            ),
            atoms_list,
        )
    )

    results_serial = [
        calculate_properties(atoms, uma_predict_unit) for atoms in atoms_list
    ]

    assert len(results_batched) == len(results_serial)
    for r_batch, r_serial in zip(results_batched, results_serial):
        npt.assert_allclose(r_batch["energy"], r_serial["energy"], atol=1e-4)
        npt.assert_allclose(r_batch["forces"], r_serial["forces"], atol=1e-4)


def test_checkpoint_swap_with_energy_verification(uma_predict_unit, uma_predict_unit_alt):
    """Test that checkpoint swapping produces different energies and swapping back recovers originals."""
    try:
        setup_ray()
        
        # Create batcher with first model
        # Use fewer CPUs to allow room for checkpoint swaps and multiple operations
        batcher = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
            concurrency_backend="threads",
            concurrency_backend_options={"max_workers": 4},
            ray_actor_options={"num_cpus": 4},  # Use fewer CPUs to allow room for swaps
        )

        # Test atoms
        atoms_list = [bulk("Cu"), bulk("Al")]
        
        # Get energies with first checkpoint
        energies_initial = []
        for atoms in atoms_list:
            atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
            energies_initial.append(atoms.get_potential_energy())
        
        # Swap to different checkpoint
        batcher.update_checkpoint(uma_predict_unit_alt)
        time.sleep(0.2)  # Give time for checkpoint swap to complete
        
        # Get energies with swapped checkpoint - should be different
        energies_swapped = []
        for atoms in atoms_list:
            atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
            energies_swapped.append(atoms.get_potential_energy())
        
        # Verify energies are different between models
        for e_initial, e_swapped in zip(energies_initial, energies_swapped):
            assert abs(e_initial - e_swapped) > 1e-5, f"Energies should differ between models but got {e_initial} and {e_swapped}"
        
        # Swap back to original checkpoint
        batcher.update_checkpoint(uma_predict_unit)
        time.sleep(0.2)  # Give time for checkpoint swap to complete
        
        # Get energies with original checkpoint - should match initial
        energies_restored = []
        for atoms in atoms_list:
            atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
            energies_restored.append(atoms.get_potential_energy())
        
        # Verify energies match original
        npt.assert_allclose(energies_initial, energies_restored, atol=1e-4)
        
        batcher.shutdown(shutdown_ray=False)
    finally:
        cleanup_ray()


def test_ray_server_complete_shutdown_and_restart(uma_predict_unit):
    """Test complete shutdown and restart of the Ray server."""
    try:
        setup_ray()
        
        # Create first batcher
        batcher1 = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
        )
        
        # Test atoms
        atoms = bulk("Cu")
        atoms.calc = FAIRChemCalculator(batcher1.batch_predict_unit, task_name="omat")
        energy_before_shutdown = atoms.get_potential_energy()
        
        # Shutdown batcher and Ray explicitly
        batcher1.shutdown(shutdown_ray=True)
        cleanup_ray()
        
        # Verify Ray is shut down
        assert not ray.is_initialized(), "Ray should be shut down"
        
        # Reinitialize Ray and create new batcher
        setup_ray()
        
        batcher2 = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
        )
        
        # Calculate with new batcher - should get same energy
        atoms_new = bulk("Cu")
        atoms_new.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
        energy_after_restart = atoms_new.get_potential_energy()
        
        # Verify energies match
        npt.assert_allclose(energy_before_shutdown, energy_after_restart, atol=1e-4)
        
        batcher2.shutdown(shutdown_ray=False)
    finally:
        cleanup_ray()


def test_batcher_shutdown_method(uma_predict_unit):
    """Test that the shutdown method properly cleans up resources."""
    try:
        setup_ray()
        
        # Verify Ray is initialized
        assert ray.is_initialized(), "Ray should be initialized before creating batcher"
        
        # Create a batcher
        batcher = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
        )
        
        # Verify batcher is initialized
        assert hasattr(batcher, "executor")
        assert hasattr(batcher, "predict_server_handle")
        assert batcher.predict_server_handle is not None
        executor = batcher.executor
        
        # Shutdown the batcher with Ray shutdown
        batcher.shutdown(shutdown_ray=True)
        
        # Verify executor is shutdown (should not be able to submit new tasks)
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            executor.submit(time.sleep, 1)
        
        # Verify predict_server_handle is cleared
        assert batcher.predict_server_handle is None
        
        # Verify Ray server is shut down
        assert not ray.is_initialized(), "Ray server should be shut down after shutdown(shutdown_ray=True)"
        
    finally:
        cleanup_ray()


def test_batcher_delete_method(uma_predict_unit):
    """Test that the delete method removes deployment but keeps Ray running."""
    try:
        setup_ray()
        
        # Create a batcher
        batcher = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
        )
        
        # Verify batcher is initialized
        assert batcher.predict_server_handle is not None
        deployment_name = batcher.deployment_name
        
        # Delete the deployment
        batcher.delete()
        
        # Verify predict_server_handle is cleared
        assert batcher.predict_server_handle is None
        
        # Verify Ray is still running
        assert ray.is_initialized(), "Ray should still be running after delete()"
        
        # Cleanup executor manually since we only called delete()
        if hasattr(batcher, "executor"):
            batcher.executor.shutdown(wait=True)
        
    finally:
        cleanup_ray()


def test_multiple_concurrent_batchers(uma_predict_unit, uma_predict_unit_alt):
    """Test that multiple InferenceBatchers can coexist on the same Ray cluster."""
    try:
        setup_ray()
        
        # Create two batchers with different models
        # Use fewer CPUs per batcher to allow both to run concurrently
        batcher1 = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
            ray_actor_options={"num_cpus": 4},  # Use fewer CPUs to allow concurrent batchers
        )
        
        batcher2 = InferenceBatcher(
            predict_unit=uma_predict_unit_alt,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
            ray_actor_options={"num_cpus": 4},  # Use fewer CPUs to allow concurrent batchers
        )
        
        # Verify both have unique deployment names
        assert batcher1.deployment_name != batcher2.deployment_name, "Deployment names should be unique"
        
        # Test atoms
        atoms = bulk("Cu")
        
        # Get energy from first batcher
        atoms1 = atoms.copy()
        atoms1.calc = FAIRChemCalculator(batcher1.batch_predict_unit, task_name="omat")
        energy1 = atoms1.get_potential_energy()
        
        # Get energy from second batcher - should be different since it's a different model
        atoms2 = atoms.copy()
        atoms2.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
        energy2 = atoms2.get_potential_energy()
        
        # Energies should differ between different models
        assert abs(energy1 - energy2) > 1e-5, f"Energies should differ between models but got {energy1} and {energy2}"
        
        # Shutdown first batcher (keep Ray running)
        batcher1.shutdown(shutdown_ray=False)
        
        # Verify second batcher still works
        atoms3 = atoms.copy()
        atoms3.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
        energy3 = atoms3.get_potential_energy()
        
        # Energy from second batcher should still match
        npt.assert_allclose(energy2, energy3, atol=1e-4)
        
        # Cleanup
        batcher2.shutdown(shutdown_ray=False)
        
    finally:
        cleanup_ray()


def test_multiple_batchers_same_model(uma_predict_unit):
    """Test that multiple InferenceBatchers with the same model can coexist."""
    try:
        setup_ray()
        
        # Create two batchers with the same model
        # Use fewer CPUs per batcher to allow both to run concurrently
        batcher1 = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
            ray_actor_options={"num_cpus": 4},  # Use fewer CPUs to allow concurrent batchers
        )
        
        batcher2 = InferenceBatcher(
            predict_unit=uma_predict_unit,
            max_batch_size=8,
            batch_wait_timeout_s=0.05,
            num_replicas=1,
            ray_actor_options={"num_cpus": 4},  # Use fewer CPUs to allow concurrent batchers
        )
        
        # Verify both have unique deployment names
        assert batcher1.deployment_name != batcher2.deployment_name, "Deployment names should be unique"
        
        # Test atoms
        atoms = bulk("Cu")
        
        # Get energy from first batcher
        atoms1 = atoms.copy()
        atoms1.calc = FAIRChemCalculator(batcher1.batch_predict_unit, task_name="omat")
        energy1 = atoms1.get_potential_energy()
        
        # Get energy from second batcher - should be the same since it's the same model
        atoms2 = atoms.copy()
        atoms2.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
        energy2 = atoms2.get_potential_energy()
        
        # Energies should match for the same model
        npt.assert_allclose(energy1, energy2, atol=1e-4)
        
        # Both batchers can work concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            def calc_energy(batcher, atoms):
                atoms_copy = atoms.copy()
                atoms_copy.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
                return atoms_copy.get_potential_energy()
            
            future1 = executor.submit(calc_energy, batcher1, atoms)
            future2 = executor.submit(calc_energy, batcher2, atoms)
            
            result1 = future1.result()
            result2 = future2.result()
            
            # Both should give consistent results
            npt.assert_allclose(result1, energy1, atol=1e-4)
            npt.assert_allclose(result2, energy1, atol=1e-4)
        
        # Cleanup - shutdown without shutting down Ray for the first one
        batcher1.shutdown(shutdown_ray=False)
        batcher2.shutdown(shutdown_ray=False)
        
    finally:
        cleanup_ray()

