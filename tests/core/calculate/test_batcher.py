"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy.testing as npt
import pytest
import ray
import torch
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate._batch import InferenceBatcher
from tests.conftest import get_predict_unit_for_test

# mark all tests in this module as serial (Ray needs serial execution due to
# large number of subprocesses) and pretrained (sweep-eligible).
pytestmark = [
    pytest.mark.serial,
    pytest.mark.gpu,
    pytest.mark.pretrained("uma-s-1p2"),
]


@pytest.fixture(autouse=True)
def setup_before_each_test():
    pass  # Override root conftest to prevent it from calling ray.shutdown() between tests


@pytest.fixture(scope="module")
def uma_predict_unit(pretrained_checkpoint):
    """Get a UMA predict unit for testing."""
    return get_predict_unit_for_test(pretrained_checkpoint)


def setup_ray():
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")
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
        logging_level="ERROR",
        _temp_dir="/tmp/ray",  # Use larger /tmp instead of default /var/tmp (512 MB tmpfs)
        _system_config={"local_fs_capacity_threshold": 0.99},
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


@pytest.fixture(scope="module")
def ray_session():
    """Initialize Ray once for the entire test module."""
    setup_ray()
    yield
    cleanup_ray()


@pytest.fixture()
def ray_controlled():
    """For tests that need to shut down and restart Ray mid-test.

    Gives the test exclusive Ray control, then restores the module-level
    cluster so subsequent tests that depend on ray_session are unaffected.
    """
    cleanup_ray()
    setup_ray()
    yield
    cleanup_ray()
    setup_ray()


@pytest.fixture()
def inference_batcher(ray_session, uma_predict_unit):
    batcher = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        concurrency_backend="threads",
        concurrency_backend_options={"max_workers": 4},
        ray_actor_options={"num_cpus": 2},
    )
    yield batcher
    batcher.shutdown(shutdown_ray=False)


@pytest.mark.parametrize(
    "kwargs,assert_fn",
    [
        pytest.param(
            {
                "concurrency_backend": "threads",
                "concurrency_backend_options": {"max_workers": 8},
                "ray_actor_options": {"num_cpus": 2},
            },
            lambda b: isinstance(b.executor, ThreadPoolExecutor),
            id="threads_concurrency",
        ),
        pytest.param(
            {"ray_actor_options": {"num_cpus": 2}},
            lambda b: b.predict_server_handle is not None,
            id="ray_actor_options",
        ),
    ],
)
def test_initialization_options(ray_session, uma_predict_unit, kwargs, assert_fn):
    batcher = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=16,
        batch_wait_timeout_s=0.1,
        num_replicas=1,
        **kwargs,
    )
    assert assert_fn(batcher)
    batcher.shutdown(shutdown_ray=False)


def test_context_manager_enter_exit(ray_session, uma_predict_unit):
    with InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=16,
        batch_wait_timeout_s=0.1,
        num_replicas=1,
        ray_actor_options={"num_cpus": 2},
    ) as batcher:
        assert hasattr(batcher, "executor")
        assert hasattr(batcher, "predict_server_handle")
        executor = batcher.executor

    assert executor is not None

    with pytest.raises(
        RuntimeError, match="cannot schedule new futures after shutdown"
    ):
        executor.submit(time.sleep, 1)


def test_batch_vs_serial_consistency(inference_batcher, uma_predict_unit):
    """Test that batched and serial calculations produce consistent results."""
    atoms_list = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]

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


def test_checkpoint_swap_with_energy_verification(
    ray_session, uma_predict_unit, uma_predict_unit_alt
):
    """Test that checkpoint swapping produces different energies and swapping back recovers originals."""
    batcher = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        concurrency_backend="threads",
        concurrency_backend_options={"max_workers": 4},
        ray_actor_options={"num_cpus": 4},
    )

    atoms_list = [bulk("Cu"), bulk("Al")]

    energies_initial = []
    for atoms in atoms_list:
        atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
        energies_initial.append(atoms.get_potential_energy())

    batcher.update_checkpoint(uma_predict_unit_alt)

    atoms_list = [bulk("Cu"), bulk("Al")]
    energies_swapped = []
    for atoms in atoms_list:
        atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
        energies_swapped.append(atoms.get_potential_energy())

    for e_initial, e_swapped in zip(energies_initial, energies_swapped):
        assert (
            abs(e_initial - e_swapped) > 1e-5
        ), f"Energies should differ between models but got {e_initial} and {e_swapped}"

    batcher.update_checkpoint(uma_predict_unit)

    energies_restored = []
    for atoms in atoms_list:
        atoms.calc = FAIRChemCalculator(batcher.batch_predict_unit, task_name="omat")
        energies_restored.append(atoms.get_potential_energy())

    npt.assert_allclose(energies_initial, energies_restored, atol=1e-4)
    batcher.shutdown(shutdown_ray=False)


def test_batcher_shutdown(ray_controlled, uma_predict_unit):
    """Test that shutdown(shutdown_ray=True) cleans up all resources including Ray."""
    batcher = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        ray_actor_options={"num_cpus": 2},
    )

    assert batcher.predict_server_handle is not None
    executor = batcher.executor

    batcher.shutdown(shutdown_ray=True)
    with pytest.raises(
        RuntimeError, match="cannot schedule new futures after shutdown"
    ):
        executor.submit(time.sleep, 1)

    assert batcher.predict_server_handle is None
    assert not ray.is_initialized()


@pytest.mark.parametrize(
    "same_model",
    [
        pytest.param(True, id="same_model"),
        pytest.param(False, id="different_models"),
    ],
)
def test_multiple_batchers(
    ray_session, uma_predict_unit, uma_predict_unit_alt, same_model
):
    """Test that multiple InferenceBatchers can coexist on the same Ray cluster."""
    predict_unit2 = uma_predict_unit if same_model else uma_predict_unit_alt

    batcher1 = InferenceBatcher(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        ray_actor_options={"num_cpus": 4, "num_gpus": 0.5},
    )
    batcher2 = InferenceBatcher(
        predict_unit=predict_unit2,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        ray_actor_options={"num_cpus": 4, "num_gpus": 0.5},
    )

    assert batcher1.deployment_name != batcher2.deployment_name

    atoms = bulk("Cu")

    atoms1 = atoms.copy()
    atoms1.calc = FAIRChemCalculator(batcher1.batch_predict_unit, task_name="omat")
    energy1 = atoms1.get_potential_energy()

    atoms2 = atoms.copy()
    atoms2.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
    energy2 = atoms2.get_potential_energy()

    if same_model:
        npt.assert_allclose(energy1, energy2, atol=1e-4)
    else:
        assert (
            abs(energy1 - energy2) > 1e-5
        ), f"Energies should differ between models but got {energy1} and {energy2}"

    # Verify both batchers work concurrently
    def calc_energy(batcher, atoms):
        atoms_copy = atoms.copy()
        atoms_copy.calc = FAIRChemCalculator(
            batcher.batch_predict_unit, task_name="omat"
        )
        return atoms_copy.get_potential_energy()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(calc_energy, batcher1, atoms)
        future2 = executor.submit(calc_energy, batcher2, atoms)
        result1 = future1.result()
        result2 = future2.result()

    npt.assert_allclose(result1, energy1, atol=1e-4)
    npt.assert_allclose(result2, energy2, atol=1e-4)

    # Verify batcher2 still works after batcher1 shuts down
    batcher1.shutdown(shutdown_ray=False)
    atoms3 = atoms.copy()
    atoms3.calc = FAIRChemCalculator(batcher2.batch_predict_unit, task_name="omat")
    npt.assert_allclose(energy2, atoms3.get_potential_energy(), atol=1e-4)

    batcher2.shutdown(shutdown_ray=False)
