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


def setup_ray():
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")

    if ray.is_initialized():
        with contextlib.suppress(Exception):
            serve.shutdown()
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=1 if torch.cuda.is_available() else 0,
        logging_level="ERROR",  # Reduce noise in test output
    )


def cleanup_ray():
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
