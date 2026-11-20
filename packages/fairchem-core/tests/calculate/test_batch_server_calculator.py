"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import numpy.testing as npt
import pytest
import ray
import torch
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator
from fairchem.core.components.batch_server import setup_batch_predict_server
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit
from tests.conftest import get_predict_unit_for_test

ATOL = 5e-4


@pytest.fixture(scope="module")
def served_predict_unit(pretrained_checkpoint):
    """
    Load the predict unit served by the batch server.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return get_predict_unit_for_test(pretrained_checkpoint, device=device)


@pytest.fixture()
def batch_server_handle(served_predict_unit):
    """
    Set up a batch server for testing.
    """
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")

    if ray.is_initialized():
        with suppress(Exception):
            serve.shutdown()
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=10,
        num_gpus=1 if torch.cuda.is_available() else 0,
        logging_level="ERROR",
    )

    server_handle = setup_batch_predict_server(
        predict_unit=served_predict_unit,
        deployment_config={
            "num_replicas": 1,
            "ray_actor_options": {
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "num_cpus": 2,
            },
        },
        batch_config={
            "max_batch_size": 8,
            "batch_wait_timeout_s": 0.05,
        },
    )

    yield server_handle

    with suppress(Exception):
        serve.shutdown()
    ray.shutdown()


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_batch_server_predict_unit_with_calculator(
    batch_server_handle, served_predict_unit
):
    """
    Test BatchServerPredictUnit with FAIRChemCalculator.
    """
    batch_predict_unit = BatchServerPredictUnit(server_handle=batch_server_handle)

    atoms = bulk("Cu")
    atoms.calc = FAIRChemCalculator(batch_predict_unit, task_name="omat")

    reference_atoms = bulk("Cu")
    reference_atoms.calc = FAIRChemCalculator(served_predict_unit, task_name="omat")

    npt.assert_allclose(
        atoms.get_potential_energy(),
        reference_atoms.get_potential_energy(),
        atol=ATOL,
    )
    npt.assert_allclose(atoms.get_forces(), reference_atoms.get_forces(), atol=ATOL)
    npt.assert_allclose(
        atoms.get_stress(voigt=False),
        reference_atoms.get_stress(voigt=False),
        atol=ATOL,
    )


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_batch_server_predict_unit_multiple_systems(batch_server_handle):
    """
    Test BatchServerPredictUnit with multiple concurrent requests.
    """
    batch_predict_unit = BatchServerPredictUnit(server_handle=batch_server_handle)
    atoms_list = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(batch_predict_unit.predict, data)
            for data in atomic_data_list
        ]
        results = [future.result() for future in futures]

    assert len(results) == len(atoms_list)
    for atoms, predictions in zip(atoms_list, results):
        assert "energy" in predictions
        assert "forces" in predictions
        assert "stress" in predictions
        assert predictions["energy"].shape == (1,)
        assert predictions["forces"].shape == (len(atoms), 3)
