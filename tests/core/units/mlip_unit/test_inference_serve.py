"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the Ray Serve inference server.

Two testing modes:
1. Ray remote tasks: Tests run as Ray remote tasks submitted to the cluster.
   These test the typical usage pattern where calculations are submitted as Ray tasks.
2. External client: Tests run from outside Ray, connecting to the inference server.
   These test the client-side code that connects to an existing service.
"""

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from pathlib import Path

import numpy.testing as npt
import pytest
import ray
import torch
from ase import Atoms
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.launchers.cluster.ray_cluster import find_free_port
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.batch_server import (
    get_ray_connection_info,
    setup_batch_predict_server,
    setup_multiplexed_batch_predict_server,
    wait_for_serve_ready,
)
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

ATOL = 5e-4
DEPLOYMENT_NAME = "predict-server"
MULTIPLEXED_DEPLOYMENT_NAME = "multiplexed-predict-server"
NAMESPACE = "fairchem_inference_test"


@pytest.fixture(scope="module")
def uma_predict_unit():
    """Module-scoped predict unit using the first available UMA model."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    if not uma_models:
        pytest.skip("No UMA models available")
    settings = InferenceSettings(
        merge_mole=False,
        external_graph_gen=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pretrained_mlip.get_predict_unit(
        uma_models[0], device=device, inference_settings=settings
    )


@pytest.fixture(scope="module")
def local_ray_cluster_with_inference(uma_predict_unit):
    """Start a local Ray instance with the FAIRChem inference server."""
    num_gpus = 1 if torch.cuda.is_available() else 0
    dashboard_port = find_free_port()

    ray.init(
        num_cpus=8,
        num_gpus=num_gpus if num_gpus > 0 else None,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_config=ray.LoggingConfig(log_level="WARNING"),
        dashboard_port=dashboard_port,
        namespace=NAMESPACE,
    )

    setup_batch_predict_server(uma_predict_unit, deployment_name=DEPLOYMENT_NAME)
    wait_for_serve_ready(app_name=DEPLOYMENT_NAME)

    cluster_id = str(uuid.uuid4())
    head_file_path = Path.home() / ".fairray" / cluster_id / "head.json"
    head_file_path.parent.mkdir(parents=True, exist_ok=True)
    head_file_path.write_text(
        json.dumps(
            {
                "hostname": "localhost",
                "dashboard_port": dashboard_port,
                "local": True,
                "num_cpus": 8,
                "num_gpus": num_gpus,
                "namespace_serve_fairchem": NAMESPACE,
            }
        )
    )

    yield str(head_file_path)

    with suppress(Exception):
        serve.shutdown()
    ray.shutdown()

    if head_file_path.exists():
        head_file_path.unlink()
        with suppress(OSError):
            head_file_path.parent.rmdir()


# ---------------------------------------------------------------------------
# Ray Remote Task Tests
# These tests submit work as Ray remote tasks to the cluster.
# This is the typical usage pattern for distributed inference.
# ---------------------------------------------------------------------------


@pytest.mark.gpu()
def test_rayserve_remote_task_single_system(
    local_ray_cluster_with_inference, uma_predict_unit
):
    """Test BatchServerPredictUnit via Ray remote task - single system."""

    @ray.remote
    def compute_energy_forces(dep_name: str, atoms_dict: dict):
        """Ray remote task that uses the inference server."""
        atoms = Atoms.fromdict(atoms_dict)

        unit = BatchServerPredictUnit.from_deployment_connection_info(
            deployment_name=dep_name
        )
        atoms.calc = FAIRChemCalculator(unit, task_name="omat")

        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(voigt=False),
        }

    atoms = bulk("Cu")
    atoms_dict = atoms.todict()

    result = ray.get(compute_energy_forces.remote(DEPLOYMENT_NAME, atoms_dict))

    # Compare with local prediction
    atoms_local = bulk("Cu")
    atoms_local.calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")
    energy_local = atoms_local.get_potential_energy()
    forces_local = atoms_local.get_forces()
    stress_local = atoms_local.get_stress(voigt=False)

    npt.assert_allclose(result["energy"], energy_local, atol=ATOL)
    npt.assert_allclose(result["forces"], forces_local, atol=ATOL)
    npt.assert_allclose(result["stress"], stress_local, atol=ATOL)


@pytest.mark.gpu()
def test_rayserve_remote_task_multiple_concurrent(local_ray_cluster_with_inference):
    """Test multiple concurrent Ray remote tasks hitting the inference server."""

    @ray.remote
    def compute_energy(dep_name: str, atoms_dict: dict):
        """Ray remote task that computes energy via inference server."""
        atoms = Atoms.fromdict(atoms_dict)

        unit = BatchServerPredictUnit.from_deployment_connection_info(
            deployment_name=dep_name
        )
        atoms.calc = FAIRChemCalculator(unit, task_name="omat")

        return atoms.get_potential_energy()

    systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atoms_dicts = [atoms.todict() for atoms in systems]

    futures = [compute_energy.remote(DEPLOYMENT_NAME, d) for d in atoms_dicts]
    results = ray.get(futures)

    assert len(results) == len(systems)
    for energy in results:
        assert isinstance(energy, float)
        assert energy == energy  # Check not NaN


@pytest.mark.gpu()
def test_rayserve_remote_task_batching(local_ray_cluster_with_inference):
    """Test that concurrent requests are batched by the inference server."""

    @ray.remote
    def compute_via_handle(dep_name: str, atoms_dict: dict):
        """Ray remote task using BatchServerPredictUnit directly."""
        atoms = Atoms.fromdict(atoms_dict)
        atomic_data = AtomicData.from_ase(atoms, task_name="omat")

        unit = BatchServerPredictUnit.from_deployment_connection_info(
            deployment_name=dep_name
        )
        return unit.predict(atomic_data, undo_element_references=True)

    systems = [bulk("Cu") for _ in range(10)]
    atoms_dicts = [atoms.todict() for atoms in systems]

    futures = [compute_via_handle.remote(DEPLOYMENT_NAME, d) for d in atoms_dicts]
    results = ray.get(futures)

    assert len(results) == len(systems)
    for result in results:
        assert "energy" in result
        assert "forces" in result


# ---------------------------------------------------------------------------
# External Client Tests
# These tests run from outside Ray, connecting to the inference server.
# This tests the client-side code for accessing an existing service.
# ---------------------------------------------------------------------------


@pytest.mark.gpu()
def test_rayserve_external_client_single(local_ray_cluster_with_inference):
    """Test BatchServerPredictUnit.from_deployment_connection_info from outside Ray cluster."""

    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=DEPLOYMENT_NAME
    )

    atoms = bulk("Cu")
    atomic_data = AtomicData.from_ase(atoms, task_name="omat")

    result = unit.predict(atomic_data, undo_element_references=True)

    assert "energy" in result, "Result missing 'energy' key"
    assert "forces" in result, "Result missing 'forces' key"
    assert "stress" in result, "Result missing 'stress' key"
    assert result["forces"].shape == (len(atoms), 3)


@pytest.mark.gpu()
def test_rayserve_external_multiple_systems(local_ray_cluster_with_inference):
    """Test BatchServerPredictUnit from outside Ray with multiple systems."""
    conn_info = get_ray_connection_info(local_ray_cluster_with_inference)
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=DEPLOYMENT_NAME,
        ray_address=conn_info["ray_address"],
        namespace=conn_info["namespace_serve_fairchem"],
    )

    systems = [
        bulk("Cu"),
        bulk("Al"),
        bulk("Fe"),
        bulk("Ni"),
        bulk("MgO", "rocksalt", a=4.213),
    ]

    for atoms in systems:
        atoms.calc = FAIRChemCalculator(unit, task_name="omat")
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)

        assert isinstance(energy, float), f"Energy should be float, got {type(energy)}"
        assert forces.shape == (
            len(atoms),
            3,
        ), f"Forces shape mismatch for {atoms.get_chemical_formula()}"
        assert stress.shape == (
            3,
            3,
        ), f"Stress shape mismatch for {atoms.get_chemical_formula()}"


@pytest.mark.gpu()
def test_rayserve_external_model_metadata(local_ray_cluster_with_inference):
    """Test that BatchServerPredictUnit correctly fetches model metadata."""

    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=DEPLOYMENT_NAME
    )

    dataset_to_tasks = unit.dataset_to_tasks

    assert dataset_to_tasks is not None, "dataset_to_tasks should not be None"
    assert len(dataset_to_tasks) > 0, "dataset_to_tasks should not be empty"
    assert (
        "omat" in dataset_to_tasks
    ), f"Expected 'omat' in tasks, got: {list(dataset_to_tasks.keys())}"


@pytest.mark.gpu()
def test_rayserve_external_vs_local_comparison(
    local_ray_cluster_with_inference, uma_predict_unit
):
    """Compare BatchServerPredictUnit predictions with local predict unit."""
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=DEPLOYMENT_NAME
    )

    # Test with the served calculator
    atoms_served = bulk("Cu")
    atoms_served.calc = FAIRChemCalculator(unit, task_name="omat")

    # Test with local predict unit for comparison
    atoms_local = bulk("Cu")
    atoms_local.calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")

    energy_served = atoms_served.get_potential_energy()
    forces_served = atoms_served.get_forces()
    stress_served = atoms_served.get_stress(voigt=False)

    energy_local = atoms_local.get_potential_energy()
    forces_local = atoms_local.get_forces()
    stress_local = atoms_local.get_stress(voigt=False)

    npt.assert_allclose(
        energy_served,
        energy_local,
        atol=ATOL,
        err_msg="Energy mismatch between BatchServerPredictUnit and local predict unit",
    )
    npt.assert_allclose(
        forces_served,
        forces_local,
        atol=ATOL,
        err_msg="Forces mismatch between BatchServerPredictUnit and local predict unit",
    )
    npt.assert_allclose(
        stress_served,
        stress_local,
        atol=ATOL,
        err_msg="Stress mismatch between BatchServerPredictUnit and local predict unit",
    )


# ---------------------------------------------------------------------------
# Multiplexed Server Tests
# These tests exercise the MultiplexedBatchPredictServer and
# BatchServerPredictUnit for on-demand model loading.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uma_multiplexed_model_id():
    """Return the multiplexed model ID for the first available UMA model."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    if not uma_models:
        pytest.skip("No UMA models available")
    return f"{uma_models[0]}:default"


@pytest.fixture(scope="module")
def local_multiplexed_cluster():
    """Set up a local Ray cluster with a multiplexed inference server."""

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=8,
            num_gpus=1 if torch.cuda.is_available() else 0,
        )

    setup_multiplexed_batch_predict_server(
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
        ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0},
    )
    wait_for_serve_ready(app_name=MULTIPLEXED_DEPLOYMENT_NAME)
    yield
    serve.delete(MULTIPLEXED_DEPLOYMENT_NAME)


@pytest.mark.gpu()
def test_multiplexed_single_model(
    local_multiplexed_cluster, uma_multiplexed_model_id, uma_predict_unit
):
    """Test loading a single model via the multiplexed server."""
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        multiplexed_model_id=uma_multiplexed_model_id,
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
    )

    atoms = bulk("Cu")
    atoms.calc = FAIRChemCalculator(unit, task_name="omat")
    energy_mux = atoms.get_potential_energy()
    forces_mux = atoms.get_forces()
    stress_mux = atoms.get_stress(voigt=False)

    # Compare with local prediction
    atoms_local = bulk("Cu")
    atoms_local.calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")
    energy_local = atoms_local.get_potential_energy()
    forces_local = atoms_local.get_forces()
    stress_local = atoms_local.get_stress(voigt=False)

    npt.assert_allclose(energy_mux, energy_local, atol=ATOL)
    npt.assert_allclose(forces_mux, forces_local, atol=ATOL)
    npt.assert_allclose(stress_mux, stress_local, atol=ATOL)


@pytest.mark.gpu()
def test_multiplexed_switch_models(local_multiplexed_cluster):
    """Test switching between two different model keys."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    if len(uma_models) < 2:
        pytest.skip("Need at least 2 UMA models to test switching")

    key_a = f"{uma_models[0]}:default"
    key_b = f"{uma_models[1]}:default"

    unit_a = BatchServerPredictUnit.from_deployment_connection_info(
        multiplexed_model_id=key_a,
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
    )
    unit_b = BatchServerPredictUnit.from_deployment_connection_info(
        multiplexed_model_id=key_b,
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
    )

    atoms = bulk("Cu")

    data = AtomicData.from_ase(atoms, task_name="omat")
    result_a = unit_a.predict(data)
    result_b = unit_b.predict(data)

    # Both should produce valid results (not necessarily equal)
    assert "energy" in result_a
    assert "energy" in result_b
    assert "forces" in result_a
    assert "forces" in result_b


@pytest.mark.gpu()
def test_multiplexed_concurrent_requests(
    local_multiplexed_cluster, uma_multiplexed_model_id
):
    """Test concurrent requests to the multiplexed server."""

    @ray.remote
    def compute_energy_mux(dep_name: str, multiplexed_model_id: str, atoms_dict: dict):
        """Ray remote task using BatchServerPredictUnit."""

        atoms = Atoms.fromdict(atoms_dict)
        unit = BatchServerPredictUnit.from_deployment_connection_info(
            multiplexed_model_id=multiplexed_model_id,
            deployment_name=dep_name,
        )
        atoms.calc = FAIRChemCalculator(unit, task_name="omat")
        return atoms.get_potential_energy()

    systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atoms_dicts = [a.todict() for a in systems]

    futures = [
        compute_energy_mux.remote(
            MULTIPLEXED_DEPLOYMENT_NAME, uma_multiplexed_model_id, d
        )
        for d in atoms_dicts
    ]
    results = ray.get(futures)

    assert len(results) == len(systems)
    for energy in results:
        assert isinstance(energy, float)
        assert energy == energy  # Check not NaN
