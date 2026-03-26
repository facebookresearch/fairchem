"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the Ray Serve inference server using get_local_ray_cluster.

Two testing modes:
1. Ray remote tasks: Tests run as Ray remote tasks submitted to the cluster.
   These test the typical usage pattern where calculations are submitted as Ray tasks.
2. External client: Tests run from outside Ray, connecting to the inference server.
   These test the client-side code that connects to an existing service.
"""

from __future__ import annotations

import numpy.testing as npt
import pytest
import ray
import torch
from ase.build import bulk

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings

ATOL = 5e-4
DEPLOYMENT_NAME = "predict-server"


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
    """Set up a local Ray cluster with the FAIRChem inference server."""
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")

    from fairchem.core.launchers.cluster.ray_cluster_utils import get_local_ray_cluster

    with get_local_ray_cluster(
        num_cpus=8,
        num_gpus=1 if torch.cuda.is_available() else 0,
        start_inference_server=True,
        predict_unit=uma_predict_unit,
        log_level="WARNING",
        deployment_name=DEPLOYMENT_NAME,
    ) as head_file:
        yield head_file


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
        from ase import Atoms

        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

        atoms = Atoms.fromdict(atoms_dict)

        unit = BatchServerPredictUnit.from_deployment(deployment_name=dep_name)
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
        from ase import Atoms

        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

        atoms = Atoms.fromdict(atoms_dict)

        unit = BatchServerPredictUnit.from_deployment(deployment_name=dep_name)
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
        from ase import Atoms

        from fairchem.core.datasets.atomic_data import AtomicData
        from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

        atoms = Atoms.fromdict(atoms_dict)
        atomic_data = AtomicData.from_ase(atoms, task_name="omat")

        unit = BatchServerPredictUnit.from_deployment(deployment_name=dep_name)
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
    """Test BatchServerPredictUnit.from_deployment from outside Ray cluster."""
    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    unit = BatchServerPredictUnit.from_deployment(deployment_name=DEPLOYMENT_NAME)

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
    from fairchem.core.units.mlip_unit.batch import get_ray_connection_info
    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    conn_info = get_ray_connection_info(local_ray_cluster_with_inference)
    unit = BatchServerPredictUnit.from_deployment(
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
    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    unit = BatchServerPredictUnit.from_deployment(deployment_name=DEPLOYMENT_NAME)

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
    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    unit = BatchServerPredictUnit.from_deployment(deployment_name=DEPLOYMENT_NAME)

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
