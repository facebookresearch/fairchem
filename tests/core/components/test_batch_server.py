"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Ray Serve inference server in three modes:
        1. Ray remote tasks (typical usage — submit as Ray tasks).
        2. External client (run from outside Ray, connect to the
           live deployment).
        3. Multiplexed server (on-demand model loading via
           MultiplexedBatchPredictServer).
Models: uma-s-1p1, uma-s-1p2 (integration-test markers). Locked to
        UMA-S only because the base GPU runner OOMs with uma-m-1p1's
        Ray Serve replicas. Pure ModelSpec tests remain CPU-safe.
CI:     test_gpu_sweep (models shard).
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict, defaultdict
from contextlib import suppress
from pathlib import Path

import numpy.testing as npt
import pytest
import ray
import torch
from ase import Atoms
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator
from fairchem.core.components.batch_server import (
    MODEL_SPEC_CACHE_CAPACITY,
    BatchConfig,
    ModelSpec,
    MultiplexedBatchPredictServer,
    get_ray_connection_info,
    setup_batch_predict_server,
    setup_multiplexed_batch_predict_server,
    wait_for_serve_ready,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.launchers.cluster.ray_cluster import find_free_port
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit
from tests.conftest import sweep_model, uma_models

ATOL = 5e-4
DEPLOYMENT_NAME = "predict-server"
MULTIPLEXED_DEPLOYMENT_NAME = "multiplexed-predict-server"
NAMESPACE = "fairchem_inference_test"


def test_batch_config_rejects_max_concurrent_batches():
    with pytest.raises(TypeError, match="max_concurrent_batches"):
        BatchConfig(max_concurrent_batches=2)


def test_model_spec_default_preset_has_canonical_identity():
    implicit = ModelSpec("x")
    explicit = ModelSpec("x", "default")

    assert implicit.inference_settings == explicit.inference_settings
    assert implicit.model_id == explicit.model_id


def test_model_spec_preserves_colon_bearing_checkpoint():
    spec = ModelSpec("s3://bucket/uma.pt", source="path")

    assert spec.checkpoint == "s3://bucket/uma.pt"
    assert spec.canonical_dict()["checkpoint"] == "s3://bucket/uma.pt"
    assert spec.model_id.startswith("uma.pt-")


def test_model_spec_identity_covers_settings_and_canonicalizes_sets():
    stress_a = InferenceSettings(
        execution_mode="general",
        predict_untrained_stress={"omat", "omol"},
    )
    stress_b = InferenceSettings(
        execution_mode="general",
        predict_untrained_stress={"omol", "omat"},
    )
    fast = InferenceSettings(
        execution_mode="umas_fast_pytorch",
        predict_untrained_stress={"omat", "omol"},
    )

    assert ModelSpec("x", stress_a).model_id == ModelSpec("x", stress_b).model_id
    assert ModelSpec("x", stress_a).model_id != ModelSpec("x", fast).model_id
    assert (
        ModelSpec("x", stress_a).model_id
        != ModelSpec("x", InferenceSettings(execution_mode="general")).model_id
    )
    assert (
        ModelSpec("x", stress_a, device="cpu").model_id
        != ModelSpec("x", stress_a, device="cuda").model_id
    )
    assert (
        ModelSpec("x", stress_a, overrides={"backbone": {"max_neighbors": 64}}).model_id
        != ModelSpec(
            "x", stress_a, overrides={"backbone": {"max_neighbors": 128}}
        ).model_id
    )


def test_model_spec_rejects_unknown_preset_at_construction():
    with pytest.raises(AssertionError, match="inference setting name"):
        ModelSpec(checkpoint="x", inference_settings="nonsense")


def test_model_spec_model_id_is_stable():
    assert ModelSpec("x").model_id == "x-050e09e7dbf7"


def test_model_spec_identity_is_immutable_after_construction():
    settings = InferenceSettings(predict_untrained_stress={"omat"})
    overrides = {"backbone": {"max_neighbors": 64}}
    spec = ModelSpec("x", settings, overrides=overrides)
    original_id = spec.model_id

    settings.predict_untrained_stress.add("omol")
    overrides["backbone"]["max_neighbors"] = 128
    spec.inference_settings.predict_untrained_stress.add("oc20")
    spec.overrides["backbone"]["max_neighbors"] = 256

    assert spec.model_id == original_id
    assert spec.canonical_dict()["inference_settings"]["predict_untrained_stress"] == [
        "omat"
    ]
    assert spec.canonical_dict()["overrides"] == {"backbone": {"max_neighbors": 64}}
    assert spec.loader_settings().predict_untrained_stress == {"omat"}
    assert spec.loader_overrides() == {"backbone": {"max_neighbors": 64}}


def test_batch_server_predict_unit_binds_and_sends_model_spec():
    class FakeResponse:
        def __init__(self, value):
            self.value = value

        def result(self, timeout_s=None):
            return self.value

    class FakeRemoteMethod:
        def __init__(self, value):
            self.value = value

        def remote(self, *args, **kwargs):
            return FakeResponse(self.value)

    class FakeHandle:
        def __init__(self):
            self.is_multiplexed = FakeRemoteMethod(True)
            self.bound_model_id = None
            self.calls = []

        def options(self, *, multiplexed_model_id):
            self.bound_model_id = multiplexed_model_id
            return self

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return FakeResponse({"energy": torch.tensor([1.0])})

    handle = FakeHandle()
    spec = ModelSpec("x", InferenceSettings(execution_mode="general"))
    unit = BatchServerPredictUnit(handle, model_spec=spec)
    data = object()

    result = unit.predict(data, undo_element_references=False)

    assert handle.bound_model_id == spec.model_id
    assert unit.model_spec is spec
    assert unit.multiplexed_model_id == spec.model_id
    assert handle.calls == [
        (
            (data,),
            {"spec": spec, "undo_element_references": False},
        )
    ]
    assert result["energy"].item() == 1.0


def test_model_spec_registry_evicts_least_recently_used_spec():
    server_class = MultiplexedBatchPredictServer.func_or_class
    server = object.__new__(server_class)
    server._specs = OrderedDict()
    server._active_spec_counts = defaultdict(int)
    server._spec_capacity = MODEL_SPEC_CACHE_CAPACITY
    specs = [
        ModelSpec(f"model-{index}") for index in range(MODEL_SPEC_CACHE_CAPACITY + 1)
    ]

    for spec in specs[:MODEL_SPEC_CACHE_CAPACITY]:
        server._register_spec(spec)
    server._register_spec(specs[0])
    server._register_spec(specs[-1])

    assert len(server._specs) == MODEL_SPEC_CACHE_CAPACITY
    assert specs[0].model_id in server._specs
    assert specs[1].model_id not in server._specs
    assert specs[-1].model_id in server._specs


def test_model_spec_registry_does_not_evict_in_flight_specs():
    server_class = MultiplexedBatchPredictServer.func_or_class
    server = object.__new__(server_class)
    server._specs = OrderedDict()
    server._active_spec_counts = defaultdict(int)
    server._spec_capacity = 2
    specs = [ModelSpec(f"model-{index}") for index in range(3)]

    for spec in specs:
        server._register_spec(spec, pin=True)

    assert list(server._specs) == [spec.model_id for spec in specs]

    server._release_spec(specs[0].model_id)

    assert list(server._specs) == [spec.model_id for spec in specs[1:]]
    assert len(server._specs) == server._spec_capacity


@pytest.fixture()
def dashboard_port():
    return find_free_port()


@pytest.fixture()
def local_ray_cluster_with_inference(uma_predict_unit, dashboard_port):
    """Start a local Ray instance with the FAIRChem inference server.

    Function-scoped: a fresh Ray cluster and Serve deployment is created
    for every test and fully torn down afterwards. This avoids GPU/actor
    contention between tests (important on single-GPU CI runners) at the
    cost of one Ray init per test.
    """
    num_gpus = 1 if torch.cuda.is_available() else 0

    ray.init(
        num_cpus=8,
        num_gpus=num_gpus if num_gpus > 0 else None,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_config=ray.LoggingConfig(log_level="WARNING"),
        dashboard_port=dashboard_port,
        namespace=NAMESPACE,
    )

    setup_batch_predict_server(
        uma_predict_unit,
        deployment_name=DEPLOYMENT_NAME,
        deployment_config={
            "ray_actor_options": {
                "num_cpus": 1,
                "num_gpus": 1 if num_gpus > 0 else 0,
            },
        },
    )
    wait_for_serve_ready(app_name=DEPLOYMENT_NAME)

    yield

    # Cached handles point at the now-dead deployment; clear so the next
    # test's fresh deployment isn't shadowed by a stale handle.
    BatchServerPredictUnit._handle_cache.clear()

    with suppress(Exception):
        serve.shutdown()
    ray.shutdown()


@pytest.fixture()
def local_ray_cluster_with_head_file(local_ray_cluster_with_inference, dashboard_port):
    """Extend local_ray_cluster_with_inference with a head.json for external client tests.

    Only tests that call get_ray_connection_info need this fixture.
    """
    num_gpus = 1 if torch.cuda.is_available() else 0
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
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_rayserve_remote_task_multiple_concurrent(local_ray_cluster_with_inference):
    """Test multiple concurrent Ray remote tasks hitting the inference server."""

    @ray.remote
    def compute_predictions(dep_name: str, atoms_dict: dict):
        """Ray remote task that computes predictions via inference server."""
        atoms = Atoms.fromdict(atoms_dict)
        atomic_data = AtomicData.from_ase(atoms, task_name="omat")

        unit = BatchServerPredictUnit.from_deployment_connection_info(
            deployment_name=dep_name
        )
        return unit.predict(atomic_data, undo_element_references=True)

    systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atoms_dicts = [atoms.todict() for atoms in systems]

    futures = [compute_predictions.remote(DEPLOYMENT_NAME, d) for d in atoms_dicts]
    results = ray.get(futures)

    assert len(results) == len(systems)
    for result, atoms in zip(results, systems):
        assert "energy" in result
        assert "forces" in result
        assert torch.isfinite(result["energy"]).all()
        assert result["forces"].shape == (len(atoms), 3)


# ---------------------------------------------------------------------------
# External Client Tests
# These tests run from outside Ray, connecting to the inference server.
# This tests the client-side code for accessing an existing service.
# ---------------------------------------------------------------------------


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_rayserve_external_multiple_systems(local_ray_cluster_with_head_file):
    """Test BatchServerPredictUnit from outside Ray with multiple systems."""
    conn_info = get_ray_connection_info(local_ray_cluster_with_head_file)
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
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
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
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
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


@pytest.fixture()
def uma_model_spec(request):
    """
    Model spec for the sweep model, or first available UMA model.

    Honors ``--sweep-model`` so per-model sweep CI jobs target the
    requested checkpoint. Skips when the sweep value is a filesystem
    path because this fixture exercises registry-backed model loading.
    """
    available_uma = uma_models()
    if not available_uma:
        pytest.skip("No UMA models available")
    sweep = sweep_model(request.config)
    if sweep:
        if sweep not in available_uma:
            pytest.skip(
                f"--sweep-model={sweep!r} is not a registered UMA model; "
                "multiplexed server tests need a registered name."
            )
        model = sweep
    else:
        model = available_uma[0]
    return ModelSpec(model, source="registry")


@pytest.fixture()
def local_multiplexed_cluster():
    """Set up a local Ray cluster with a multiplexed inference server.

    Function-scoped: full Ray + Serve teardown after each test so the GPU
    and actor resources are returned to the pool before the next test
    runs.
    """
    num_gpus = 1 if torch.cuda.is_available() else 0

    ray.init(
        log_to_driver=False,
        logging_config=ray.LoggingConfig(log_level="WARNING"),
        num_cpus=8,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
    )

    setup_multiplexed_batch_predict_server(
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
        deployment_config={
            "ray_actor_options": {
                "num_cpus": 1,
                "num_gpus": num_gpus,
            },
        },
    )
    wait_for_serve_ready(app_name=MULTIPLEXED_DEPLOYMENT_NAME)

    yield

    BatchServerPredictUnit._handle_cache.clear()
    with suppress(Exception):
        serve.shutdown()
    ray.shutdown()


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_multiplexed_single_model(
    local_multiplexed_cluster, uma_model_spec, uma_predict_unit
):
    """Test loading a single model via the multiplexed server."""
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        model_spec=uma_model_spec,
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
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_multiplexed_switch_models(local_multiplexed_cluster, uma_model_spec):
    """Test switching between two different model keys."""
    available_uma = uma_models()
    if len(available_uma) < 2:
        pytest.skip("Need at least 2 UMA models to test switching")

    # uma_model_spec already identifies the sweep target (or first UMA model).
    # Pick any other UMA model as the second spec.
    primary = uma_model_spec.checkpoint
    other_candidates = [m for m in available_uma if m != primary]
    if not other_candidates:
        pytest.skip("No second UMA model available that differs from the primary")

    spec_a = uma_model_spec
    spec_b = ModelSpec(other_candidates[0])

    unit_a = BatchServerPredictUnit.from_deployment_connection_info(
        model_spec=spec_a,
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
    )
    unit_b = BatchServerPredictUnit.from_deployment_connection_info(
        model_spec=spec_b,
        deployment_name=MULTIPLEXED_DEPLOYMENT_NAME,
    )

    data = AtomicData.from_ase(bulk("Cu"), task_name="omat")
    result_a = unit_a.predict(data)
    result_b = unit_b.predict(data)

    assert "energy" in result_a
    assert "forces" in result_a
    assert "energy" in result_b
    assert "forces" in result_b
    assert not torch.allclose(
        result_a["energy"], result_b["energy"]
    ), "Different models should produce different energies"


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_multiplexed_concurrent_requests(local_multiplexed_cluster, uma_model_spec):
    """Test concurrent requests to the multiplexed server."""

    @ray.remote
    def compute_predictions_mux(dep_name: str, model_spec: ModelSpec, atoms_dict: dict):
        """Ray remote task using BatchServerPredictUnit directly."""
        atoms = Atoms.fromdict(atoms_dict)
        atomic_data = AtomicData.from_ase(atoms, task_name="omat")
        unit = BatchServerPredictUnit.from_deployment_connection_info(
            model_spec=model_spec,
            deployment_name=dep_name,
        )
        return unit.predict(atomic_data, undo_element_references=True)

    systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atoms_dicts = [a.todict() for a in systems]

    futures = [
        compute_predictions_mux.remote(MULTIPLEXED_DEPLOYMENT_NAME, uma_model_spec, d)
        for d in atoms_dicts
    ]
    results = ray.get(futures)

    assert len(results) == len(systems)
    for result, atoms in zip(results, systems):
        assert "energy" in result
        assert "forces" in result
        assert torch.isfinite(result["energy"]).all()
        assert result["forces"].shape == (len(atoms), 3)
