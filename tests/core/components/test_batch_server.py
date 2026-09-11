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
import logging
import os
import pickle
import uuid
from collections import OrderedDict, defaultdict
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy.testing as npt
import pytest
import ray
import torch
from ase import Atoms
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator
from fairchem.core.components import batch_server
from fairchem.core.components.batch_server import (
    MODEL_SPEC_CACHE_CAPACITY,
    BatchConfig,
    BatchPredictServer,
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
    assert ModelSpec("x").model_id == "x-7aea04de3f8a"


def test_model_spec_auto_source_does_not_alias_registry_model():
    """``source="auto"`` must resolve, not mint a second id for one model."""
    auto = ModelSpec("uma-s-1p1")

    assert auto.source == "registry"
    assert auto.model_id == ModelSpec("uma-s-1p1", source="registry").model_id


def test_model_spec_auto_source_does_not_alias_checkpoint_file(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.touch()
    auto = ModelSpec(str(checkpoint))

    assert auto.source == "path"
    assert auto.model_id == ModelSpec(str(checkpoint), source="path").model_id


def test_model_spec_relative_and_absolute_paths_share_identity(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.touch()
    monkeypatch.chdir(tmp_path)

    relative = ModelSpec("./ckpt.pt")

    assert relative.checkpoint == os.path.realpath(str(checkpoint))
    assert relative.model_id == ModelSpec(str(checkpoint)).model_id


def test_model_spec_path_and_registry_remain_distinct():
    """Distinct loaders for one name are genuinely different models."""
    assert (
        ModelSpec("x", source="path").model_id
        != ModelSpec("x", source="registry").model_id
    )


def test_model_spec_is_hashable_despite_unhashable_fields():
    """
    ``frozen=True`` advertises hashability, but the generated ``__hash__``
    would hash a tuple containing an unhashable dict and ``InferenceSettings``.
    """
    spec = ModelSpec(
        "x",
        InferenceSettings(predict_untrained_stress={"omat"}),
        overrides={"backbone": {"max_neighbors": 64}},
    )

    assert hash(spec) == hash(spec.model_id)


def test_model_spec_equal_specs_hash_equal():
    overrides = {"backbone": {"max_neighbors": 64}}
    first = ModelSpec("x", overrides=overrides)
    second = ModelSpec("x", overrides=dict(overrides))

    assert first == second
    assert hash(first) == hash(second)


def test_model_spec_usable_as_set_member_and_dict_key():
    first = ModelSpec("x")
    duplicate = ModelSpec("x")
    other = ModelSpec("y")

    assert len({first, duplicate, other}) == 2
    assert {first: "a", other: "b"}[duplicate] == "a"
    # Aliased specs collapse to one entry (see source resolution above).
    assert len({ModelSpec("uma-s-1p1"), ModelSpec("uma-s-1p1", source="registry")}) == 1


def test_model_spec_hash_survives_pickling():
    """Specs are pickled to the replica with every request."""
    spec = ModelSpec("x", overrides={"backbone": {"max_neighbors": 64}})
    restored = pickle.loads(pickle.dumps(spec))

    assert restored == spec
    assert hash(restored) == hash(spec)
    assert restored.model_id == spec.model_id


def test_model_spec_normalizes_and_validates_device():
    assert ModelSpec("x", device="cpu").device == "cpu"
    assert ModelSpec("x").device is None

    with pytest.raises(ValueError, match="valid torch device"):
        ModelSpec("x", device="nonsense")


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


def _bare_server(server_class):
    """Build an un-initialized deployment instance for pure-logic unit tests."""
    return object.__new__(server_class.func_or_class)


class _RecordingBatchQueue:
    """Stands in for the ``@serve.batch`` wrapper on ``predict``."""

    def __init__(self):
        self.max_batch_size = None
        self.batch_wait_timeout_s = None

    def set_max_batch_size(self, value):
        self.max_batch_size = value

    def set_batch_wait_timeout_s(self, value):
        self.batch_wait_timeout_s = value


def _reconfigurable_server(server_class):
    server = _bare_server(server_class)
    server.predict = _RecordingBatchQueue()
    server.split_oom_batch = False
    return server


def test_batch_config_to_user_config_is_json_serializable():
    config = BatchConfig(max_batch_size=4096, batch_wait_timeout_s=0.02)
    user_config = config.to_user_config()

    assert user_config == {
        "max_batch_size": 4096,
        "batch_wait_timeout_s": 0.02,
        "split_oom_batch": False,
    }
    # Ray Serve ships user_config through the controller, so it must round-trip.
    assert json.loads(json.dumps(user_config)) == user_config


@pytest.mark.parametrize(
    "server_class", [BatchPredictServer, MultiplexedBatchPredictServer]
)
def test_server_exposes_reconfigure(server_class):
    """
    Ray Serve raises ``RayServeException`` at replica startup if ``user_config``
    is set on a deployment whose class has no ``reconfigure`` method.
    """
    assert hasattr(server_class.func_or_class, "reconfigure")


@pytest.mark.parametrize(
    "server_class", [BatchPredictServer, MultiplexedBatchPredictServer]
)
def test_reconfigure_applies_batch_config(server_class):
    server = _reconfigurable_server(server_class)

    server.reconfigure(
        BatchConfig(
            max_batch_size=4096, batch_wait_timeout_s=0.02, split_oom_batch=True
        ).to_user_config()
    )

    assert server.predict.max_batch_size == 4096
    assert server.predict.batch_wait_timeout_s == 0.02
    assert server.split_oom_batch is True


def test_reconfigure_ignores_empty_payload():
    server = _reconfigurable_server(BatchPredictServer)

    server.reconfigure({})
    server.reconfigure(None)

    assert server.predict.max_batch_size is None
    assert server.predict.batch_wait_timeout_s is None


def test_reconfigure_leaves_absent_keys_untouched():
    server = _reconfigurable_server(BatchPredictServer)

    server.reconfigure({"max_batch_size": 128})

    assert server.predict.max_batch_size == 128
    assert server.predict.batch_wait_timeout_s is None
    assert server.split_oom_batch is False


def test_deploy_app_pushes_batch_config_through_user_config(monkeypatch):
    """Batching must travel as user_config so Serve applies it to all replicas."""
    recorded = {}

    class FakeDeployment:
        def options(self, **kwargs):
            recorded["options"] = kwargs
            return self

        def bind(self, *args, **kwargs):
            recorded["bind"] = (args, kwargs)
            return "app"

    monkeypatch.setattr(batch_server.serve, "run", lambda *a, **k: "handle")
    monkeypatch.setattr(batch_server, "_DEPLOYED_APPS", {})

    config = BatchConfig(max_batch_size=256, batch_wait_timeout_s=0.5)
    handle = batch_server._deploy_app(
        deployment=FakeDeployment(),
        options_kwargs={"num_replicas": 3, "user_config": {"unrelated": 1}},
        bind_args=("ref",),
        bind_kwargs=config.to_init_kwargs(),
        batch_config=config,
        deployment_name="app-under-test",
        route_prefix="/x",
    )

    assert handle == "handle"
    assert recorded["options"]["num_replicas"] == 3
    # Caller-supplied user_config is preserved; batch keys are layered on top.
    assert recorded["options"]["user_config"] == {
        "unrelated": 1,
        "max_batch_size": 256,
        "batch_wait_timeout_s": 0.5,
        "split_oom_batch": False,
    }
    assert "app-under-test" in batch_server._DEPLOYED_APPS


def test_update_batch_config_redeploys_with_only_user_config_changed(monkeypatch):
    """
    Holding every other option identical is what makes Ray Serve classify the
    redeploy as a lightweight update (reconfigure in place, no model reload).
    """
    deploys = []

    class FakeDeployment:
        def options(self, **kwargs):
            deploys.append(kwargs)
            return self

        def bind(self, *args, **kwargs):
            return "app"

    monkeypatch.setattr(batch_server.serve, "run", lambda *a, **k: "handle")
    monkeypatch.setattr(batch_server, "_DEPLOYED_APPS", {})

    original = BatchConfig(max_batch_size=256, batch_wait_timeout_s=0.5)
    batch_server._deploy_app(
        deployment=FakeDeployment(),
        options_kwargs={"num_replicas": 3, "ray_actor_options": {"num_gpus": 1}},
        bind_args=("ref",),
        bind_kwargs=original.to_init_kwargs(),
        batch_config=original,
        deployment_name="app-under-test",
        route_prefix="/x",
    )

    batch_server.update_batch_config(
        "app-under-test", BatchConfig(max_batch_size=4096, batch_wait_timeout_s=0.02)
    )

    assert len(deploys) == 2
    first, second = deploys
    assert {k: v for k, v in first.items() if k != "user_config"} == {
        k: v for k, v in second.items() if k != "user_config"
    }
    assert second["user_config"]["max_batch_size"] == 4096
    assert second["user_config"]["batch_wait_timeout_s"] == 0.02


def test_prepare_deployment_config_honors_explicit_num_gpus():
    dc = batch_server._prepare_deployment_config(None, 2, "explicit")

    assert dc.ray_actor_options["num_gpus"] == 2


def test_prepare_deployment_config_does_not_override_pinned_num_gpus():
    """An explicit ray_actor_options value always wins over the default."""
    dc = batch_server._prepare_deployment_config(
        {"ray_actor_options": {"num_gpus": 0}}, 1, "inferred"
    )

    assert dc.ray_actor_options["num_gpus"] == 0


@pytest.mark.parametrize(
    ("device", "num_gpus"),
    [("cpu", 0), ("cpu", 1), ("cuda:0", 1), ("cuda:1", 0)],
    ids=["cpu-no-gpu", "cpu-with-gpu", "cuda0-fits", "cuda1-unmapped"],
)
def test_check_predict_unit_device_accepts_valid_placements(device, num_gpus):
    unit = SimpleNamespace(device=device)

    batch_server._check_predict_unit_device(unit, num_gpus)


def test_check_predict_unit_device_rejects_out_of_range_ordinal():
    """
    Ray remaps CUDA_VISIBLE_DEVICES per replica, so a unit pinned to cuda:1
    cannot deserialize in a replica granted a single GPU.
    """
    unit = SimpleNamespace(device="cuda:1")

    with pytest.raises(ValueError, match="invalid device ordinal"):
        batch_server._check_predict_unit_device(unit, 1)


def test_resolve_device_returns_pinned_cpu_device():
    assert ModelSpec("x", device="cpu").resolve_device() == "cpu"


def test_resolve_device_raises_when_pinned_cuda_is_unavailable(monkeypatch):
    """Silently running a GPU workload on CPU is worse than failing fast."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no CUDA device"):
        ModelSpec("x", device="cuda").resolve_device()


def test_resolve_device_warns_when_unpinned_spec_falls_back_to_cpu(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level(logging.WARNING):
        assert ModelSpec("x").resolve_device() == "cpu"

    assert "will load on CPU" in caplog.text


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert ModelSpec("x").resolve_device() == "cuda"


def test_update_batch_config_rejects_unknown_deployment(monkeypatch):
    monkeypatch.setattr(batch_server, "_DEPLOYED_APPS", {})

    with pytest.raises(KeyError, match="No deployment named 'never-deployed'"):
        batch_server.update_batch_config("never-deployed", BatchConfig())


@pytest.mark.parametrize(
    ("server_class", "batched", "num_requests", "expected"),
    [
        (BatchPredictServer, True, 3, [True, False, True]),
        (MultiplexedBatchPredictServer, True, 3, [True, False, True]),
        (BatchPredictServer, False, 3, [True, True, True]),
        (MultiplexedBatchPredictServer, False, 3, [True, True, True]),
    ],
    ids=["single-batched", "mux-batched", "single-scalar", "mux-scalar"],
)
def test_normalize_undo_flags(server_class, batched, num_requests, expected):
    """``@serve.batch`` turns the scalar arg into one value per request."""
    server = _bare_server(server_class)
    value = [True, False, True] if batched else True

    assert server._normalize_undo_flags(value, num_requests) == expected


def test_normalize_undo_flags_rejects_length_mismatch():
    server = _bare_server(BatchPredictServer)

    with pytest.raises(ValueError, match="length 2 but the batch contains 3"):
        server._normalize_undo_flags([True, False], 3)


def test_run_grouped_inference_dispatches_one_sub_batch_per_flag():
    """
    Requests disagreeing on ``undo_element_references`` must not share a
    forward pass, and results must come back in the original request order.
    """
    server = _bare_server(BatchPredictServer)
    calls = []

    def fake_run_batched_inference(items, predict_unit, undo_element_references):
        # Regression guard: the flag must be a real bool, never the list that
        # ``@serve.batch`` hands to the batch function.
        assert isinstance(undo_element_references, bool)
        calls.append((list(items), undo_element_references))
        return [{"item": item, "undo": undo_element_references} for item in items]

    server._run_batched_inference = fake_run_batched_inference

    data = ["a", "b", "c", "d"]
    results = server._run_grouped_inference(data, [True, False, True, False], "unit")

    assert results == [
        {"item": "a", "undo": True},
        {"item": "b", "undo": False},
        {"item": "c", "undo": True},
        {"item": "d", "undo": False},
    ]
    assert sorted(calls, key=lambda call: call[1]) == [
        (["b", "d"], False),
        (["a", "c"], True),
    ]


def test_run_grouped_inference_uses_single_sub_batch_when_flags_agree():
    server = _bare_server(BatchPredictServer)
    calls = []

    def fake_run_batched_inference(items, predict_unit, undo_element_references):
        calls.append((list(items), undo_element_references))
        return [{"item": item} for item in items]

    server._run_batched_inference = fake_run_batched_inference

    results = server._run_grouped_inference(["a", "b"], [False, False], "unit")

    assert results == [{"item": "a"}, {"item": "b"}]
    assert calls == [(["a", "b"], False)]


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
