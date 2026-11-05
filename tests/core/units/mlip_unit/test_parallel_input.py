"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase import Atoms

from fairchem.core import pretrained_mlip
from fairchem.core.common import distutils
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit
from fairchem.core.units.mlip_unit import predict as predict_module
from fairchem.core.units.mlip_unit.predict import (
    ParallelMLIPPredictUnit,
    _only_geometry_changed,
)
from tests.core.models.uma.uma_nitro.nitro_test_utils import random_checkpoint


def make_batch(atoms: Atoms) -> AtomicData:
    """
    Build a molecular request without external graph generation.
    """
    atoms = atoms.copy()
    atoms.info.update(charge=0, spin=1)
    return atomicdata_list_to_batch(
        [
            AtomicData.from_ase(
                atoms,
                task_name="omol",
                radius=2.0,
                r_edges=False,
                r_data_keys=["charge", "spin"],
            )
        ]
    )


@pytest.fixture()
def water_batch():
    return make_batch(
        Atoms(
            "OH2",
            positions=[[0.4, 0.4, 0.4], [1.34, 0.5, 0.4], [0.28, 1.23, 0.63]],
            cell=[2.1, 2.3, 2.5],
        )
    )


@pytest.fixture()
def tiny_parallel_checkpoint(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        random_checkpoint(
            num_layers=1,
            sphere_channels=8,
            hidden_channels=8,
            edge_channels=8,
            num_experts=2,
            num_distance_basis=8,
            cutoff=2.0,
        ),
        checkpoint,
    )
    return str(checkpoint)


def settings(merge_mole: bool) -> InferenceSettings:
    return InferenceSettings(
        merge_mole=merge_mole,
        compile=False,
        activation_checkpointing=False,
        external_graph_gen=False,
        auto_add_default_untrained_tasks=False,
        execution_mode="general",
    )


@pytest.mark.parametrize("assume_fixed_input", [False, True])
def test_pretrained_factory_forwards_input_mode(
    tmp_path, monkeypatch, assume_fixed_input
):
    checkpoint = tmp_path / "local.pt"
    checkpoint.touch()
    captured = {}
    sentinel = object()

    def create_parallel(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(predict_module, "ParallelMLIPPredictUnit", create_parallel)
    result = pretrained_mlip.get_predict_unit(
        str(checkpoint),
        workers=2,
        device="cpu",
        inference_settings=settings(True),
        assume_fixed_input=assume_fixed_input,
    )
    assert result is sentinel
    assert captured["num_workers"] == 2
    assert captured["assume_fixed_input"] is assume_fixed_input


@pytest.mark.parametrize(
    "field",
    [
        "atomic_numbers",
        "charge",
        "spin",
        "dataset",
        "pbc",
        "fixed",
        "tags",
        "natoms",
        "batch",
        "nedges",
        "edge_index",
        "cell_offsets",
        "sid",
    ],
)
def test_metadata_changes_require_full_refresh(water_batch, field):
    previous = water_batch.clone()
    value = water_batch[field]
    if field == "dataset":
        value[0] = "omat"
    elif field == "sid":
        value[0] = "another structure"
    elif field == "pbc":
        value.logical_not_()
    elif value.numel():
        value.reshape(-1)[0] += 1
    else:
        water_batch[field] = torch.ones_like(value).resize_(1)
    assert not _only_geometry_changed(water_batch, previous)


def test_geometry_and_optional_fields(water_batch):
    previous = water_batch.clone()
    water_batch.pos += 0.01
    water_batch.cell *= 1.01
    assert _only_geometry_changed(water_batch, previous)
    water_batch.energy = torch.ones(1)
    assert not _only_geometry_changed(water_batch, previous)
    previous = water_batch.clone()
    del water_batch["energy"]
    assert not _only_geometry_changed(water_batch, previous)


@pytest.mark.parametrize("field", ["pos", "cell"])
def test_geometry_buffer_changes_require_refresh(water_batch, field):
    previous = water_batch.clone()
    water_batch[field] = water_batch[field].double()
    assert not _only_geometry_changed(water_batch, previous)
    water_batch[field] = previous[field].repeat(2, 1, 1)
    assert not _only_geometry_changed(water_batch, previous)


def changing_requests(water_batch):
    """
    Exercise permutation, resize, precision, and permanent MOLE fallback.
    """
    first = water_batch.clone()
    moved = first.clone()
    moved.pos[1, 0] += 0.03
    atoms = first.to_ase_single()
    reordered = make_batch(atoms[[1, 0, 2]])
    larger = make_batch(atoms.repeat((2, 1, 1)))
    double = reordered.clone()
    for key, value in double:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            double[key] = value.double()
    charged = first.clone()
    charged.charge += 1
    spin = charged.clone()
    spin.spin += 1
    material = spin.clone()
    material.dataset = ["omat"]
    periodic = material.clone()
    periodic.pbc.fill_(True)
    tagged = periodic.clone()
    tagged.tags[0] = 2
    tagged.fixed[1] = 1
    return [
        first,
        moved,
        reordered,
        larger,
        first.clone(),
        double,
        charged,
        spin,
        material,
        periodic,
        tagged,
        first.clone(),
    ]


@pytest.mark.serial()
@pytest.mark.parametrize("merge_mole", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_parallel_changing_inputs(
    tiny_parallel_checkpoint, water_batch, device, merge_mole
):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    requests = changing_requests(water_batch)
    reference = MLIPPredictUnit(
        tiny_parallel_checkpoint, device=device, inference_settings=settings(False)
    )
    expected = [reference.predict(data.clone()) for data in requests]
    parallel = ParallelMLIPPredictUnit(
        tiny_parallel_checkpoint,
        device=device,
        inference_settings=settings(merge_mole),
        num_workers=2,
        num_workers_per_node=2,
    )
    try:
        for index, (data, result) in enumerate(zip(requests, expected, strict=True)):
            current = data.clone()
            # Reuse the first request object with in-place metadata mutations.
            if index == 2:
                requests[0].atomic_numbers.copy_(data.atomic_numbers)
                requests[0].pos.copy_(data.pos)
                current = requests[0]
            actual = parallel.predict(current)
            for key in result:
                torch.testing.assert_close(
                    actual[key].cpu(), result[key].cpu(), atol=1e-5, rtol=1e-5
                )
            if index == 1:
                invalid = data.clone()
                invalid.natoms += 1
                with pytest.raises(AssertionError):
                    parallel.predict(invalid)
            assert parallel.inference_settings.merge_mole is (merge_mole and index < 6)
        assert parallel._broadcast_inputs is merge_mole
    finally:
        distutils.cleanup_gp_ray()


@pytest.mark.serial()
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_parallel_fixed_input_skips_checks(
    tiny_parallel_checkpoint, water_batch, device, monkeypatch
):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    requests = [water_batch.clone() for _ in range(3)]
    requests[1].pos[1, 0] += 0.03
    requests[2].cell *= 1.01
    reference = MLIPPredictUnit(
        tiny_parallel_checkpoint, device=device, inference_settings=settings(False)
    )
    expected = [reference.predict(data.clone()) for data in requests]

    def unexpected_check(*args, **kwargs):
        pytest.fail("Fixed-input mode must not check or serialize input metadata")

    monkeypatch.setattr(predict_module, "_only_geometry_changed", unexpected_check)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", unexpected_check)
    parallel = ParallelMLIPPredictUnit(
        tiny_parallel_checkpoint,
        device=device,
        inference_settings=settings(True),
        num_workers=2,
        num_workers_per_node=2,
        assume_fixed_input=True,
    )
    try:
        for data, result in zip(requests, expected, strict=True):
            actual = parallel.predict(data)
            for key in result:
                torch.testing.assert_close(
                    actual[key].cpu(), result[key].cpu(), atol=1e-5, rtol=1e-5
                )
        assert parallel._input_snapshot is None
        assert parallel._input_update is None
    finally:
        distutils.cleanup_gp_ray()
