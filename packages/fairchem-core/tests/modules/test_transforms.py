"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.modules import transforms


@pytest.mark.parametrize("name", ["ani1x", "trans1x", "spice", "qm9", "omol", "qmof"])
def test_molecular_transforms_preserve_geometry_and_set_dataset(name):
    data = AtomicData.from_ase(molecule("H2O"))
    data.energy = -1.0
    original = torch.cdist(data.pos, data.pos)
    result = getattr(transforms, f"{name}_transform")(data, {"dataset_name": name})
    assert result is data
    assert result.dataset == name
    assert result.energy.dtype == torch.float32
    assert result.energy.shape == (1,)
    assert torch.allclose(torch.cdist(result.pos, result.pos), original, atol=1e-5)
    assert torch.equal(result.fixed, torch.zeros(3))
    if name != "qmof":
        assert torch.all(result.cell.diagonal(dim1=-2, dim2=-1) >= 240)
    if name == "spice":
        assert torch.equal(result.tags, torch.full((3,), 2))


def test_common_transform_defaults_and_tensor_conversion():
    data = SimpleNamespace(energy=2.0)
    transforms.common_transform(data, {"dataset_name": "test"})
    assert data.charge == data.spin == 0
    assert torch.equal(data.energy, torch.tensor([2.0]))
    transforms.ensure_tensor(data, ["energy", "missing"])
    assert data.energy.shape == (1,)
    assert not hasattr(data, "missing")


def test_transform_pipeline_skips_normalization_and_handles_empty_config():
    data = SimpleNamespace(energy=2.0)
    assert transforms.DataTransforms(None)(data) is data
    pipeline = transforms.DataTransforms(
        {
            "normalizer": {},
            "element_references": {},
            "common_transform": {"dataset_name": "test"},
        }
    )
    assert pipeline(data).dataset == "test"


def test_stress_reshape_and_tensor_decomposition():
    data = {"stress": torch.eye(3), "stress_iso": torch.ones(1)}
    transforms.stress_reshape_transform(data, {})
    assert data["stress"].shape == (1, 9)
    assert data["stress_iso"].shape == (1,)
    config = {
        "tensor": "stress",
        "rank": 2,
        "decomposition": {"scalar": {"irrep_dim": 0}, "traceless": {"irrep_dim": 2}},
    }
    transforms.decompose_tensor(data, config)
    assert data["scalar"].shape == (1, 1)
    assert data["traceless"].shape == (1, 5)
    assert torch.allclose(data["traceless"], torch.zeros(1, 5), atol=1e-6)
    assert transforms.decompose_tensor({}, config) == {}
    with pytest.raises(NotImplementedError):
        transforms.decompose_tensor(data, {**config, "rank": 1})
