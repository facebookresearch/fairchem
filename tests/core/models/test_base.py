"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core.common.registry import registry
from fairchem.core.models.base import HydraModel


@pytest.mark.parametrize("model_id", [None, "", "   "])
def test_uma_moe_hydra_model_requires_model_id(model_id):
    backbone = {
        "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        "num_experts": 8,
    }

    with pytest.raises(ValueError, match="require a nonblank model_id"):
        HydraModel(backbone=backbone, heads={}, model_id=model_id)


def test_uma_moe_hydra_model_accepts_model_id(monkeypatch):
    backbone = {
        "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        "num_experts": 8,
    }

    class DummyBackbone:
        pass

    monkeypatch.setattr(registry, "get_model_class", lambda _: DummyBackbone)
    model = HydraModel(backbone=backbone, heads={}, model_id="UMA-explicit")

    assert model.model_id == "UMA-explicit"
    assert model.backbone.model_id == "UMA-explicit"


def test_non_uma_hydra_model_does_not_require_model_id(monkeypatch):
    backbone = {"model": "some.module.Backbone"}

    class DummyBackbone:
        pass

    monkeypatch.setattr(registry, "get_model_class", lambda _: DummyBackbone)
    model = HydraModel(backbone=backbone, heads={})

    assert model.model_id is None
    assert model.backbone.model_id is None
