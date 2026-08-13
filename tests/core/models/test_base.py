"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import re

import pytest

from fairchem.core.common.registry import registry
from fairchem.core.models.base import HydraModel


@pytest.mark.parametrize("model_id", [None, "", "   "])
def test_uma_moe_hydra_model_generates_model_id(model_id, caplog, monkeypatch):
    backbone = {
        "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        "num_experts": 8,
    }

    class DummyBackbone:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(registry, "get_model_class", lambda _: DummyBackbone)
    with caplog.at_level(logging.WARNING):
        model = HydraModel(backbone=backbone, heads={}, model_id=model_id)

    assert re.fullmatch(r"UMA-[0-9a-f]{12}", model.model_id)
    assert model.backbone.model_id == model.model_id
    assert f"Generated model_id='{model.model_id}'" in caplog.text
