"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.common import utils
from fairchem.core.modules.scaling.scale_factor import ScaleFactor


def test_checkpoint_loading_matches_distributed_prefixes(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.save(
        {
            "config": {
                "model": {"name": "linear", "in_features": 2, "out_features": 2}
            },
            "state_dict": {"module.weight": weight, "module.bias": torch.zeros(2)},
        },
        path,
    )
    monkeypatch.setattr(utils.registry, "get_model_class", lambda name: torch.nn.Linear)
    model = utils.load_model_and_weights_from_checkpoint(str(path))
    assert torch.equal(model.weight, weight)
    assert torch.equal(model(torch.ones(1, 2)), torch.tensor([[3.0, 7.0]]))
    prefixed = utils.match_state_dict({"module.weight": weight}, {"weight": weight})
    assert set(prefixed) == {"module.weight"}
    assert prefixed["module.weight"] is weight
    with pytest.raises(FileNotFoundError):
        utils.load_model_and_weights_from_checkpoint(str(tmp_path / "missing.pt"))


def test_checkpoint_compatibility_filters_scale_factors_but_reports_weights(caplog):
    model = torch.nn.ModuleDict(
        {"scale": ScaleFactor(), "linear": torch.nn.Linear(1, 1)}
    )
    state = model.state_dict()
    state.pop("scale.scale_factor")
    state.pop("linear.weight")
    state["extra.weight"] = torch.ones(1)
    missing, unexpected = utils.load_state_dict(model, state, strict=False)
    assert missing == ["linear.weight"]
    assert unexpected == ["extra.weight"]
    assert "Missing key(s)" in caplog.text
    with pytest.raises(RuntimeError, match="Missing key"):
        utils.load_state_dict(model, state)
    state = model.state_dict()
    state["scale.legacy_buffer"] = torch.ones(1)
    assert utils.load_state_dict(model, state) == ([], [])


def test_conservative_force_configuration_enables_input_gradients():
    class Probe:
        @utils.conditional_grad(torch.enable_grad())
        def evaluate(self):
            return torch.is_grad_enabled()

    for new_config in (False, True):
        for forces, direct, expected in [
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ]:
            probe = Probe()
            if new_config:
                probe.regress_config = SimpleNamespace(
                    forces=forces, direct_forces=direct
                )
            else:
                probe.regress_forces = forces
                probe.direct_forces = direct
            with torch.no_grad():
                assert probe.evaluate() is expected


def test_weight_statistics_include_gradient_values_and_validate_checkpoint_format():
    model = torch.nn.Linear(2, 2)
    columns, rows = utils.get_weight_table(model)
    assert len(rows) == 2
    assert all(value is None for value in rows[0][-6:])
    model(torch.ones(1, 2)).sum().backward()
    columns, rows = utils.get_weight_table(model)
    assert all(value is not None for value in rows[0][-6:])
    assert len(rows[0]) == len(columns)
    assert utils.get_checkpoint_format({}) == "pt"
    assert utils.get_checkpoint_format({"optim": {"checkpoint_format": "dcp"}}) == "dcp"
    with pytest.raises(AssertionError, match="checkpoint format"):
        utils.get_checkpoint_format({"optim": {"checkpoint_format": "invalid"}})
