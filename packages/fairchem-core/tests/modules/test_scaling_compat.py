"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json

import pytest
import torch

from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.modules.scaling.scale_factor import ScaleFactor


@pytest.mark.parametrize("source", ["dict", "json", "pt"])
def test_load_named_and_unnamed_scales(tmp_path, caplog, source):
    module = torch.nn.ModuleDict(
        {"first": ScaleFactor("named"), "second": ScaleFactor()}
    )
    scales = {"named": 2.0, "second": 3.0, "unused": 4.0}
    if source == "json":
        path = tmp_path / "scales.json"
        path.write_text(json.dumps({**scales, "comment": "legacy format"}))
        scales = str(path)
    elif source == "pt":
        path = tmp_path / "scales.pt"
        torch.save(scales, path)
        scales = str(path)
    load_scales_compat(module, scales)
    assert module["first"].scale_factor == 2.0
    assert module["second"].scale_factor == 3.0
    assert "unused not found" in caplog.text


def test_missing_and_invalid_scale_files(tmp_path):
    module = ScaleFactor("required")
    for empty in (None, {}, ""):
        load_scales_compat(module, empty)
        assert not module.fitted
    with pytest.raises(ValueError, match="Missing entries"):
        load_scales_compat(module, {"other": 1.0})
    with pytest.raises(ValueError, match="does not exist"):
        load_scales_compat(module, str(tmp_path / "missing.json"))
    path = tmp_path / "scale.txt"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported"):
        load_scales_compat(module, str(path))
    path = tmp_path / "empty.json"
    path.write_text("{}")
    load_scales_compat(module, str(path))
    assert not module.fitted
