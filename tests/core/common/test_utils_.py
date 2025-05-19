"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.common.utils import get_deep


def test_get_deep() -> None:
    d = {"oc20": {"energy": 1.5}}
    assert get_deep(d, "oc20.energy") == 1.5
    assert get_deep(d, "oc20.force", 0.9) == 0.9
    assert get_deep(d, "omol.energy") is None
