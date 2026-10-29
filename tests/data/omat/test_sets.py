"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from fairchem.data.omat.vasp.sets import OMat24RelaxSet, OMat24StaticSet


@pytest.mark.parametrize("input_set_cls", [OMat24StaticSet, OMat24RelaxSet])
def test_omat24_kpoint_defaults(input_set_cls):
    input_set = input_set_cls()
    assert input_set.force_gamma is True
    assert input_set.auto_metal_kpoints is True


def test_omat24_kpoint_defaults_overridable():
    input_set = OMat24StaticSet(force_gamma=False, auto_metal_kpoints=False)
    assert input_set.force_gamma is False
    assert input_set.auto_metal_kpoints is False
