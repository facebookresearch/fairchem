"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import tempfile

import pytest

from tests.core.units.mlip_unit.create_fake_dataset import (
    create_fake_puma_dataset,
)


@pytest.fixture(scope="session")
def fake_puma_dataset():
    with tempfile.TemporaryDirectory() as tempdirname:
        datasets_yaml = create_fake_puma_dataset(tempdirname)
        yield datasets_yaml
