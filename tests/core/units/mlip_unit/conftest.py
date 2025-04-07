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
