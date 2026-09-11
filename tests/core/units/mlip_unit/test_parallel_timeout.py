"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core.units.mlip_unit.predict import (
    MLIPWorkerLocal,
    _parallel_mlip_process_group_timeout,
)


def test_parallel_mlip_process_group_timeout(monkeypatch):
    monkeypatch.delenv("FAIRCHEM_PARALLEL_MLIP_TIMEOUT_SECONDS", raising=False)
    assert _parallel_mlip_process_group_timeout() is None

    monkeypatch.setenv("FAIRCHEM_PARALLEL_MLIP_TIMEOUT_SECONDS", "3600.5")
    timeout = _parallel_mlip_process_group_timeout()
    assert timeout is not None
    assert timeout.total_seconds() == 3600.5


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_parallel_mlip_process_group_timeout_invalid(monkeypatch, value):
    monkeypatch.setenv("FAIRCHEM_PARALLEL_MLIP_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="FAIRCHEM_PARALLEL_MLIP_TIMEOUT_SECONDS"):
        _parallel_mlip_process_group_timeout()


def test_parallel_worker_applies_process_group_timeout(monkeypatch):
    calls = []
    monkeypatch.setenv("FAIRCHEM_PARALLEL_MLIP_TIMEOUT_SECONDS", "3600")
    monkeypatch.setattr(
        "fairchem.core.units.mlip_unit.predict.setup_env_local_multi_gpu",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "fairchem.core.units.mlip_unit.predict.assign_device_for_local_rank",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "fairchem.core.units.mlip_unit.predict.dist.init_process_group",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "fairchem.core.units.mlip_unit.predict.hydra.utils.instantiate",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "fairchem.core.units.mlip_unit.predict.get_device_for_local_rank",
        lambda: "cpu",
    )
    worker = MLIPWorkerLocal(
        worker_id=0,
        world_size=1,
        predictor_config={"device": "cpu"},
        master_port=12345,
        master_address="localhost",
    )

    worker._distributed_setup()

    assert calls[0]["timeout"].total_seconds() == 3600
