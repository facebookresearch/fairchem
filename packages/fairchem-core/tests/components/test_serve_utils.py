"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fairchem.core.components import serve_utils


@pytest.fixture()
def ray_serve():
    """
    Import the optional Serve API used by the lifecycle helpers.
    """
    return pytest.importorskip(
        "ray.serve",
        reason="Requires `ray[serve]` to be installed",
        exc_type=ImportError,
    )


@pytest.fixture()
def application_status(ray_serve):
    """
    Import application states after checking that Serve is available.
    """
    from ray.serve.schema import ApplicationStatus

    return ApplicationStatus


@pytest.fixture()
def clock(monkeypatch, ray_serve):
    """
    Advance a virtual clock instead of sleeping during retries.
    """
    now = [0.0]
    monkeypatch.setattr(serve_utils.time, "monotonic", lambda: now[0])

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(serve_utils.time, "sleep", advance)
    return now


@pytest.mark.parametrize(
    "message",
    [
        "SERVE_CONTROLLER_ACTOR not found",
        "There is no Serve instance",
        "Failed to look up actor",
    ],
)
def test_handle_lookup_retries_and_initializes_local_routing(
    monkeypatch, clock, ray_serve, message
):
    handle = Mock()
    lookup = Mock(side_effect=[RuntimeError(message), handle])
    monkeypatch.setattr(ray_serve, "get_app_handle", lookup)
    assert serve_utils.get_app_handle_with_retry("model") is handle
    assert lookup.call_count == 2
    handle._init.assert_called_once_with(_prefer_local_routing=True)
    assert clock[0] == 1


@pytest.mark.parametrize("message", ["invalid application", "Failed to look up actor"])
def test_handle_lookup_propagates_permanent_or_expired_error(
    monkeypatch, clock, ray_serve, message
):
    lookup = Mock(side_effect=RuntimeError(message))
    monkeypatch.setattr(ray_serve, "get_app_handle", lookup)
    with pytest.raises(RuntimeError, match=message):
        serve_utils.get_app_handle_with_retry("model", timeout_seconds=1)
    assert lookup.call_count == (3 if "actor" in message else 1)


def _status(status, message=""):
    return SimpleNamespace(
        applications={"model": SimpleNamespace(status=status, message=message)}
    )


def test_wait_for_serve_controller_and_deployment(
    monkeypatch, clock, ray_serve, application_status
):
    statuses = Mock(
        side_effect=[
            RuntimeError("SERVE_CONTROLLER_ACTOR not found"),
            SimpleNamespace(applications={}),
            SimpleNamespace(applications={}),
            _status(application_status.DEPLOYING),
            ValueError("temporary status failure"),
            _status(application_status.NOT_STARTED),
            _status(application_status.RUNNING),
        ]
    )
    monkeypatch.setattr(ray_serve, "status", statuses)
    assert serve_utils.wait_for_serve_ready("model", poll_interval_seconds=1)
    assert clock[0] == 5


@pytest.mark.parametrize("status", ["DEPLOY_FAILED", "UNHEALTHY"])
def test_wait_for_serve_reports_failed_application(
    monkeypatch, clock, ray_serve, application_status, status
):
    monkeypatch.setattr(
        ray_serve,
        "status",
        Mock(return_value=_status(application_status[status], "bad model")),
    )
    with pytest.raises(RuntimeError, match="bad model"):
        serve_utils.wait_for_serve_ready("model")


@pytest.mark.parametrize("controller_ready", [False, True])
def test_wait_for_serve_times_out(monkeypatch, clock, ray_serve, controller_ready):
    status = Mock(return_value=SimpleNamespace(applications={}))
    if not controller_ready:
        status.side_effect = RuntimeError("Failed to look up actor")
    monkeypatch.setattr(ray_serve, "status", status)
    with pytest.raises(TimeoutError, match="model"):
        serve_utils.wait_for_serve_ready("model", timeout_seconds=1)


def test_wait_for_serve_propagates_controller_error(monkeypatch, clock, ray_serve):
    monkeypatch.setattr(
        ray_serve, "status", Mock(side_effect=ValueError("invalid controller"))
    )
    with pytest.raises(ValueError, match="invalid controller"):
        serve_utils.wait_for_serve_ready("model")


@pytest.mark.parametrize(
    "head, expected",
    [
        ({"local": True}, {"ray_address": None, "local": True}),
        (
            {"hostname": "head", "client_port": 10001},
            {"ray_address": "ray://head:10001", "local": False},
        ),
    ],
)
def test_read_ray_connection_info(tmp_path, head, expected):
    head["namespace_serve_fairchem"] = "test"
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head))
    assert serve_utils.get_ray_connection_info(str(path)) == {
        **expected,
        "namespace_serve_fairchem": "test",
    }


@pytest.mark.parametrize("head", [{}, {"hostname": "head"}, {"client_port": 10001}])
def test_read_ray_connection_info_rejects_missing_address(tmp_path, head):
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head))
    with pytest.raises(ValueError, match="missing hostname or client_port"):
        serve_utils.get_ray_connection_info(str(path))
