"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import time

from ray import serve
from ray.serve.schema import ApplicationStatus

# This module is deliberately a leaf: Ray Serve lifecycle helpers with no
# fairchem imports, so consumers such as ``units.mlip_unit.predict`` can use
# them without importing ``components.batch_server`` (which imports back into
# ``units.mlip_unit`` and would form a cycle).

__all__ = [
    "get_app_handle_with_retry",
    "get_ray_connection_info",
    "wait_for_serve_ready",
]


def get_app_handle_with_retry(
    deployment_name: str,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 1.0,
):
    """
    Look up a Ray Serve app handle by name, retrying transient lookup
    failures for up to ``timeout_seconds``.

    The Serve controller is registered in the "serve" Ray namespace by the
    driver that called ``serve.start()``. A consumer (e.g. a Ray task on a
    fresh worker) may race the GCS sync of that actor entry and see a
    transient "SERVE_CONTROLLER_ACTOR not found" failure. Non-transient
    errors are re-raised immediately.

    ``_prefer_local_routing=True`` is applied via ``handle._init()``
    immediately after the handle is obtained, before any ``.options()``
    or ``.remote()`` call can implicitly initialize it with the default
    (``False``) value.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            handle = serve.get_app_handle(deployment_name)
            # ``_prefer_local_routing`` must be set via ``_init()`` before any
            # ``.options()`` or ``.remote()`` call initializes the handle.
            handle._init(_prefer_local_routing=True)
            return handle
        except Exception as exc:
            msg = str(exc)
            transient = (
                "SERVE_CONTROLLER_ACTOR" in msg
                or "There is no Serve instance" in msg
                or "Failed to look up actor" in msg
            )
            if not transient or time.monotonic() > deadline:
                raise
            logging.debug("Serve controller not visible yet (%s); retrying.", msg)
            time.sleep(poll_interval_seconds)


def wait_for_serve_ready(
    app_name: str,
    poll_interval_seconds: float = 2,
    timeout_seconds: float = 600,
) -> bool:
    """
    Wait for Ray Serve to be fully ready to accept requests.

    Blocks until the Ray Serve controller is running and the specified
    application reaches RUNNING status.

    Args:
        app_name: Name of the Ray Serve application to wait for.
        poll_interval_seconds: How often to check status.
        timeout_seconds: Maximum total time to wait before raising
            ``TimeoutError``. Prevents indefinite hangs when a deployment
            cannot be scheduled (e.g. no free GPU).

    Returns:
        True if server is ready.

    Raises:
        RuntimeError: If server fails to deploy.
        TimeoutError: If the application does not reach RUNNING within
            ``timeout_seconds``.
    """
    deadline = time.monotonic() + timeout_seconds

    def _check_deadline(phase: str) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for Ray Serve "
                f"({phase}) for app {app_name!r}."
            )

    # Phase 1: Wait for Ray Serve controller
    logging.info("Waiting for Ray Serve controller to start...")
    while True:
        try:
            status = serve.status()
            logging.info("Ray Serve controller is running")
            break
        except Exception as e:
            error_msg = str(e)
            if (
                "SERVE_CONTROLLER_ACTOR" in error_msg
                or "Failed to look up actor" in error_msg
            ):
                logging.debug(f"Ray Serve controller not ready yet: {error_msg}")
                _check_deadline("controller startup")
                time.sleep(poll_interval_seconds)
            else:
                raise

    # Phase 2: Wait for the application to be deployed and running
    logging.info(f"Waiting for application '{app_name}' to be ready...")
    while True:
        _check_deadline("application RUNNING")
        try:
            status = serve.status()

            if app_name not in status.applications:
                logging.debug(f"Application '{app_name}' not found yet, waiting...")
                time.sleep(poll_interval_seconds)
                continue

            app_status = status.applications[app_name]

            if app_status.status == ApplicationStatus.RUNNING:
                logging.info(f"Application '{app_name}' is RUNNING and ready")
                return True
            elif app_status.status == ApplicationStatus.DEPLOYING:
                logging.debug(f"Application '{app_name}' is still deploying...")
                time.sleep(poll_interval_seconds)
            elif app_status.status in (
                ApplicationStatus.DEPLOY_FAILED,
                ApplicationStatus.UNHEALTHY,
            ):
                raise RuntimeError(
                    f"Application '{app_name}' failed to deploy. "
                    f"Status: {app_status.status}, Message: {app_status.message}"
                )
            else:
                logging.debug(f"Application '{app_name}' status: {app_status.status}")
                time.sleep(poll_interval_seconds)

        except RuntimeError:
            raise
        except Exception as e:
            logging.warning(f"Error checking serve status: {e}")
            time.sleep(poll_interval_seconds)


def get_ray_connection_info(head_file: str) -> dict[str, str | None]:
    """
    Read Ray connection info from a head.json file.

    Args:
        head_file: Path to head.json file from a Ray cluster.

    Returns:
        Dictionary with ``ray_address``, ``namespace_serve_fairchem``, and
        ``local`` keys. For local clusters ``ray_address`` is *None*.
    """
    with open(head_file) as f:
        head_info = json.load(f)

    namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")
    is_local = head_info.get("local", False)

    if is_local:
        return {
            "ray_address": None,
            "namespace_serve_fairchem": namespace_serve_fairchem,
            "local": True,
        }

    hostname = head_info.get("hostname")
    client_port = head_info.get("client_port")

    if not hostname or not client_port:
        raise ValueError(
            f"Invalid head.json: missing hostname or client_port in {head_file}"
        )

    return {
        "ray_address": f"ray://{hostname}:{client_port}",
        "namespace_serve_fairchem": namespace_serve_fairchem,
        "local": False,
    }
