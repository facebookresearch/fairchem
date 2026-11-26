"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    # The repo-root conftest registers the same option; when this
    # suite is collected together with the core tests (the CI core
    # shard), both conftests load and registration would conflict.
    try:
        parser.addoption(
            "--skip-ocpapi-integration",
            action="store_true",
            default=False,
            help="skip ocpapi integration tests",
        )
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ocpapi_integration_test: ocpapi integration test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-ocpapi-integration"):
        skip_ocpapi_integration = pytest.mark.skip(reason="skipping ocpapi integration")
        for item in items:
            if "ocpapi_integration_test" in item.keywords:
                item.add_marker(skip_ocpapi_integration)
