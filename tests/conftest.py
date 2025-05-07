"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# conftest.py
from __future__ import annotations

from contextlib import suppress

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--skip-ocpapi-integration",
        action="store_true",
        default=False,
        help="skip ocpapi integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ocpapi_integration: ocpapi integration test")
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")


def pytest_runtest_setup(item):
    # Check if the test has the 'gpu' marker
    if (
        "gpu" in item.keywords
        and "cpu_and_gpu" not in item.keywords
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA not available, skipping GPU test")
    if "dgl" in item.keywords:
        # check dgl is installed
        fairchem_cpp_found = False
        with suppress(ModuleNotFoundError):
            import fairchem_cpp

            unused = (  # noqa: F841
                fairchem_cpp.__file__
            )  # prevent the linter from deleting the import
            fairchem_cpp_found = True
        if not fairchem_cpp_found:
            pytest.skip(
                "fairchem_cpp not found, skipping DGL tests! please install fairchem if you want to run these"
            )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-ocpapi-integration"):
        skip_ocpapi_integration = pytest.mark.skip(reason="skipping ocpapi integration")
        for item in items:
            if "ocpapi_integration_test" in item.keywords:
                item.add_marker(skip_ocpapi_integration)
        return
