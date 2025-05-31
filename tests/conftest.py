"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# conftest.py
from __future__ import annotations

import random
from contextlib import suppress

import numpy as np
import pytest
import torch


@pytest.fixture()
def command_line_inference_checkpoint(request):
    return request.config.getoption("--inference-checkpoint")


@pytest.fixture()
def command_line_inference_dataset(request):
    return request.config.getoption("--inference-dataset")


def pytest_addoption(parser):
    parser.addoption(
        "--skip-ocpapi-integration",
        action="store_true",
        default=False,
        help="skip ocpapi integration tests",
    )
    parser.addoption(
        "--inference-checkpoint",
        action="store",
        help="inference checkpoint to run check on",
    )
    parser.addoption(
        "--inference-dataset", action="store", help="inference dataset to run check on"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ocpapi_integration: ocpapi integration test")
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")
    config.addinivalue_line(
        "markers", "cpu_and_gpu: mark test to run on both GPU and CPU workers"
    )


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
    if config.getoption("--inference-checkpoint"):
        # Skip all tests not marked with 'inference_check'
        for item in items:
            if "inference_check" not in item.keywords:
                item.add_marker(pytest.mark.skip(reason="skip all but inference check"))
    else:
        # Skip all tests marked with 'inference_check' by default
        skip_inference_check = pytest.mark.skip(
            reason="skipping inference check by default"
        )
        for item in items:
            if "inference_check" in item.keywords:
                item.add_marker(skip_inference_check)


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="function")
def seed_fixture():
    seed_everywhere(42)  # You can set your desired seed value here


@pytest.fixture(scope="function")
def compile_reset_state():
    torch.compiler.reset()
    yield
    torch.compiler.reset()
