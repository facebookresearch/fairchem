"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Cross-package suites live here; the shared test helpers and options
# are owned by fairchem-core's test suite. Load that conftest by path
# (it cannot be imported as ``tests.conftest`` from this directory)
# and re-export the pieces these suites use.
_CORE_TESTS_DIR = Path(__file__).parent.parent / "packages/fairchem-core/tests"
# Parent dir so ``tests`` (fairchem-core's test package) is importable,
# and the tests dir itself so helpers like ``performance_report`` are.
sys.path.insert(0, str(_CORE_TESTS_DIR.parent))
sys.path.insert(0, str(_CORE_TESTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "fairchem_core_tests_conftest", _CORE_TESTS_DIR / "conftest.py"
)
_core_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core_conftest)

get_predict_unit_for_test = _core_conftest.get_predict_unit_for_test
models_to_test = _core_conftest.models_to_test


def pytest_addoption(parser):
    _core_conftest.pytest_addoption(parser)


def pytest_configure(config):
    _core_conftest.pytest_configure(config)


def pytest_generate_tests(metafunc):
    _core_conftest.pytest_generate_tests(metafunc)


def pytest_runtest_setup(item):
    _core_conftest.pytest_runtest_setup(item)


def pytest_collection_modifyitems(config, items):
    _core_conftest.pytest_collection_modifyitems(config, items)


@pytest.fixture(autouse=True)
def setup_before_each_test():
    yield from _core_conftest.setup_before_each_test.__wrapped__()
