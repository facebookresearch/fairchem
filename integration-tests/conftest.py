"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Cross-package suites reuse fairchem-core's test machinery: checkpoint
# parametrization (models_to_test / --sweep-model), the predict-unit
# factory, sweep options and markers, and the per-test seeding fixture.
# It is registered as a pytest plugin; pytest.ini's pythonpath setting
# makes fairchem-core's tests package importable from here.
pytest_plugins = ["tests.conftest"]

from tests.conftest import get_predict_unit_for_test, models_to_test  # noqa: E402, F401
