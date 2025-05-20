"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def max_num_elements(dummy_element_refs):
    return len(dummy_element_refs) - 1
