"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import ray

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core.common import distutils


# Override the parent conftest's setup_before_each_test for tests in this directory
# that use module-scoped Ray clusters (like test_inference_serve.py)
@pytest.fixture(autouse=True)
def setup_before_each_test(request):
    """
    Modified version of parent's setup_before_each_test.
    
    Skips ray.shutdown() when using fixtures that manage Ray cluster lifecycle
    (like local_ray_cluster_with_inference) to avoid destroying the shared cluster.
    """
    # Check if this test uses a fixture that manages Ray
    uses_managed_ray = any(
        fixture_name in ("local_ray_cluster_with_inference",)
        for fixture_name in request.fixturenames
    )
    
    if not uses_managed_ray:
        # Standard cleanup for tests not using managed Ray clusters
        if ray.is_initialized():
            ray.shutdown()
    
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
    
    yield
    
    if not uses_managed_ray:
        # Standard cleanup
        if ray.is_initialized():
            ray.shutdown()
    
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()

