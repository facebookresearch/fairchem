"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pretrained_checkpoint(request):
    """
    Name of the pretrained checkpoint under test.

    Resolved from the closest ``@pytest.mark.pretrained(...)`` marker,
    mirroring the repo-root conftest so this suite runs standalone
    from ``packages/fairchem-lammps``.
    """
    if hasattr(request, "param"):
        return request.param
    marker = request.node.get_closest_marker("pretrained")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{request.node.nodeid} uses pretrained_checkpoint but its "
            "closest @pytest.mark.pretrained marker is missing or has "
            "no model arguments."
        )
    return marker.args[0]
