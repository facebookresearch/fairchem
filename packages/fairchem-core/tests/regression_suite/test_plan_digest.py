"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np

from fairchem.core.regression_suite.spec import transforms as transforms_module
from fairchem.core.regression_suite.spec.transforms import hashed_matrix
from fairchem.core.regression_suite.suites import SUITES, load_suite
from fairchem.core.regression_suite.suites.plan_digest import plan_digests


def test_plan_digests_cover_structures_rotations_and_every_plan():
    lines = plan_digests()

    assert lines[0].startswith("machine ")
    assert sum(line.startswith("structures ") for line in lines) == 7
    assert sum(line.startswith("rotations ") for line in lines) == 1
    assert [line for line in lines if line.startswith("plan ")] == [
        f"plan {suite} ok" for suite in SUITES
    ]


def test_hashed_rotations_ignore_last_bit_differences_between_machines(monkeypatch):
    seeds = set()
    rotation = transforms_module.haar_rotation
    monkeypatch.setattr(
        transforms_module,
        "haar_rotation",
        lambda seed: seeds.add(seed) or rotation(seed),
    )
    for suite in SUITES:
        load_suite(suite).plan()

    # Other CPUs and math libraries differ from this one by a few ulps.
    assert len(seeds) == 560
    for seed in seeds:
        matrix = np.asarray(rotation(seed))
        expected = hashed_matrix(matrix)
        for direction in (np.inf, -np.inf):
            nudged = matrix.copy()
            for _ in range(8):
                nudged = np.nextafter(nudged, direction)
                assert hashed_matrix(nudged) == expected, seed
