"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import platform
from pathlib import Path

from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.plan import orientation_seed
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.spec.transforms import haar_rotation, hashed_matrix
from fairchem.core.regression_suite.suites import SUITES, load_suite
from fairchem.core.regression_suite.suites.prediction_fidelity.suite import TASKS


def plan_digests() -> list[str]:
    """
    Describe what plan hashes depend on, to compare machines line by line.

    Structure digests cover the accuracy suite's checked-in structures as ASE
    reads them, the rotation digest covers seeded Haar rotations as plans hash
    them (truncated to 1e-7), and the last lines report whether each suite's
    plan reproduces its audited hash.
    """
    lines = [f"machine {platform.machine()}"]
    structures = Path(__file__).parent / "prediction_fidelity/structures"
    for task in TASKS:
        identities = frame_identities(structures / f"{task}.extxyz.xz")
        lines.append(f"structures {task} {stable_hash(identities)[:12]}")
    rotations = [
        hashed_matrix(haar_rotation(orientation_seed(index, orientation)))
        for index in range(70)
        for orientation in range(1, 9)
    ]
    lines.append(f"rotations {stable_hash(rotations)[:12]}")
    for suite in SUITES:
        try:
            load_suite(suite).plan()
            lines.append(f"plan {suite} ok")
        except Exception as error:
            lines.append(f"plan {suite} FAILS: {error}")
    return lines


if __name__ == "__main__":
    print("\n".join(plan_digests()))
