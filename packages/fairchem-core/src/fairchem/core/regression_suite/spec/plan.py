"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.schema import (
    PLAN_SCHEMA,
    SCHEMA_VERSION,
    CaseSpec,
    RegressionSchemaError,
    SuitePlan,
)
from fairchem.core.regression_suite.spec.transforms import (
    hashed_matrix,
    seeded_transform,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _hashed_case(case: CaseSpec) -> dict[str, Any]:
    payload = dataclasses.asdict(case)
    if case.transform.kind == "rotation":
        payload["transform"]["matrix"] = hashed_matrix(case.transform.matrix)
    return payload


def declarative_plan(
    suite: str,
    cases: Sequence[CaseSpec],
    expected_hash: str,
) -> SuitePlan:
    """
    Build a plan and reject changes to its audited expanded representation.
    """
    identities = [(case.case_id, case.transform.transform_id) for case in cases]
    if len(identities) != len(set(identities)):
        raise RegressionSchemaError(f"plan {suite!r} case identities must be unique")
    plan_hash = stable_hash(
        {
            "schema": PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "suite": suite,
            "cases": [_hashed_case(case) for case in cases],
        }
    )
    if plan_hash != expected_hash:
        raise RegressionSchemaError(
            f"declarative plan {suite!r} no longer matches its audited hash; "
            f"it now hashes to {plan_hash}"
        )
    return SuitePlan(suite=suite, cases=tuple(cases), plan_hash=plan_hash)


def precision_rotation_sweep(*, sample_seed: int, **case: Any) -> list[CaseSpec]:
    """
    Declare one identity case followed by eight seeded rotations.
    """
    return [
        CaseSpec(
            **case,
            transform=seeded_transform(sample_seed + offset, identity=offset == 0),
        )
        for offset in range(9)
    ]


def orientation_seed(index: int, orientation: int) -> int:
    """
    Return the rotation seed of structure ``index`` in ``orientation``.
    """
    return 42 + 10_000 * index + orientation


def oriented_case(index: int, orientation: int, **case: Any) -> CaseSpec:
    """
    Declare structure ``index`` in one of its seeded orientations; orientation
    zero is the identity.
    """
    return CaseSpec(
        **case,
        transform=seeded_transform(
            orientation_seed(index, orientation),
            identity=orientation == 0,
            transform_id=f"orientation-{orientation:02d}",
        ),
    )


def unrotated_case(
    *, sample_seed: int = 42, transform_id: str = "identity", **case: Any
) -> CaseSpec:
    """
    Declare a case in its source orientation.
    """
    return CaseSpec(
        **case,
        transform=seeded_transform(
            sample_seed, identity=True, transform_id=transform_id
        ),
    )
