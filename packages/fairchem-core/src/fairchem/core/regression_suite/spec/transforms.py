"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.schema import (
    SCHEMA_VERSION,
    CaseSpec,
    RegressionSchemaError,
    SuiteCase,
    TransformSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms


# Hashes see rotation matrices truncated to this many decimals, so machines
# whose floating-point libraries differ in the last bits agree on plan hashes and
# transform IDs. Computation always uses the full-precision matrix.
_HASHED_DECIMALS = 7


def apply_case_transform(atoms: Atoms, case: CaseSpec) -> SuiteCase:
    """
    Apply a frozen proper rotation without modifying the source atoms.
    """
    transformed = atoms.copy()
    transformed.info = copy.deepcopy(atoms.info)
    matrix = np.asarray(case.transform.matrix, dtype=np.float64)
    transformed.positions[:] = transformed.positions @ matrix.T
    transformed.set_cell(np.asarray(transformed.cell) @ matrix.T, scale_atoms=False)
    return SuiteCase(spec=case, transformed_atoms=transformed)


def canonicalize_vectors(values: Any, case: SuiteCase) -> np.ndarray:
    """
    Rotate Cartesian vectors from the transformed frame into the source frame.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (3,):
        raise RegressionSchemaError("Cartesian vectors must end in dimension 3")
    matrix = np.asarray(case.spec.transform.matrix, dtype=np.float64)
    return array @ matrix


def canonicalize_rank_two(values: Any, case: SuiteCase) -> np.ndarray:
    """
    Rotate a Cartesian rank-two tensor into the source frame.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3, 3):
        raise RegressionSchemaError("rank-two tensor must have shape (3, 3)")
    matrix = np.asarray(case.spec.transform.matrix, dtype=np.float64)
    return matrix.T @ array @ matrix


def hashed_matrix(matrix: Sequence[Sequence[float]]) -> list[list[int]]:
    """
    Return a rotation matrix as truncated integer counts of 1e-7 for hashing.
    """
    scale = 10**_HASHED_DECIMALS
    return [[int(value * scale) for value in row] for row in matrix]


def haar_rotation(seed: int) -> list[list[float]]:
    """
    Return the rotation of a seeded, uniformly random unit quaternion.
    """
    quaternion = np.random.default_rng(seed).normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    w, x, y, z = quaternion
    rows = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    return np.asarray(rows, dtype=np.float64).tolist()


def seeded_transform(
    sample_seed: int,
    *,
    identity: bool = False,
    transform_id: str | None = None,
) -> TransformSpec:
    """
    Recreate an audited identity or seeded Haar rotation.
    """
    kind = "identity" if identity else "rotation"
    matrix = np.eye(3).tolist() if identity else haar_rotation(sample_seed)
    if transform_id is None:
        parameters = (
            {}
            if identity
            else {"distribution": "uniform_haar_so3", "matrix": hashed_matrix(matrix)}
        )
        identity_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "sample_seed": sample_seed,
            "parameters": parameters,
        }
        transform_id = (
            f"precision-transform-v1/{kind}/{stable_hash(identity_payload)[:20]}"
        )
    return TransformSpec(
        transform_id=transform_id,
        kind=kind,
        matrix=tuple(tuple(row) for row in matrix),
        sample_seed=sample_seed,
    )
