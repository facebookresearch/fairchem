"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


ComparisonKind = Literal["absolute", "relative_with_floor", "log_ratio"]

PLAN_SCHEMA = "fairchem-regression-plan"
REFERENCE_SCHEMA = "fairchem-regression-reference"
BASELINE_SCHEMA = "fairchem-regression-baseline"
CANDIDATE_REPORT_SCHEMA = "fairchem-regression-candidate-report"
# Plans and records retain their original schema. Baselines changed as
# qualification gained separate primary and absolute-error channels,
# baseline-fitted report axes, execution-platform provenance, per-suite source
# commits, and independent finite-difference and autograd frequency suites.
SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 8
CANDIDATE_REPORT_SCHEMA_VERSION = 4
PRECISION_TIERS = ("FP64", "FP32", "TF32")
SUMMARY_STATISTICS = (
    "median_absolute",
    "mean_absolute",
    "iqr",
    "p95",
    "p99",
)
DECISION_STATISTICS = ("median_absolute", "p95")
# The summary statistics and decision-cluster mean field of each error channel.
CHANNEL_FIELDS = {
    "primary": ("statistics", "primary_mean"),
    "absolute": ("absolute_statistics", "absolute_mean"),
}


class RegressionSchemaError(ValueError):
    """
    Raised when a regression-suite artifact violates its schema.
    """


class CommandError(Exception):
    """
    Raised for an expected failure, such as a bad argument or failed tasks,
    that the command line reports in one line instead of a traceback.
    """


@dataclass(frozen=True)
class HistogramSpec:
    """
    Logarithmic bounds for one observable's errors.

    Report axes fall back to these bounds when a baseline has no positive
    decision values. ``bins_per_decade`` is unused, but every field is hashed
    into suite contracts, so removing it would invalidate published baselines.
    """

    minimum: float
    maximum: float
    bins_per_decade: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum < self.maximum < math.inf:
            raise RegressionSchemaError(
                "histogram bounds must satisfy 0 < minimum < maximum"
            )


@dataclass(frozen=True)
class ObservableSpec:
    """
    Comparison and presentation contract for a suite observable.
    """

    units: str
    comparison: ComparisonKind
    histogram: HistogramSpec
    floor: float = 0.0
    report_only: bool = False

    def __post_init__(self) -> None:
        if not self.units:
            raise RegressionSchemaError("observable units must be non-empty")
        if self.comparison not in get_args(ComparisonKind):
            raise RegressionSchemaError(
                f"unsupported comparison kind: {self.comparison}"
            )
        if not math.isfinite(self.floor) or self.floor < 0.0:
            raise RegressionSchemaError("observable floor must be non-negative")
        if self.comparison == "relative_with_floor" and self.floor <= 0.0:
            raise RegressionSchemaError(
                "relative_with_floor observables require a positive floor"
            )


@dataclass(frozen=True)
class TransformSpec:
    """
    Fully realized transformation stored in a suite plan.
    """

    transform_id: str
    kind: Literal["identity", "rotation"]
    matrix: tuple[tuple[float, float, float], ...]
    sample_seed: int

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12, rtol=0.0):
            raise RegressionSchemaError("transform matrix must be orthogonal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12, rtol=0.0):
            raise RegressionSchemaError("transform matrix must be a proper rotation")
        if self.kind == "identity" and not np.array_equal(matrix, np.eye(3)):
            raise RegressionSchemaError(
                "identity transform must use the identity matrix"
            )


@dataclass(frozen=True)
class CaseSpec:
    """
    One frozen structure/transformation execution from a suite plan.
    """

    case_id: str
    structure: str
    structure_index: int
    structure_hash: str
    task_name: str
    transform: TransformSpec
    runtime_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuitePlan:
    """
    Immutable collection of cases for one independent suite.
    """

    suite: str
    cases: tuple[CaseSpec, ...]
    plan_hash: str


@dataclass(frozen=True)
class SuiteCase:
    """
    Materialized plan case passed to a suite implementation.
    """

    spec: CaseSpec
    transformed_atoms: Atoms


@dataclass(frozen=True)
class MeasurementRecord:
    """
    Measurement plus the exact case identity needed for pairing.
    """

    case_id: str
    transform_id: str
    observable: str
    values: np.ndarray
    units: str
    comparison: ComparisonKind
    floor: float
    cluster_id: str
    report_only: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.size == 0 or values.dtype.kind not in "fiu":
            raise RegressionSchemaError(f"{self.observable!r} values must be numeric")
        if not np.isfinite(values).all():
            raise RegressionSchemaError(f"{self.observable!r} values must be finite")
        object.__setattr__(self, "values", values.astype(np.float64, copy=False))

    @property
    def key(self) -> tuple[str, str, str]:
        """
        Return the exact pairing identity.
        """
        return self.case_id, self.transform_id, self.observable


PROFILE_TIERS = (
    ("fp64-eager", "FP64"),
    ("fp32-eager", "FP32"),
    ("tf32-eager", "TF32"),
)


class PerformanceMetricSpec(NamedTuple):
    """
    Contract for a directional performance metric.
    """

    units: str
    objective: Literal["minimize", "maximize"]
    group_by: tuple[str, ...]
