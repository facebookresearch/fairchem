"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# ``MeasurementRecord.values`` is a NumPy array, not a pandas object.
# ruff: noqa: PD011
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core.regression_suite.artifacts.storage import load_reference
from fairchem.core.regression_suite.spec.schema import (
    CHANNEL_FIELDS,
    MeasurementRecord,
    ObservableSpec,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from fairchem.core.regression_suite.suites.base import Suite


def summary_statistics(values: Sequence[float]) -> dict[str, float]:
    """
    Compute the canonical statistics stored for one observable summary.
    """
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mean_absolute": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
    }


def _primary_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    spec: ObservableSpec,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(candidate - reference)
    if spec.comparison == "absolute":
        return absolute, absolute
    if spec.comparison == "relative_with_floor":
        return absolute / np.maximum(np.abs(reference), spec.floor), absolute
    if np.any(reference <= 0.0) or np.any(candidate <= 0.0):
        raise RegressionSchemaError(
            "log-ratio comparisons require strictly positive values"
        )
    return np.abs(np.log(candidate / reference)), absolute


def _observable_summary(
    spec: ObservableSpec,
    clusters: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    case_count: int,
) -> dict[str, Any]:
    """
    Summarize one observable from each cluster's primary and absolute
    differences.
    """
    decision_clusters = [
        {
            "cluster_id": cluster_id,
            "primary_mean": float(np.mean(primary)),
            "absolute_mean": float(np.mean(absolute)),
        }
        for cluster_id, (primary, absolute) in clusters.items()
    ]
    return {
        "units": spec.units,
        "comparison": spec.comparison,
        "floor": spec.floor,
        "report_only": spec.report_only,
        "population": {
            "cases": case_count,
            "clusters": len(decision_clusters),
            "values": sum(primary.size for primary, _ in clusters.values()),
        },
        "decision_clusters": decision_clusters,
        **{
            statistics_key: summary_statistics(
                [cluster[value_key] for cluster in decision_clusters]
            )
            for statistics_key, value_key in CHANNEL_FIELDS.values()
        },
    }


def compare_records(
    reference_records: Sequence[MeasurementRecord],
    candidate_records: Sequence[MeasurementRecord],
    observable_specs: Mapping[str, ObservableSpec],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Pair one complete profile with FP64 and build compact observable summaries.
    """
    reference = {record.key: record for record in reference_records}
    candidate = {record.key: record for record in candidate_records}
    if len(reference) != len(reference_records) or len(candidate) != len(
        candidate_records
    ):
        raise RegressionSchemaError("comparison record keys must be unique")
    missing = reference.keys() - candidate.keys()
    extra = candidate.keys() - reference.keys()
    if missing or extra:
        raise RegressionSchemaError(
            "candidate and FP64 populations differ: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    paired_case_count = len(
        {(record.case_id, record.transform_id) for record in reference_records}
    )
    if identity["case_count"] != paired_case_count:
        raise RegressionSchemaError(
            "comparison case count differs from the paired population"
        )

    # Each cluster collects its primary and absolute differences in key order.
    clusters: dict[str, dict[str, tuple[list, list]]] = defaultdict(
        lambda: defaultdict(lambda: ([], []))
    )
    cases: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key in sorted(reference):
        reference_record = reference[key]
        candidate_record = candidate[key]
        observable = reference_record.observable
        if candidate_record.values.shape != reference_record.values.shape:
            raise RegressionSchemaError(f"comparison shape differs for {key}")
        primary, absolute = _primary_difference(
            reference_record.values,
            candidate_record.values,
            observable_specs[observable],
        )
        primary_parts, absolute_parts = clusters[observable][
            reference_record.cluster_id
        ]
        primary_parts.append(primary.reshape(-1))
        absolute_parts.append(absolute.reshape(-1))
        cases[observable].add((reference_record.case_id, reference_record.transform_id))

    summaries = {
        observable: _observable_summary(
            observable_specs[observable],
            {
                cluster_id: (np.concatenate(primary), np.concatenate(absolute))
                for cluster_id, (primary, absolute) in sorted(
                    observable_clusters.items()
                )
            },
            case_count=len(cases[observable]),
        )
        for observable, observable_clusters in sorted(clusters.items())
    }
    return {
        "identity": dict(identity),
        "completion": {
            "requested_cases": paired_case_count,
            "completed_cases": paired_case_count,
        },
        "observables": summaries,
    }


def summarize_run(
    records_dir: str | Path,
    reference_dir: str | Path,
    suite: Suite,
) -> dict[str, Any]:
    """
    Pair one run's saved records with its FP64 reference and summarize them.

    GPU jobs only save full-resolution records. Pairing happens here, when a
    summary is needed, so runs never wait for or read the reference. Values
    the suite computes across several cases are summarized beside the paired
    observables.

    Args:
        records_dir: Records directory written by one profile run.
        reference_dir: FP64 reference records for the same suite.
        suite: The suite whose observable contracts and group values apply.

    Returns:
        The comparison summary, whose identity also names the reference.
    """
    manifest, records = load_reference(records_dir)
    reference_manifest, reference_records = load_reference(reference_dir)
    identity = manifest["identity"]
    reference_identity = reference_manifest["identity"]
    if identity.get("run_id") == reference_identity.get("run_id"):
        raise RegressionSchemaError(
            "a compared run must be distinct from its FP64 reference"
        )
    for field in ("plan_hash", "suite_contract_hash"):
        if identity.get(field) != reference_identity.get(field):
            raise RegressionSchemaError(f"run {field} differs from the FP64 reference")
    if identity.get("scope") == "smoke":
        selected = {
            (case["case_id"], case["transform_id"])
            for case in identity["selected_cases"]
        }
        reference_records = [
            record
            for record in reference_records
            if (record.case_id, record.transform_id) in selected
        ]
    summary = compare_records(
        reference_records,
        records,
        suite.observables,
        identity={
            **identity,
            "reference_run_id": reference_identity["run_id"],
            "reference_hash": reference_manifest["reference_hash"],
        },
    )
    for observable, values in suite.group_values(reference_records, records).items():
        summary["observables"][observable] = _observable_summary(
            suite.observables[observable],
            {
                cluster_id: (np.array([value.value]), np.array([value.value]))
                for cluster_id, value in sorted(values.items())
            },
            case_count=sum(value.case_count for value in values.values()),
        )
    return summary
