"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core.regression_suite.artifacts.storage import load_sealed
from fairchem.core.regression_suite.spec.hashing import canonical_json, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    PerformanceMetricSpec,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


PERFORMANCE_SCHEMA = "fairchem-regression-performance"
PERFORMANCE_SCHEMA_VERSION = 1
PERFORMANCE_COMPARISON_SCHEMA = "fairchem-regression-performance-comparison"
PERFORMANCE_COMPARISON_SCHEMA_VERSION = 1
COMPARABILITY_IDENTITY_FIELDS = (
    "suite",
    "plan_hash",
    "checkpoint",
    "suite_contract_hash",
)
HARDWARE_FIELDS = (
    "device_type",
    "cpu_architecture",
    "accelerator_name",
    "compute_capability",
)
SAMPLE_ID_FIELDS = ("metric", "case_id", "transform_id", "cluster_id")


def _sample_key(sample: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sample[field] for field in SAMPLE_ID_FIELDS)


def build_performance_sidecar(
    metric_specs: Mapping[str, PerformanceMetricSpec],
    samples: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    identity: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a deterministic content-addressed sidecar of positive samples.

    Each sample names its metric, case, transform, and cluster, and holds its
    grouping ``coordinates`` and scalar ``value``.
    """
    samples = sorted(samples, key=_sample_key)
    keys = [_sample_key(sample) for sample in samples]
    if len(keys) != len(set(keys)):
        raise RegressionSchemaError("performance sample keys must be unique")
    for sample in samples:
        if not math.isfinite(sample["value"]) or sample["value"] <= 0.0:
            raise RegressionSchemaError(
                "performance sample value must be positive and finite"
            )
        missing = set(metric_specs[sample["metric"]].group_by) - set(
            sample["coordinates"]
        )
        if missing:
            raise RegressionSchemaError(
                f"sample for {sample['metric']!r} is missing coordinates: "
                f"{sorted(missing)}"
            )
    if set(metric_specs) != {sample["metric"] for sample in samples}:
        raise RegressionSchemaError("every performance metric requires samples")
    payload = {
        "schema": PERFORMANCE_SCHEMA,
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "identity": dict(identity),
        "environment": dict(environment),
        "protocol": dict(protocol),
        "metrics": {
            name: {**spec._asdict(), "group_by": list(spec.group_by)}
            for name, spec in sorted(metric_specs.items())
        },
        "samples": samples,
    }
    payload["sidecar_id"] = stable_hash(payload)
    return payload


def load_performance_sidecar(path: str | Path) -> dict[str, Any]:
    """
    Load a sidecar after checking its content hash.
    """
    return load_sealed(path, "sidecar_id", f"performance sidecar {path}")


def _sample_index(sidecar: Mapping[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {_sample_key(sample): sample for sample in sidecar["samples"]}


def _incomparability(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], label: str
) -> str | None:
    """
    Say why timings of ``baseline`` cannot be compared with the candidate's, or
    return None. Software versions alone do not prevent a comparison.
    """
    for field in COMPARABILITY_IDENTITY_FIELDS:
        if candidate["identity"].get(field) != baseline["identity"].get(field):
            return f"main {label} differs at {field}"
    for field in HARDWARE_FIELDS:
        if candidate["environment"].get(field) != baseline["environment"].get(field):
            return (
                f"main {label} ran on different hardware ({field}: "
                f"{baseline['environment'].get(field)}, candidate "
                f"{candidate['environment'].get(field)})"
            )
    for field in ("protocol", "metrics"):
        if candidate[field] != baseline[field]:
            return f"main {label} used a different {field}"
    populations = [
        {key: sample["coordinates"] for key, sample in _sample_index(run).items()}
        for run in (candidate, baseline)
    ]
    if populations[0] != populations[1]:
        return f"main {label} measured a different population"
    return None


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "median": float(median),
        "iqr": float(q75 - q25),
        "range": float(array.max()) - float(array.min()),
    }


def _metric_comparison(
    name: str,
    metric: Mapping[str, Any],
    candidate: Mapping[tuple[str, ...], Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[tuple[str, ...], Mapping[str, Any]]],
) -> dict[str, Any]:
    minimize = metric["objective"] == "minimize"
    grouped: dict[str, tuple[dict[str, Any], list[tuple[str, ...]]]] = {}
    for key, sample in candidate.items():
        if key[0] == name:
            coordinates = {
                field: sample["coordinates"][field] for field in metric["group_by"]
            }
            grouped.setdefault(canonical_json(coordinates), (coordinates, []))[
                1
            ].append(key)
    groups = []
    for group_key in sorted(grouped):
        coordinates, keys = grouped[group_key]
        keys.sort()
        pairs = [
            (candidate[key]["value"], run[key]["value"])
            for run in baseline_runs
            for key in keys
        ]
        factors = _stats(
            [main / value if minimize else value / main for value, main in pairs]
        )
        groups.append(
            {
                "coordinates": coordinates,
                "sample_count": len(keys),
                "candidate_raw": _stats([candidate[key]["value"] for key in keys]),
                "baseline_raw": _stats([main for _value, main in pairs]),
                "advantage_factor": factors,
                "headline_percent": (
                    100.0 * (1.0 - 1.0 / factors["median"])
                    if minimize
                    else 100.0 * (factors["median"] - 1.0)
                ),
            }
        )
    return {
        "units": metric["units"],
        "objective": metric["objective"],
        "percent_kind": "reduction" if minimize else "increase",
        "group_by": list(metric["group_by"]),
        "groups": groups,
    }


def compare_performance(
    candidate: Mapping[str, Any],
    baselines: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """
    Compare exact samples by named baseline, without a gate or overall score.

    Runs on different hardware, protocols, or populations are NOT COMPARABLE,
    with the first reason found.
    """
    candidate_samples = _sample_index(candidate)
    comparisons = {}
    for label, value in sorted(baselines.items()):
        runs = sorted(value, key=lambda run: run["sidecar_id"])
        for run in runs:
            reason = _incomparability(candidate, run, label)
            if reason is not None:
                return {"status": "NOT COMPARABLE", "reason": reason}
        indexed_runs = [_sample_index(run) for run in runs]
        comparisons[label] = {
            "sidecar_ids": [run["sidecar_id"] for run in runs],
            "run_count": len(runs),
            "metrics": {
                name: _metric_comparison(name, metric, candidate_samples, indexed_runs)
                for name, metric in candidate["metrics"].items()
            },
        }
    return {
        "schema": PERFORMANCE_COMPARISON_SCHEMA,
        "schema_version": PERFORMANCE_COMPARISON_SCHEMA_VERSION,
        "status": "COMPARABLE",
        "candidate_sidecar_id": candidate["sidecar_id"],
        "protocol": candidate["protocol"],
        "environment": candidate["environment"],
        "baselines": comparisons,
    }
