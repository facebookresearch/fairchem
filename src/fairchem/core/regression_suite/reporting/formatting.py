"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.spec.schema import (
    PRECISION_TIERS,
    SUMMARY_STATISTICS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3e}"


def display_name(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def tier_flags(value: Mapping[str, bool] | None) -> str:
    if value is None:
        return "-"
    return " ".join(
        f"{tier}={'yes' if value[tier] else 'no'}" for tier in PRECISION_TIERS
    )


def limiting_text(limiting: Mapping[str, Any] | None) -> str:
    if limiting is None:
        return "-"
    return (
        f"{limiting['tier']}: "
        f"{display_name(limiting['observable'])}/"
        f"{limiting['channel']}/{limiting['statistic']} "
        f"{limiting['ratio_to_limit']:.3f}x"
    )


def coordinates_text(coordinates: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in coordinates.items())


SUITE_HEADERS = (
    "Suite",
    "Status",
    "Completion",
    "Decision population",
    "Tier results",
    "Tightest tier",
)


def suite_cells(suite: Mapping[str, Any]) -> list[str]:
    """
    Return a suite's qualification cells under ``SUITE_HEADERS``.
    """
    completion = suite["completion"]
    population = suite["population"]
    return [
        display_name(suite["suite"]),
        suite["status"],
        "-"
        if completion is None
        else f"{completion['completed_cases']}/{completion['requested_cases']}",
        "-"
        if population is None
        else f"{population['clusters']} clusters / {population['values']} values",
        tier_flags(suite["tier_pass"]),
        suite["tightest_tier"] or "-",
    ]


# Short names of the statistics a channel can be graded on.
STATISTIC_LABELS = {"median_absolute": "median", "mean_absolute": "mean", "p95": "p95"}


def channel_label(
    observable: Mapping[str, Any], channel: str, *, show_floor: bool = False
) -> str:
    """
    Name the error an observable channel measures, optionally with the floor
    of a normalized error.
    """
    comparison = observable["comparison"]
    if channel == "absolute" or comparison == "absolute":
        return f"Absolute error ({observable['units']})"
    if comparison == "relative_with_floor":
        floor = f" (floor={observable['floor']:.1e})" if show_floor else ""
        return f"Normalized error{floor}"
    return "Absolute log-ratio error"


EXECUTION_HEADERS = (
    "Source",
    "Role",
    "Profile",
    "Accelerator",
    "Compute capability",
    "CPU architecture",
    "PyTorch",
    "CUDA",
    "Suites",
)


def execution_rows(report: Mapping[str, Any]) -> list[list[Any]]:
    """
    Describe where each run of a report executed, one row per execution.
    """
    rows = []
    for execution in report["executions"]:
        environment = execution["environment"]
        capability = environment["compute_capability"]
        rows.append(
            [
                display_name(execution["source"]),
                display_name(execution["role"]),
                execution["profile"],
                environment["accelerator_name"] or environment["device_type"],
                "-" if capability is None else f"{capability[0]}.{capability[1]}",
                environment["cpu_architecture"],
                environment["torch_version"],
                environment["cuda_version"] or "-",
                len(execution["suites"]),
            ]
        )
    return rows


STATISTICS_HEADERS = (
    "Source",
    "Median",
    "Mean abs",
    "IQR",
    "p95",
    "p99",
    "Decision limits or channel pass",
)


def statistics_rows(channel: Mapping[str, Any]) -> list[list[Any]]:
    """
    Rows of a channel's summary statistics under ``STATISTICS_HEADERS``: the
    candidate, ending with the tiers its channel passes, then each main tier,
    ending with its decision limits.
    """
    statistics = channel["statistics"] or {}
    limits = channel["limits"]
    if channel["channel_pass"] is not None:
        result = tier_flags(channel["channel_pass"])
    else:
        # A channel without limits never decides, even when it runs.
        result = "REPORT ONLY" if limits is None else "-"
    return [
        [
            "Candidate",
            *(format_number(statistics.get(name)) for name in SUMMARY_STATISTICS),
            result,
        ],
        *(
            [
                f"Main {tier}",
                *(
                    format_number(channel["calibration"][tier][name])
                    for name in SUMMARY_STATISTICS
                ),
                "-"
                if limits is None
                else ", ".join(
                    f"{STATISTIC_LABELS[name]} {format_number(limit)}"
                    for name, limit in limits[tier].items()
                ),
            ]
            for tier in PRECISION_TIERS
        ),
    ]


def performance_statistics(values: Mapping[str, Any]) -> str:
    return "/".join(
        format_number(values[field]) for field in ("median", "iqr", "range")
    )


PERFORMANCE_HEADERS = [
    "Metric",
    "Group",
    "vs main",
    "Candidate median/IQR/range",
    "Main median/IQR/range",
    "Advantage median/IQR/range",
    "Better",
]


def performance_rows(performance: Mapping[str, Any]) -> list[list[str]]:
    """
    Return one row per performance metric group and main precision.
    """
    return [
        [
            f"{display_name(metric_name)} ({metric['units']})",
            coordinates_text(group["coordinates"]),
            tier,
            performance_statistics(group["candidate_raw"]),
            performance_statistics(group["baseline_raw"]),
            performance_statistics(group["advantage_factor"]),
            f"{group['headline_percent']:.1f}%",
        ]
        for tier in PRECISION_TIERS
        for metric_name, metric in sorted(
            performance["baselines"][tier]["metrics"].items()
        )
        for group in sorted(
            metric["groups"], key=lambda item: sorted(item["coordinates"].items())
        )
    ]


# Performance is summarized against main FP32 eager, so a TF32 candidate's
# speedup includes what TF32 alone buys.
PERFORMANCE_REFERENCE = "FP32"


def performance_highlights(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Compare candidate performance with main FP32 eager, one row per metric and
    group. Factors above one are better: faster, more throughput, less memory.
    """
    performance = report.get("performance") or {}
    if performance.get("status") != "COMPARABLE":
        return []
    rows = []
    metrics = performance["baselines"][PERFORMANCE_REFERENCE]["metrics"]
    for name, metric in sorted(metrics.items()):
        for group in sorted(
            metric["groups"], key=lambda item: sorted(item["coordinates"].items())
        ):
            factor = group["advantage_factor"]["median"]
            rows.append(
                {
                    "metric": name,
                    "units": metric["units"],
                    "objective": metric["objective"],
                    "coordinates": group["coordinates"],
                    "candidate": group["candidate_raw"]["median"],
                    "main": group["baseline_raw"]["median"],
                    "factor": factor,
                    "better": factor > 1.0,
                }
            )
    return rows


def highlight_cells(row: Mapping[str, Any]) -> list[str]:
    """
    Return a performance highlight's metric, group, candidate, and main cells.
    """
    return [
        f"{display_name(row['metric'])} ({row['units']})",
        coordinates_text(row["coordinates"]),
        format_number(row["candidate"]),
        format_number(row["main"]),
    ]


def performance_headline(report: Mapping[str, Any]) -> str | None:
    """
    Summarize the step-time factor on the largest system in one phrase.
    """
    steps = [
        row
        for row in performance_highlights(report)
        if row["metric"] == "performance_ms_per_step"
    ]
    if not steps:
        return None
    largest = steps[-1]
    size = coordinates_text(largest["coordinates"])
    speed = "faster" if largest["better"] else "slower"
    factor = largest["factor"] if largest["better"] else 1.0 / largest["factor"]
    return f"{factor:.2f}x {speed} on the largest system ({size})"


def performance_context(performance: Mapping[str, Any]) -> str:
    environment = performance["environment"]
    protocol = performance["protocol"]
    accelerator = environment.get("accelerator_name") or environment["device_type"]
    details = [f"device={accelerator}", f"protocol={protocol['name']}"]
    details.extend(
        f"{name}={protocol[name]}"
        for name in ("warmup_steps", "measured_steps", "timestep_fs")
        if name in protocol
    )
    return ", ".join(details)


# Decision observables at or above this fraction of a limit are flagged.
NEAR_LIMIT = 0.8


def observable_rows(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """
    Return a suite's run observables, failing and near-limit ones first.
    """
    return sorted(
        (observable for observable in suite["observables"] if observable["status"]),
        key=lambda observable: (
            observable["limiting"] is None,
            -(observable["limiting"] or {}).get("ratio_to_limit", 0.0),
            observable["name"],
        ),
    )


def provenance_lines(provenance: Mapping[str, Any]) -> list[str]:
    """
    Render a baseline's provenance as a ``Baseline:`` line and indented details.
    """
    created = provenance["created_at"]
    if created:
        moment = datetime.datetime.fromisoformat(created).astimezone(datetime.UTC)
        created = f"created {moment:%Y-%m-%d %H:%M} UTC"
    lines = [
        f"Baseline: {provenance['name']} (id {provenance['baseline_id'][:8]}), "
        f"{created or 'creation time unknown'}",
        f"  {provenance['checkpoint']}, {provenance['campaign']} campaign, "
        f"{provenance['scope']} scope, {provenance['device']}",
    ]
    software = [
        f"{label} {provenance[key]}"
        for label, key in (
            ("fairchem", "fairchem_version"),
            ("torch", "torch_version"),
            ("on", "device_name"),
        )
        if key in provenance
    ]
    if software:
        lines.append(f"  {', '.join(software)}")
    lines.extend(
        f"  commit {commit[:8]}: {', '.join(suites)}"
        for commit, suites in provenance["commits"].items()
    )
    return lines
