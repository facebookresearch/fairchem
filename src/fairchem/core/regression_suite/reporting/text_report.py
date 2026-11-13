"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.reporting.formatting import (
    EXECUTION_HEADERS,
    NEAR_LIMIT,
    PERFORMANCE_HEADERS,
    STATISTIC_LABELS,
    STATISTICS_HEADERS,
    SUITE_HEADERS,
    channel_label,
    display_name,
    execution_rows,
    highlight_cells,
    limiting_text,
    observable_rows,
    performance_context,
    performance_headline,
    performance_highlights,
    performance_rows,
    provenance_lines,
    statistics_rows,
    suite_cells,
    tier_flags,
)
from fairchem.core.regression_suite.spec.schema import PRECISION_TIERS

if TYPE_CHECKING:
    from collections.abc import Mapping


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    """
    Render rows as a plain-text table with left-aligned columns.
    """
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in text_rows)])


def _observable_table(suite: Mapping[str, Any]) -> str:
    blocks = []
    for observable in suite["observables"]:
        population = observable["population"]
        population_label = (
            "candidate not run"
            if population is None
            else (
                f"{population['cases']} cases, {population['clusters']} clusters, "
                f"{population['values']} scalar values"
            )
        )
        if not observable["report_only"] and observable["tier_pass"] is not None:
            selected = " ".join(
                f"{tier}={observable['selected_channel'][tier]}"
                for tier in PRECISION_TIERS
            )
            blocks.append(
                f"Qualification: {tier_flags(observable['tier_pass'])}; "
                f"selected {selected}"
            )
        for channel in observable["channels"]:
            blocks.extend(
                [
                    (
                        f"{display_name(observable['name'])} — "
                        f"{channel_label(observable, channel['name'], show_floor=True)}"
                    ),
                    f"Population: {population_label}",
                ]
            )
            table = format_table(list(STATISTICS_HEADERS), statistics_rows(channel))
            blocks.extend([table, ""])
    return "\n".join(blocks).rstrip()


def _observable_summary_table(report: Mapping[str, Any]) -> str:
    rows = []
    for suite in report["suites"]:
        for observable in observable_rows(suite):
            limiting = observable["limiting"]
            ratio = limiting["ratio_to_limit"] if limiting else None
            rows.append(
                [
                    display_name(suite["suite"]),
                    f"{display_name(observable['name'])} ({observable['units']})",
                    observable["status"],
                    "-"
                    if limiting is None
                    else f"{limiting['tier']}: {limiting['channel']}/"
                    f"{limiting['statistic']}",
                    "-" if ratio is None else f"{ratio:.3f}x",
                    "!" if ratio is not None and ratio >= NEAR_LIMIT else "",
                ]
            )
    return format_table(
        ["Suite", "Observable", "Status", "Limiting check", "Ratio", ""], rows
    )


def _worst_cluster_lines(report: Mapping[str, Any], count: int = 5) -> list[str]:
    lines = []
    for suite in report["suites"]:
        for observable in observable_rows(suite):
            limiting = observable["limiting"]
            if limiting is None or limiting["ratio_to_limit"] < NEAR_LIMIT:
                continue
            lines.append(
                f"{display_name(suite['suite'])} / "
                f"{display_name(observable['name'])} (cluster mean / "
                f"{limiting['tier']} {limiting['channel']} "
                f"{STATISTIC_LABELS[observable['cluster_statistic']]} limit):"
            )
            lines.extend(
                f"  {row['cluster_id']}: {row['ratio']:.3f}x"
                for row in observable["clusters"][:count]
            )
    return lines


def _performance_highlight_table(report: Mapping[str, Any]) -> str:
    return format_table(
        ["Metric", "Group", "Candidate", "Main FP32", "Factor", "Better"],
        [
            [
                *highlight_cells(row),
                f"{row['factor']:.2f}x",
                "yes" if row["better"] else "no",
            ]
            for row in performance_highlights(report)
        ],
    )


def _overview_lines(report: Mapping[str, Any]) -> list[str]:
    """
    Return the suite, observable, and performance overviews.
    """
    suite_rows = [
        [
            *suite_cells(suite),
            "profile limitation" if suite["note"] else limiting_text(suite["limiting"]),
        ]
        for suite in report["suites"]
    ]
    lines = [
        "",
        "Suite qualification",
        format_table(
            [*SUITE_HEADERS, "Limiting check (observed/limit; <=1 passes)"],
            suite_rows,
        ),
    ]
    notes = [
        f"{display_name(suite['suite'])}: {suite['note']}"
        for suite in report["suites"]
        if suite.get("note")
    ]
    if notes:
        lines.extend(["", "Suite notes", *notes])
    lines.extend(
        [
            "",
            f"Observables (! marks a ratio at or above {NEAR_LIMIT:g} of its limit)",
            _observable_summary_table(report),
        ]
    )
    worst = _worst_cluster_lines(report)
    if worst:
        lines.extend(
            ["", "Worst clusters of failing or near-limit observables", *worst]
        )
    if performance_highlights(report):
        lines.extend(
            [
                "",
                f"Inference performance vs main FP32 eager: {performance_headline(report)}",
                "Factors above 1 are better: faster, more throughput, less memory.",
                _performance_highlight_table(report),
            ]
        )
    return lines


def _detail_lines(report: Mapping[str, Any]) -> list[str]:
    """
    Return every observable's statistics and the raw inference performance.
    """
    lines = []
    for suite in report["suites"]:
        if not suite["observables"]:
            continue
        lines.extend(
            [
                "",
                f"{display_name(suite['suite'])} observable statistics",
                "Statistics are over per-cluster mean differences. Calibration "
                "rows show median/mean absolute/IQR/p95/p99; the last column "
                "shows each tier's decision limits.",
                _observable_table(suite),
            ]
        )
    if "performance" not in report:
        return lines
    performance = report["performance"]
    lines.extend(["", "Raw inference performance"])
    if performance["status"] != "COMPARABLE":
        lines.append(f"{performance['status']}: {performance['reason']}.")
    else:
        lines.extend(
            [
                "Each statistics cell is median/IQR/range; factors above 1 are better.",
                performance_context(performance),
                format_table(PERFORMANCE_HEADERS, performance_rows(performance)),
            ]
        )
    return lines


def render_candidate_report(report: Mapping[str, Any], *, summary: bool = False) -> str:
    """
    Render a candidate report as deterministic plain-text tables.

    The detailed statistics come first and the suite, observable, and
    performance overviews last, so a terminal ends on the verdicts. ``summary``
    leaves out the detailed statistics.
    """
    candidate = report["candidate"]
    lines = [
        f"Candidate: {candidate['profile']} @ {candidate['git_commit']}",
        *provenance_lines(report["baseline"]),
        "",
        "Execution environments",
        format_table(list(EXECUTION_HEADERS), execution_rows(report)),
    ]
    if not summary:
        lines.extend(_detail_lines(report))
    lines.extend(_overview_lines(report))
    return "\n".join(lines)
