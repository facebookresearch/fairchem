"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime
import html
import math
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.reporting.formatting import provenance_lines
from fairchem.core.regression_suite.reporting.html_report import (
    REPORT_STYLE,
    _html_table,
)
from fairchem.core.regression_suite.reporting.text_report import format_table
from fairchem.core.regression_suite.spec.schema import CHANNEL_FIELDS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Per mode: the reference label, the verdict column, the channels shown, and
# what the result does not claim.
_MODE_TEXT = {
    "quick": (
        "Reference",
        "Verdict",
        tuple(CHANNEL_FIELDS),
        "Quick checks compare with the reference profile only and grade nothing.",
    ),
    "fast": (
        "Baseline",
        "Tier",
        ("primary",),
        "Fast checks cover only the smoke cases and do not constitute a "
        "full-suite qualification.",
    ),
}
_CHANNEL_LABELS = {"primary": "", "absolute": "Abs "}
_SUMMARY_HEADERS = ("Suite", "Status", "Wall time")


def finite_or_none(value: Any) -> Any:
    """
    Replace non-finite floats, which JSON cannot hold, with None.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_or_none(item) for item in value]
    return value


def build_check_report(
    results: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine one quick or fast check's suite results into one JSON document.

    Args:
        results: Each checked suite's result, in the order checked.
        baseline: A fast check's baseline provenance; None for a quick check.

    Returns:
        The check report: its mode, profile, reference, checkpoint, creation
        time, baseline provenance, suite results with non-finite statistics as
        null, and the command's exit code.
    """
    first = results[0]
    return {
        "mode": first["mode"],
        "profile": first["profile"],
        "reference": first["reference"],
        "checkpoint": first["checkpoint"],
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "baseline": None if baseline is None else dict(baseline),
        "suites": finite_or_none(list(results)),
        "exit_code": int(not all(result["passed"] for result in results)),
    }


def _statistics(result: Mapping[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """
    Return the headers and rows of one suite's observable statistics.
    """
    _reference, verdict_label, channels, _disclaimer = _MODE_TEXT[result["mode"]]
    columns = [
        (channel, statistic)
        for channel in channels
        for statistic in ("median_absolute", "p95")
    ]
    headers = [
        "Observable",
        *(
            _CHANNEL_LABELS[channel] + ("Median" if statistic != "p95" else "P95")
            for channel, statistic in columns
        ),
        verdict_label,
    ]
    rows = []
    for name, verdict in sorted(result["verdicts"].items()):
        observable = result["summary"]["observables"].get(name)
        values = [
            None
            if observable is None
            else observable[CHANNEL_FIELDS[channel][0]][statistic]
            for channel, statistic in columns
        ]
        rows.append([name, *values, verdict])
    return headers, rows


def _value(value: float | None) -> str:
    return "-" if value is None else f"{value:.4e}"


def _details(result: Mapping[str, Any]) -> list[str]:
    """
    Return one suite's provenance, completion, status, and wall-time lines.
    """
    reference_label = _MODE_TEXT[result["mode"]][0]
    completion = result["summary"]["completion"]
    lines = [
        f"Suite: {result['suite']}",
        f"Check: {result['mode']} (smoke cases)",
        f"Profile: {result['profile']}",
        f"{reference_label}: {result['reference']}",
        f"Checkpoint: {result['checkpoint']}",
        f"Cases: {completion['completed_cases']} / {completion['requested_cases']}",
        f"Status: {result['status']}",
    ]
    if result["qualification"] is not None:
        tier = result["qualification"]["tightest_tier"] or "BEYOND_TF32"
        lines.append(f"Tightest tier: {tier}")
    lines.append(f"Wall time: {result['elapsed_seconds']:.1f} s")
    return lines


def render_smoke_result(result: Mapping[str, Any]) -> str:
    """
    Render one suite's quick or fast result for interactive use.
    """
    headers, rows = _statistics(result)
    observable, *columns, verdict = headers
    header = "".join(f" {column:>12}" for column in columns)
    lines = [
        *_details(result),
        "",
        f"{observable:<38}{header} {verdict:>12}",
        "-" * (51 + len(header)),
    ]
    for name, *values, verdict in rows:
        cells = "".join(f" {_value(value):>12}" for value in values)
        lines.append(f"{name:<38}{cells} {verdict:>12}")
    disclaimer = _MODE_TEXT[result["mode"]][3]
    lines.extend(("", f"Artifact: {result['records_path']}", disclaimer))
    return "\n".join(lines)


def render_check_summary(report: Mapping[str, Any]) -> str:
    """
    Render a quick or fast check report as one table, one row per suite and
    observable, followed by the total wall time.
    """
    suites = report["suites"]
    headers = []
    rows = []
    for suite in suites:
        headers, statistics = _statistics(suite)
        prefix = [suite["suite"], suite["status"]]
        rows.extend(
            [*prefix, name, *(_value(value) for value in values), verdict]
            for name, *values, verdict in statistics
        )
        if not statistics:
            rows.append([*prefix, *["-"] * (len(headers) - 1), "-"])
    total = sum(suite["elapsed_seconds"] for suite in suites)
    return "\n".join(
        [
            format_table(["Suite", "Status", *headers], rows),
            f"Total wall time: {total:.1f} s",
        ]
    )


def _status_cell(suite: Mapping[str, Any]) -> tuple[str, str]:
    return suite["status"], f"status status-{'pass' if suite['passed'] else 'fail'}"


def render_check_html(report: Mapping[str, Any]) -> str:
    """
    Render a quick or fast check report as a self-contained HTML page.
    """
    mode = report["mode"]
    reference_label, _verdict, _channels, disclaimer = _MODE_TEXT[mode]
    identity = [
        f"Profile: {report['profile']}",
        f"{reference_label}: {report['reference']}",
        f"Checkpoint: {report['checkpoint']}",
        f"Created: {report['created_at']}",
    ]
    if report["baseline"] is not None:
        identity.extend(provenance_lines(report["baseline"]))
    suites = report["suites"]
    summary = _html_table(
        _SUMMARY_HEADERS,
        [
            [
                suite["suite"],
                _status_cell(suite),
                f"{suite['elapsed_seconds']:.1f} s",
            ]
            for suite in suites
        ],
        container="table-scroll compact",
    )
    sections = []
    for suite in suites:
        headers, rows = _statistics(suite)
        table = _html_table(
            headers,
            [
                [name, *(_value(value) for value in values), verdict]
                for name, *values, verdict in rows
            ],
        )
        details = "".join(
            f'<p class="identity">{html.escape(line)}</p>'
            for line in [*_details(suite), f"Artifact: {suite['records_path']}"]
        )
        sections.append(
            f"<details open><summary>{html.escape(suite['suite'])} — "
            f"{html.escape(suite['status'])}</summary>"
            f'<div class="suite-body">{details}{table}</div></details>'
        )
    title = f"Regression {mode} check"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n<style>{REPORT_STYLE}</style>\n"
        f"</head>\n<body>\n<h1>{html.escape(title)}</h1>\n"
        + "".join(f'<p class="identity">{html.escape(line)}</p>' for line in identity)
        + f"<h2>Suites</h2>{summary}"
        + f'<p class="suite-note">{html.escape(disclaimer)}</p>'
        + "".join(sections)
        + "\n</body>\n</html>\n"
    )
