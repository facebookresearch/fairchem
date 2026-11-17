"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from fairchem.core.regression_suite.artifacts.storage import (
    staged_directory,
    write_json,
)
from fairchem.core.regression_suite.evaluation.decision import tightest_tier
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
    format_number,
    highlight_cells,
    limiting_text,
    observable_rows,
    performance_headline,
    performance_highlights,
    performance_rows,
    provenance_lines,
    statistics_rows,
    suite_cells,
)
from fairchem.core.regression_suite.reporting.report import FEW_CLUSTERS
from fairchem.core.regression_suite.spec.schema import (
    CANDIDATE_REPORT_SCHEMA,
    PRECISION_TIERS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_BASELINE_STYLE = {
    "FP64": "-",
    "FP32": "--",
    "TF32": ":",
}
_CANDIDATE_COLOR = "#e87500"
_MEDIAN_COLOR = "#c62828"
# Trajectories are too dense for line styles, so each tier gets a hue.
_TRAJECTORY_COLORS = {
    "FP64": "#2a78d6",
    "FP32": "#1baf7a",
    "TF32": "#6250d6",
    "candidate": _CANDIDATE_COLOR,
}
# Exact reruns can match the reference to zero, which a log axis cannot show.
_GAP_FLOOR = 1e-12


def _slug(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()
    return normalized or "item"


def _plot_gates(system: Mapping[str, Any], *, candidate: str, status: str) -> bytes:
    """
    Draw one NVE system's three gate plots: the candidate's seeds and median
    beside the medians of the FP64, FP32, and TF32 calibration runs.
    """
    figure = Figure(figsize=(17.0, 4.6), dpi=110, constrained_layout=True)
    FigureCanvasAgg(figure)
    gap_axes, drift_axes, spread_axes = figure.subplots(1, 3)
    curves = system["curves"]
    seeds = system["candidate_seeds"]
    panels = (
        (gap_axes, "fp64_gap", system["gap_times_ps"]),
        (drift_axes, "drift", system["drift_times_ps"]),
        (spread_axes, "spread", system["drift_times_ps"]),
    )
    for axes, gate, times in panels:
        for values in (seeds or {}).get(gate, []):
            axes.plot(
                times,
                [max(value, _GAP_FLOOR) for value in values]
                if gate == "fp64_gap"
                else values,
                color=_CANDIDATE_COLOR,
                linewidth=0.8,
                alpha=0.5,
            )
        for label in (*PRECISION_TIERS, "candidate"):
            if label not in curves:
                continue
            values = curves[label][gate]
            axes.plot(
                times,
                [max(value, _GAP_FLOOR) for value in values]
                if gate == "fp64_gap"
                else values,
                color=_TRAJECTORY_COLORS[label],
                linewidth=2.4 if label == "candidate" else 1.6,
                linestyle="-" if label == "candidate" else "--",
                label=f"{candidate} (median of seeds)"
                if label == "candidate"
                else f"Main {label} eager",
            )
        if "candidate" not in curves:
            axes.text(
                0.5,
                0.5,
                f"Candidate {status}",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#6b7280",
            )
        axes.set_xlabel("time (ps)")
        axes.grid(color="#d9dce1", linewidth=0.6, alpha=0.8)
        axes.spines[["top", "right"]].set_visible(False)
    gap_axes.set_yscale("log")
    gap_axes.set_ylim(bottom=_GAP_FLOOR)
    gap_axes.set_ylabel("meV/atom (log)")
    gap_axes.set_title(
        f"1. |run - FP64 run|, same seed ({system['gap_frames']}-frame average)",
        loc="left",
        fontsize=10,
    )
    drift_axes.set_ylabel("meV/atom")
    drift_axes.set_title(
        "2. |energy - energy in the first window| "
        f"({system['drift_frames'] * system['frame_ps']:g} ps average)",
        loc="left",
        fontsize=10,
    )
    spread_axes.set_ylabel("(meV/atom)^2")
    spread_axes.set_title(
        "3. Variance of gate 2's curves across seeds", loc="left", fontsize=10
    )
    gap_axes.legend(fontsize=7, loc="lower right")
    figure.suptitle(
        f"{system['name']} · {system['dataset']}, {system['timestep_fs']:g} fs/step, "
        f"{system['temperature_k']:g} K, {system['steps']:,} steps, "
        f"{system['seeds']} seeds",
        fontsize=11,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


_GATE_TITLES = {
    "fp64_gap": "1. Matches FP64",
    "drift": "2. Stays near its start",
    "spread": "3. Seeds stay together",
}


def _gate_result(observable: Mapping[str, Any]) -> tuple[str, str]:
    """
    Return a gate's result, naming every precision that shares its limit,
    and the precision pairs whose limits it keeps apart.
    """
    limits = {
        tier: next(iter(observable["channels"][0]["limits"][tier].values()))
        for tier in PRECISION_TIERS
    }
    pairs = list(zip(PRECISION_TIERS, PRECISION_TIERS[1:]))
    resolved = [f"{a} from {b}" for a, b in pairs if limits[a] < limits[b]]
    separates = ", ".join(resolved) or "none: pass/fail only"
    status = observable["status"]
    limiting = observable["limiting"]
    if status is None or limiting is None:
        return str(observable["status"] or "-"), separates
    if status == "FAIL":
        return f"FAIL ({limiting['ratio_to_limit']:.2f}x TF32 limit)", separates
    shared = [t for t in PRECISION_TIERS if limits[t] == limits[limiting["tier"]]]
    label = (
        "PASS (no precision distinction)"
        if len(shared) == len(PRECISION_TIERS)
        else f"PASS@{'/'.join(shared)}"
        if len(shared) > 1
        else status
    )
    return f"{label} ({limiting['ratio_to_limit']:.2f}x limit)", separates


def _gate_table(suite: Mapping[str, Any], system: Mapping[str, Any]) -> str:
    observables = {
        observable["name"]: observable for observable in suite["observables"]
    }
    rows = []
    for gate, name in system["observables"].items():
        observable = observables[name]
        channel = observable["channels"][0]
        statistic = next(iter(channel["limits"]["FP64"]))
        candidate = (
            None if channel["statistics"] is None else channel["statistics"][statistic]
        )
        result, separates = _gate_result(observable)
        rows.append(
            [
                _GATE_TITLES[gate],
                format_number(candidate),
                *(
                    format_number(channel["limits"][tier][statistic])
                    for tier in PRECISION_TIERS
                ),
                result,
                separates,
            ]
        )
    return _html_table(
        [
            "Gate",
            "Candidate",
            "FP64 limit",
            "FP32 limit",
            "TF32 limit",
            "Result",
            "Tells apart",
        ],
        rows,
    )


# How to read every suite's results, in plain language, shown once above them.
_READING_GUIDE = (
    (
        "What is compared",
        "The candidate runs every case, and each result is compared with an "
        "FP64 reference run of the same case: the same structures, rotations, "
        "and seeds. Each comparison measures how far the candidate's numbers "
        "are from FP64's.",
    ),
    (
        "Where the limits come from",
        "The baseline ran every case again at FP64, FP32, and TF32 with the "
        "main code. How far each precision typically lands from the FP64 "
        "reference, times a safety factor, sets that precision's limit. The "
        "factor is 2 unless a suite says otherwise. Limits only get looser from "
        "FP64 to FP32 to TF32.",
    ),
    (
        "What a result means",
        "PASS means the candidate is as close to FP64 as an FP64 rerun. "
        "PASS@FP32 means as close as an FP32 run, and PASS@TF32 as close as a "
        "TF32 run. FAIL means further than a TF32 run. UNSUPPORTED means the "
        "candidate's settings cannot run that suite.",
    ),
    (
        "Clusters and ratios",
        "Results are grouped into clusters, usually one structure in one "
        "orientation, and each cluster's differences are averaged into one "
        "number. A suite is graded on a summary of those numbers, usually "
        "their median and 95th percentile. A ratio is the candidate's summary "
        f"divided by its limit: below 1 passes, and from {NEAR_LIMIT:g} it is "
        "flagged as close.",
    ),
    (
        "Charts",
        "Every chart puts the candidate in orange beside the baseline's FP64, "
        "FP32, and TF32 runs in blue, green, and purple, on a log scale where "
        "further left is closer to FP64. Many points are drawn as cumulative "
        "curves, a few named items as one row per item, and a few runs step by "
        "step with a table. Points at zero sit at the left edge. Each chart's "
        "caption says what it shows, how its gap is calculated, and how to read "
        "it.",
    ),
)


def _caption_html(lines: Sequence[tuple[str, str]]) -> str:
    """
    Render a plot's caption as short labelled lines.
    """
    return (
        '<ul class="caption">'
        + "".join(
            f"<li><strong>{html.escape(label)}:</strong> {html.escape(text)}</li>"
            for label, text in lines
        )
        + "</ul>"
    )


def _plotted(observable: Mapping[str, Any], channel_name: str) -> str:
    units = observable["units"]
    if channel_name == "absolute" or observable["comparison"] == "absolute":
        return f"|candidate - FP64|, in {units}"
    if observable["comparison"] == "log_ratio":
        return "|ln(candidate / FP64)|, about the relative difference"
    floor = f"{observable['floor']:g}".replace("e-0", "e-").replace("e+0", "e+")
    return (
        f"|candidate - FP64| / |FP64|, with FP64 values under {floor} {units} "
        f"counted as {floor}"
    )


def _chart_caption(observable: Mapping[str, Any], kind: str) -> str:
    """
    Say what a chart's observable shows, what one point is, how its gap to
    FP64 was calculated, and how to read the chart.
    """
    description = observable["description"] or {"shows": "", "point": "one cluster"}
    point = description["point"]
    population = observable["population"]
    if population is not None:
        point += f", {population['clusters']:,} in this run"
        per_point = population["values"] / population["clusters"]
        if per_point > 1:
            point += f", each averaging about {per_point:,.0f} values"
    lines = [("Each point", f"{point}.")]
    systems = _systems_line(observable)
    if systems is not None:
        lines.insert(0, ("Systems", systems))
    if description["shows"]:
        lines.insert(0, ("Shows", f"{description['shows']}."))
    channels = [str(channel["name"]) for channel in observable["channels"]]
    shown = channels[:1] if kind in ("runs", "table") else channels
    for side, channel in zip(("Left", "Right") if len(shown) > 1 else ("",), shown):
        label = f"Plotted, {side.lower()}" if side else "Plotted"
        lines.append(
            (label, f"{_plotted(observable, channel)}. Lower is closer to FP64.")
        )
    lines.append(("Result", _result_line(observable)))
    note = _few_points_note(observable)
    if note is not None:
        lines.append(("Note", note))
    read = "items-report" if kind == "items" and observable["report_only"] else kind
    lines.append(("Read", _HOW_TO_READ[read]))
    return _caption_html(lines)


def _gate_caption(system: Mapping[str, Any]) -> str:
    """
    Say what an NVE system is and how each of its three gate curves was
    calculated.
    """
    kind = "an MD22 molecule" if system["dataset"] == "md22" else "a TM23 metal melt"
    run_ps = system["steps"] * system["timestep_fs"] / 1000.0
    gap_ps = system["gap_frames"] * system["frame_ps"]
    drift_ps = system["drift_frames"] * system["frame_ps"]
    return _caption_html(
        [
            (
                "System",
                f"{system['name']}, {kind} with {system['atoms']} atoms. "
                f"{system['seeds']} seeds start at {system['temperature_k']:g} K "
                f"and run {system['steps']:,} steps of "
                f"{system['timestep_fs']:g} fs ({run_ps:g} ps). Energy per atom "
                f"is recorded every {system['sample_interval']} steps.",
            ),
            (
                "Plot 1",
                f"|candidate - FP64| for the same seed, averaged over {gap_ps:g} "
                "ps, on a log scale.",
            ),
            (
                "Plot 2",
                f"how far each seed's energy, averaged over {drift_ps:g} ps, has "
                "moved from its first averaged value.",
            ),
            ("Plot 3", "how much plot 2's curves vary across seeds (variance)."),
            (
                "Lines",
                "thin are single seeds, thick is the median over seeds, dashed are "
                "the baseline. Each gate number is its thick line's average.",
            ),
        ]
    )


def _explanation_html(paragraphs: Sequence[Sequence[str]]) -> str:
    """
    Render (heading, paragraph) pairs as a plain-language explanation.
    """
    if not paragraphs:
        return ""
    return (
        '<div class="explanation">'
        + "".join(
            f"<p><strong>{html.escape(heading)}.</strong> {html.escape(text)}</p>"
            for heading, text in paragraphs
        )
        + "</div>"
    )


# Observables with at most this many named items, such as structures or
# crystals, are drawn item by item; more items are drawn as cumulative curves.
_FEW_ITEMS = 12
_TIER_MARKERS = {"FP64": "v", "FP32": "s", "TF32": "D"}


def _item_labels(observable: Mapping[str, Any]) -> list[str]:
    """
    Return the item each of an observable's clusters belongs to, in order.
    """
    if observable.get("items") is None:
        return [c.split("::")[0].split("/", 1)[-1] for c in observable["cluster_ids"]]
    return [observable["systems"][key]["label"] for key in observable["items"]]


def _report_only_reason(observable: Mapping[str, Any]) -> str:
    description = observable.get("description") or {}
    return description.get("report_only_reason", "")


def _report_only_label(observable: Mapping[str, Any]) -> str:
    reason = _report_only_reason(observable)
    return f"report-only ({reason})" if reason else "report-only"


def _same_composition(systems: Sequence[Mapping[str, Any]]) -> bool:
    return len(systems) > 1 and len({(s["formula"], s["atoms"]) for s in systems}) == 1


def _few_points_note(observable: Mapping[str, Any]) -> str | None:
    """
    Say how few points make the graded median and 95th percentile, if few.
    """
    population = observable["population"]
    if observable["report_only"] or population is None:
        return None
    count = population["clusters"]
    if count == 1:
        return "only one point, so the median and 95th percentile are both its value."
    # From 21 points on, the 95th percentile is at or below the second worst.
    if count >= 21:
        return None
    median = (
        "the median is the middle point"
        if count % 2
        else "the median is halfway between the two middle points"
    )
    return (
        f"only {count} points, so {median} and the 95th percentile sits between "
        "the two worst points."
    )


def _systems_line(observable: Mapping[str, Any]) -> str | None:
    systems = list((observable.get("systems") or {}).values())
    if not systems:
        return None
    if _same_composition(systems):
        first = systems[0]
        runs = "; ".join(system["label"] for system in systems)
        return (
            f"{first['formula']}, {first['atoms']} atoms, in {len(systems)} runs: "
            f"{runs}."
        )
    if len(systems) <= 6:
        return (
            "; ".join(
                f"{system['label']} ({system['formula']}, {system['atoms']} atoms)"
                for system in systems
            )
            + "."
        )
    sizes = sorted(system["atoms"] for system in systems)
    rows = "one row each" if len(systems) <= _FEW_ITEMS else "too many to list"
    return f"{len(systems)} structures of {sizes[0]:,} to {sizes[-1]:,} atoms, {rows}."


def _systems_title(observable: Mapping[str, Any]) -> str:
    systems = list((observable.get("systems") or {}).values())
    if len(systems) == 1:
        return f"{systems[0]['formula']}, {systems[0]['atoms']} atoms"
    if _same_composition(systems):
        first = systems[0]
        return f"{len(systems)} runs of {first['formula']}, {first['atoms']} atoms"
    return f"{len(systems)} systems" if systems else ""


def _log_violin(
    axes: Any, values: Sequence[float], *, row: float, color: str, minimum: float
) -> None:
    """
    Shade the spread of a row's points as a density on the log scale.
    """
    logs = np.log10(np.maximum(np.asarray(values, dtype=np.float64), minimum))
    if logs.size < 3 or np.ptp(logs) == 0.0:
        return
    bandwidth = 1.06 * logs.std() * logs.size ** (-0.2)
    grid = np.linspace(logs.min() - 2 * bandwidth, logs.max() + 2 * bandwidth, 120)
    density = np.exp(-0.5 * ((grid[:, None] - logs[None, :]) / bandwidth) ** 2).sum(1)
    half = 0.085 * density / density.max()
    axes.fill_between(
        10.0**grid, row - half, row + half, color=color, alpha=0.18, linewidth=0
    )


_STATISTIC_LEVELS = {"median_absolute": 50.0, "p95": 95.0}
_PRIMARY_PANELS = {
    "relative_with_floor": "normalized",
    "log_ratio": "log-ratio",
    "absolute": "plain-gap",
}


def _panel_name(observable: Mapping[str, Any], channel_name: str) -> str:
    if channel_name == "absolute":
        return "plain-gap"
    return _PRIMARY_PANELS[observable["comparison"]]


def _panel_title(
    observable: Mapping[str, Any], channel: Mapping[str, Any]
) -> str | None:
    """
    Name a chart panel and the strictest precision whose limits the candidate
    meets on it.
    """
    if channel["channel_pass"] is None:
        return None
    tier = tightest_tier(channel["channel_pass"])
    verdict = "outside every limit" if tier is None else f"within {tier} limits"
    if len(observable["channels"]) == 1:
        return f"Candidate {verdict}"
    name = _panel_name(observable, str(channel["name"])).capitalize()
    return f"{name} panel: candidate {verdict}"


def _times(ratio: float) -> str:
    return f"{ratio:.2f}x" if ratio < 10 else f"{ratio:,.0f}x"


def _result_line(observable: Mapping[str, Any]) -> str:
    """
    Say how the observable was graded, with the ratios that decided it.
    """
    if observable["report_only"]:
        reason = _report_only_reason(observable)
        return f"report-only, not graded: {reason}." if reason else "report-only."
    limiting = observable["limiting"]
    if observable["status"] is None or limiting is None:
        return "not run by this candidate."
    ratios = {
        str(channel["name"]): channel["limit_ratios"]
        for channel in observable["channels"]
        if channel["limit_ratios"] is not None
    }
    tier, channel = limiting["tier"], limiting["channel"]
    where = (
        f" on the {_panel_name(observable, channel)} panel" if len(ratios) > 1 else ""
    )
    decided = " and ".join(
        f"{STATISTIC_LABELS[statistic]} {_times(ratio)}"
        for statistic, ratio in ratios[channel][tier].items()
    )
    text = f"{observable['status']}, {decided} of the {tier} limits{where}."
    index = PRECISION_TIERS.index(tier)
    if observable["status"] != "FAIL" and index > 0:
        tighter = PRECISION_TIERS[index - 1]
        best = min(ratios.values(), key=lambda r: max(r[tighter].values()))[tighter]
        text += (
            f" {tighter} fails, with "
            + " and ".join(
                f"{STATISTIC_LABELS[statistic]} {_times(ratio)}"
                for statistic, ratio in best.items()
            )
            + " of its limits."
        )
    if len(ratios) > 1:
        text += (
            " A precision passes when its graded statistics are within its limits "
            "on either panel."
        )
    return text


def _display_axis(channel: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a channel's frozen axis widened by whole decades to include every
    precision's limits, which come from the baseline, not the candidate.
    """
    axis = dict(channel["plot"]["axis"])
    limits = [
        limit
        for tier in (channel["limits"] or {}).values()
        for limit in tier.values()
        if limit > 0
    ]
    if limits:
        lowest = 10.0 ** (math.floor(math.log10(min(limits))) - 1)
        highest = 10.0 ** (math.ceil(math.log10(max(limits))))
        axis["minimum"] = min(axis["minimum"], lowest)
        axis["maximum"] = max(axis["maximum"], highest)
    return axis


def _on_scale(limit: float, axis: Mapping[str, Any], tier: str) -> tuple[float, str]:
    """
    Keep a limit tick inside the axis, labelled as off scale when pinned.
    """
    pinned = min(max(limit, axis["minimum"]), axis["maximum"])
    return pinned, tier if pinned == limit else f"{tier} (off scale)"


def _plot_cumulative(observable: Mapping[str, Any], *, candidate: str) -> bytes:
    """
    Draw each run's points as a cumulative curve, one panel per channel, with
    the median and 95th-percentile limits marked where they are read.
    """
    channels = observable["channels"]
    figure = Figure(
        figsize=(6.4 * len(channels), 3.8), dpi=110, constrained_layout=True
    )
    FigureCanvasAgg(figure)
    for axes, channel in zip(
        figure.subplots(1, len(channels), squeeze=False)[0], channels
    ):
        axis = _display_axis(channel)
        for label in (*PRECISION_TIERS, "candidate"):
            values = [v for v in channel["points"][label] or [] if v is not None]
            if not values:
                continue
            xs = np.sort(np.maximum(values, axis["minimum"]))
            ys = 100.0 * np.arange(1, len(xs) + 1) / len(xs)
            axes.step(
                np.r_[axis["minimum"], xs],
                np.r_[0.0, ys],
                where="post",
                color=_TRAJECTORY_COLORS[label],
                linestyle="-" if label == "candidate" else _BASELINE_STYLE[label],
                linewidth=2.4 if label == "candidate" else 1.4,
                label=candidate if label == "candidate" else f"Main {label} eager",
            )
        for level, statistic in ((50.0, "median_absolute"), (95.0, "p95")):
            axes.axhline(level, color=_MEDIAN_COLOR, linewidth=1.2, zorder=1)
            for tier in PRECISION_TIERS:
                limits = channel["limits"]
                limit = None if limits is None else limits[tier].get(statistic)
                if limit is None:
                    continue
                limit, name = _on_scale(limit, axis, tier)
                axes.plot(
                    [limit, limit],
                    [level - 4, level + 4],
                    color="#3c4043",
                    linewidth=1.6,
                )
                axes.text(
                    limit, level + 5, name, ha="center", fontsize=7, color="#3c4043"
                )
        if channel["statistics"] is not None:
            for statistic, level in _STATISTIC_LEVELS.items():
                value = max(channel["statistics"][statistic], axis["minimum"])
                axes.plot(
                    [value, value],
                    [0.0, level],
                    color=_CANDIDATE_COLOR,
                    linestyle="--",
                    linewidth=1.2,
                    zorder=4,
                )
                median = statistic == "median_absolute"
                axes.text(
                    value,
                    1.5,
                    "median" if median else "p95",
                    rotation=90,
                    ha="right" if median else "left",
                    va="bottom",
                    fontsize=7,
                    color=_CANDIDATE_COLOR,
                )
                axes.scatter(
                    [value],
                    [level],
                    s=42,
                    color=_CANDIDATE_COLOR,
                    edgecolors="white",
                    zorder=5,
                )
        title = _panel_title(observable, channel)
        if title is not None:
            axes.set_title(title, loc="left", fontsize=9)
        axes.set_xscale("log")
        axes.set_xlim(axis["minimum"], axis["maximum"])
        axes.set_ylim(0, 105)
        axes.set_xlabel(channel_label(observable, str(channel["name"])))
        axes.set_ylabel("Points at or below (%)")
        axes.grid(color="#d9dce1", linewidth=0.6, alpha=0.8)
        axes.spines[["top", "right"]].set_visible(False)
    figure.legend(
        *figure.axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncols=4,
        fontsize=7,
        frameon=False,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


def _plot_items(observable: Mapping[str, Any], *, candidate: str) -> bytes:
    """
    Draw the candidate's graded median and 95th percentile against each
    precision's limits, then every point on its item's row, the candidate
    beside the baseline's runs.
    """
    channels = observable["channels"]
    # One summary row per graded statistic; each panel marks its own limits.
    summaries = list(
        dict.fromkeys(
            statistic
            for channel in channels
            for statistic in (channel["limits"] or {}).get("FP64", {})
        )
    )
    # Counts are whole numbers, often exactly zero, so they use a linear axis.
    linear = observable["units"] == "count"
    labels_by_cluster = _item_labels(observable)
    items = list(dict.fromkeys(labels_by_cluster))
    offset = len(summaries) + (0.5 if summaries else 0)
    rows = {item: index + offset for index, item in enumerate(items)}
    figure = Figure(
        figsize=(6.4 * len(channels) + 1.8, 0.42 * (len(items) + len(summaries)) + 1.7),
        dpi=110,
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, len(channels), squeeze=False, sharey=True)[0]
    for axes, channel in zip(panels, channels):
        axis = _display_axis(channel)
        for row, statistic in enumerate(summaries):
            if statistic not in (channel["limits"] or {}).get("FP64", {}):
                continue
            for tier, style in _BASELINE_STYLE.items():
                limit, name = _on_scale(channel["limits"][tier][statistic], axis, tier)
                axes.plot(
                    [limit, limit],
                    [row - 0.32, row + 0.32],
                    color="#3c4043",
                    linestyle=style,
                    linewidth=1.6,
                )
                axes.text(limit, row - 0.36, name, ha="center", va="bottom", fontsize=6)
            if channel["statistics"] is not None:
                axes.scatter(
                    [max(channel["statistics"][statistic], axis["minimum"])],
                    [row],
                    s=46,
                    color=_CANDIDATE_COLOR,
                    edgecolors="white",
                    zorder=5,
                )
        if summaries:
            axes.axhline(len(summaries) - 0.25, color="#9aa0a6", linewidth=0.8)
        for shift, label in zip(
            (-0.27, -0.09, 0.09, 0.27), (*PRECISION_TIERS, "candidate")
        ):
            values = channel["points"][label]
            if values is None:
                continue
            points = [
                (value if linear else max(value, axis["minimum"]), rows[item] + shift)
                for item, value in zip(labels_by_cluster, values)
                if value is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            for row in [] if linear else sorted(set(ys)):
                _log_violin(
                    axes,
                    [x for x, y in points if y == row],
                    row=row,
                    color=_TRAJECTORY_COLORS[label],
                    minimum=axis["minimum"],
                )
            axes.scatter(
                xs,
                ys,
                s=24 if label == "candidate" else 16,
                marker="o" if label == "candidate" else _TIER_MARKERS[label],
                color=_TRAJECTORY_COLORS[label],
                edgecolors="white",
                linewidths=0.4,
                label=candidate if label == "candidate" else f"Main {label} eager",
                zorder=3,
            )
        title = _panel_title(observable, channel)
        if title is not None:
            axes.set_title(title, loc="left", fontsize=9)
        labels = [
            f"{STATISTIC_LABELS[statistic].capitalize()} of all points"
            for statistic in summaries
        ] + items
        axes.set_yticks([*range(len(summaries)), *rows.values()])
        axes.set_yticklabels(labels, fontsize=8)
        axes.set_ylim(offset + len(items) - 0.5, -0.7)
        if linear:
            largest = max(
                (v for values in channel["points"].values() for v in values or [] if v),
                default=1.0,
            )
            axes.set_xlim(-0.25, largest + 0.5)
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axes.set_xscale("log")
            axes.set_xlim(axis["minimum"], axis["maximum"])
        axes.set_xlabel(channel_label(observable, str(channel["name"])))
        axes.grid(axis="x", color="#d9dce1", linewidth=0.6, alpha=0.8)
        axes.spines[["top", "right"]].set_visible(False)
    figure.legend(
        *panels[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncols=4,
        fontsize=7,
        frameon=False,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


def _plot_runs(observable: Mapping[str, Any], *, candidate: str) -> bytes:
    """
    Draw an observable run by run: its value at every recorded sample on top,
    and each run's gap to the FP64 reference below, on a log scale.
    """
    runs = sorted(observable["series"])
    figure = Figure(
        figsize=(5.0 * len(runs) + 0.6, 4.6),
        dpi=110,
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    grid = figure.subplots(
        2, len(runs), squeeze=False, height_ratios=(1, 2), sharex="col"
    )
    lines = (
        ("FP64 reference", _TRAJECTORY_COLORS["FP64"], "-"),
        ("FP32", _TRAJECTORY_COLORS["FP32"], "--"),
        ("TF32", _TRAJECTORY_COLORS["TF32"], "--"),
        ("candidate", _CANDIDATE_COLOR, "-"),
    )
    for column, run in enumerate(runs):
        series = observable["series"][run]
        values_axes, gap_axes = grid[0][column], grid[1][column]
        values_axes.plot(
            series["reference"], color=_TRAJECTORY_COLORS["FP64"], linewidth=1.2
        )
        for label, color, style in lines[1:]:
            gaps = series["gaps"].get(label)
            if not gaps:
                continue
            gap_axes.plot(
                np.maximum(gaps, _GAP_FLOOR),
                color=color,
                linestyle=style,
                linewidth=2.2 if label == "candidate" else 1.2,
                label=candidate if label == "candidate" else f"Main {label} eager",
            )
        index = observable["cluster_ids"].index(run)
        title = _item_labels(observable)[index]
        if observable.get("items") is not None:
            system = observable["systems"][observable["items"][index]]
            title += f" ({system['formula']}, {system['atoms']} atoms)"
        values_axes.set_title(title, loc="left", fontsize=9)
        values_axes.set_ylabel(observable["units"], fontsize=8)
        gap_axes.set_yscale("log")
        gap_axes.set_ylim(bottom=_GAP_FLOOR)
        gap_axes.set_ylabel(
            f"gap to FP64 ({channel_label(observable, 'primary')})", fontsize=8
        )
        gap_axes.set_xlabel("Recorded sample")
        for axes in (values_axes, gap_axes):
            axes.grid(color="#d9dce1", linewidth=0.6, alpha=0.8)
            axes.spines[["top", "right"]].set_visible(False)
    figure.legend(
        *grid[1][0].get_legend_handles_labels(),
        loc="outside upper center",
        ncols=3,
        fontsize=7,
        frameon=False,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


def _gap_table(observable: Mapping[str, Any]) -> str:
    """
    Tabulate each item's gap to FP64 for the candidate and the baseline's runs.
    """
    points = observable["channels"][0]["points"]
    rows = [
        [
            item,
            *(
                "-" if points[label] is None else format_number(points[label][index])
                for label in ("candidate", *PRECISION_TIERS)
            ),
        ]
        for index, item in enumerate(_item_labels(observable))
    ]
    return _html_table(
        ["Item", "Candidate gap", "FP64 run gap", "FP32 run gap", "TF32 run gap"],
        rows,
    )


def _chart_kind(observable: Mapping[str, Any], skipped: set[str]) -> str:
    """
    Pick how to show an observable from how many points and items it has.
    """
    if observable["name"] in skipped:
        return "none"
    if observable["series"]:
        return "runs"
    items = len(set(_item_labels(observable)))
    if len(observable["cluster_ids"]) <= FEW_CLUSTERS:
        return "table"
    return "items" if items <= _FEW_ITEMS else "cumulative"


_HOW_TO_READ = {
    "cumulative": "a curve further left is closer to FP64. The red lines mark "
    "50% and 95%, where the graded median and 95th percentile are read. The "
    "orange dots, with dashed lines down to the axis, are the candidate's "
    "values. A precision passes when both are left of its ticks.",
    "items": "the top rows grade the candidate. Its median and 95th percentile "
    "across all points (orange) must be left of a precision's ticks to pass. "
    "Below, each row is one item and each dot one point, with FP64 in blue, "
    "FP32 in green, TF32 in purple, and the candidate in orange. The shaded "
    "shapes show how each run's points spread.",
    "items-report": "each row is one item and each dot its gap, with FP64 in "
    "blue, FP32 in green, TF32 in purple, and the candidate in orange. It is "
    "report-only, so there are no limits. The exact gaps are in the table below.",
    "runs": "each column is one run. The top shows the FP64 reference's value "
    "at every recorded sample; the bottom shows how far the FP32 and TF32 runs "
    "and the candidate are from it at each sample. The grade uses each run's "
    "average gap, given in the table.",
    "table": "each row is one item's gap to FP64.",
}


def _limiting_values(suite: Mapping[str, Any]) -> tuple[str, str, str]:
    limiting = suite["limiting"]
    if limiting is None:
        return "-", "-", "-"
    observable = next(
        item for item in suite["observables"] if item["name"] == limiting["observable"]
    )
    channel = next(
        item for item in observable["channels"] if item["name"] == limiting["channel"]
    )
    statistic = limiting["statistic"]
    calibration_values = " / ".join(
        format_number(channel["calibration"][tier][statistic])
        for tier in PRECISION_TIERS
    )
    limit_values = (
        "-"
        if channel["limits"] is None
        else " / ".join(
            format_number(channel["limits"][tier][statistic])
            for tier in PRECISION_TIERS
        )
    )
    label = (
        f"{display_name(limiting['observable'])} / {limiting['channel']} / "
        f"{statistic}: {format_number(channel['statistics'][statistic])}"
    )
    return label, calibration_values, limit_values


def _html_cell(value: Any, class_name: str | None = None) -> str:
    attribute = "" if class_name is None else f' class="{html.escape(class_name)}"'
    return f"<td{attribute}>{html.escape(str(value))}</td>"


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    container: str = "table-scroll",
    table_id: str | None = None,
    expansions: Sequence[str] = (),
) -> str:
    """
    Render escaped cells; a ``(value, class)`` tuple cell also sets a class.

    Each nonempty ``expansions`` entry is HTML shown in a full-width row
    directly below the row with the same index.
    """
    identifier = "" if table_id is None else f' id="{table_id}"'
    body = "".join(
        "<tr>"
        + "".join(
            _html_cell(*cell) if isinstance(cell, tuple) else _html_cell(cell)
            for cell in row
        )
        + "</tr>"
        + (
            f'<tr class="entries"><td colspan="{len(headers)}">{expansion}</td></tr>'
            if expansion
            else ""
        )
        for row, expansion in zip(rows, [*expansions, *[""] * len(rows)])
    )
    return (
        f'<div class="{container}"><table{identifier}><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _ratio_cell(ratio: float | None) -> tuple[str, str]:
    if ratio is None:
        return "-", "ratio"
    level = "over" if ratio > 1.0 else "near" if ratio >= NEAR_LIMIT else "ok"
    return f"{ratio:.3f}x", f"ratio ratio-{level}"


def _limiting_channel(observable: Mapping[str, Any]) -> Mapping[str, Any]:
    name = observable["limiting"]["channel"]
    return next(item for item in observable["channels"] if item["name"] == name)


def _statistic_pair(observable: Mapping[str, Any], statistic: str) -> str:
    """
    Return a decision statistic as observed / limit in the limiting channel.
    """
    channel = _limiting_channel(observable)
    limits = channel["limits"][observable["limiting"]["tier"]]
    if statistic not in limits:
        return "-"
    return (
        f"{format_number(channel['statistics'][statistic])} / "
        f"{format_number(limits[statistic])}"
    )


def _cluster_table(observable: Mapping[str, Any]) -> str:
    limiting = observable["limiting"]
    channels = (
        list(observable["clusters"][0]["values"]) if observable["clusters"] else []
    )
    rows = [
        [
            row["cluster_id"],
            *(format_number(row["values"][channel]) for channel in channels),
            _ratio_cell(row["ratio"]),
        ]
        for row in observable["clusters"]
    ]
    ratio_header = (
        "Ratio to limit"
        if limiting is None
        else f"Ratio to {limiting['tier']} {limiting['channel']} "
        f"{STATISTIC_LABELS[observable['cluster_statistic']]} limit"
    )
    return (
        f'<details class="clusters"><summary>All {len(rows)} clusters of '
        f"{html.escape(display_name(observable['name']))}, worst first</summary>"
        + _html_table(
            ["Cluster", *(f"{channel} mean" for channel in channels), ratio_header],
            rows,
            container="table-scroll compact",
        )
        + "</details>"
    )


def _observable_details(suite: Mapping[str, Any]) -> str:
    observables = observable_rows(suite)
    if not observables:
        return ""
    # One observed / limit column per statistic any observable is graded on.
    statistics = list(
        dict.fromkeys(
            statistic
            for observable in observables
            if observable["limiting"] is not None
            for statistic in _limiting_channel(observable)["limits"][
                observable["limiting"]["tier"]
            ]
        )
    )
    rows = []
    for observable in observables:
        limiting = observable["limiting"]
        decides = limiting is not None
        rows.append(
            [
                f"{display_name(observable['name'])} ({observable['units']})",
                "decides" if decides else _report_only_label(observable),
                (observable["status"], f"status status-{_slug(observable['status'])}"),
                "-"
                if not decides
                else f"{limiting['tier']}: {limiting['channel']}/{limiting['statistic']}",
                *(
                    _statistic_pair(observable, statistic) if decides else "-"
                    for statistic in statistics
                ),
                _ratio_cell(limiting["ratio_to_limit"] if decides else None),
            ]
        )
    table = _html_table(
        [
            "Observable",
            "Role",
            "Status",
            "Limiting check",
            *(
                f"{STATISTIC_LABELS[statistic]}: observed / limit"
                for statistic in statistics
            ),
            "Ratio",
        ],
        rows,
        container="table-scroll compact",
    )
    clusters = "".join(_cluster_table(item) for item in observables if item["clusters"])
    return (
        f'<details class="suite-entries"><summary>{len(observables)} observables of '
        f"{html.escape(display_name(suite['suite']))}</summary>"
        f'<div class="suite-body">{table}{clusters}</div></details>'
    )


def _performance_highlights_html(report: Mapping[str, Any]) -> str:
    rows = performance_highlights(report)
    if not rows:
        return ""
    cells = [
        [
            *highlight_cells(row),
            (
                f"{'↑' if row['better'] else '↓'} {row['factor']:.2f}x",
                "performance-gain" if row["better"] else "performance-loss",
            ),
        ]
        for row in rows
    ]
    return (
        '<details class="suite-entries"><summary>Inference performance vs main '
        f"FP32 eager: {html.escape(performance_headline(report) or '')}</summary>"
        '<div class="suite-body"><p>Report only. Factors above one are better: '
        "faster steps, more throughput, or less memory.</p>"
        + _html_table(
            ["Metric", "Group", "Candidate", "Main FP32 eager", "Factor"],
            cells,
            container="table-scroll compact",
        )
        + "</div></details>"
    )


def _master_table(report: Mapping[str, Any]) -> str:
    rows = []
    for suite in report["suites"]:
        name, status, *cells = suite_cells(suite)
        rows.append(
            [
                name,
                (status, f"status status-{_slug(status)}"),
                *cells,
                *_limiting_values(suite),
                limiting_text(suite["limiting"]),
            ]
        )
    headers = (
        *SUITE_HEADERS,
        "Candidate limiting statistic",
        "Main calibration FP64 / FP32 / TF32",
        "Decision limits FP64 / FP32 / TF32",
        "Limiting check (observed/limit; ≤1 passes)",
    )
    table = _html_table(
        headers,
        rows,
        table_id="master-table",
        expansions=[_observable_details(suite) for suite in report["suites"]],
    )
    return table + _performance_highlights_html(report)


def _performance_html(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return ""
    if value["status"] != "COMPARABLE":
        return (
            "<h3>Raw inference performance</h3>"
            f'<p class="suite-note">{html.escape(value["status"])}: '
            f"{html.escape(value['reason'])}</p>"
        )
    return (
        "<h3>Raw inference performance</h3>"
        "<p>Each statistics cell is median/IQR/range; factors above one are "
        "better.</p>" + _html_table(PERFORMANCE_HEADERS, performance_rows(value))
    )


REPORT_STYLE = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { color: #202124; margin: 0 auto; max-width: 1500px; padding: 28px; }
h1 { margin-bottom: 6px; }
.identity { color: #5f6368; margin: 3px 0; overflow-wrap: anywhere; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 14px 0 24px; width: 100%; }
th, td { border-bottom: 1px solid #dfe1e5; padding: 8px 10px; text-align: left; }
th { background: #f5f6f7; font-weight: 650; white-space: nowrap; }
.compact table { margin: 4px 0 22px; max-width: 900px; }
.status { font-weight: 700; }
.status-pass, .status-pass-fp32, .status-pass-tf32 { color: #188038; }
.status-fail { color: #c5221f; }
.status-unsupported, .status-not-run { color: #6b7280; }
details { border: 1px solid #dfe1e5; border-radius: 7px; margin: 12px 0; }
summary { cursor: pointer; font-size: 17px; font-weight: 650; padding: 14px 16px; }
.suite-body { border-top: 1px solid #dfe1e5; padding: 8px 16px 18px; }
.suite-note { background: #f8f9fa; border-radius: 4px; color: #5f6368; padding: 8px; }
.explanation { background: #f8f9fa; border-radius: 6px; color: #3c4043; margin: 8px 0 16px; max-width: 1100px; padding: 4px 14px; }
.explanation p { line-height: 1.45; margin: 8px 0; }
.caption { color: #3c4043; font-size: 13px; line-height: 1.45; list-style: none; margin: 6px 0 10px; max-width: 1100px; padding: 0; }
figure { margin: 22px 0 30px; }
figcaption { font-size: 15px; font-weight: 650; margin-bottom: 8px; }
figure img { height: auto; max-width: 100%; width: 1400px; }
tr.entries > td { border-bottom: 1px solid #dfe1e5; padding: 0 10px 6px; }
details.suite-entries { margin: 4px 0; }
details.suite-entries summary { font-size: 13px; font-weight: 600; padding: 6px 10px; }
details details summary { font-size: 13px; font-weight: 500; }
.ratio { font-variant-numeric: tabular-nums; white-space: nowrap; }
.ratio-ok { color: #188038; }
.ratio-near { background: #fef7e0; color: #b06000; font-weight: 700; }
.ratio-over { background: #fce8e6; color: #c5221f; font-weight: 700; }
.status-report-only { color: #6b7280; }
.performance-gain { background: #e6f4ea; color: #188038; font-weight: 700; }
.performance-loss { background: #fef7e0; color: #b06000; font-weight: 700; }
"""


def _html_document(
    report: Mapping[str, Any],
    figures: Sequence[str],
) -> str:
    candidate = report["candidate"]
    title, *details = provenance_lines(report["baseline"])
    provenance = "".join(
        [
            '<p class="identity"><strong>Baseline:</strong> '
            f"{html.escape(title.removeprefix('Baseline: '))}</p>",
            *(f'<p class="identity">{html.escape(line)}</p>' for line in details),
        ]
    )
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FAIRChem regression qualification</title>
<style>__STYLE__</style>
</head>
<body>
<h1>Numerical qualification report</h1>
<p class="identity"><strong>Candidate:</strong> __CANDIDATE_PROFILE__ @ __CANDIDATE_COMMIT__</p>
__BASELINE_PROVENANCE__
<h2>Execution environments</h2>
__EXECUTION_ENVIRONMENTS__
<h2>Suite qualification</h2>
__MASTER_TABLE__
<h2>Suite details</h2>
__READING_GUIDE__
__SUITE_DETAILS__
</body>
</html>
"""
    replacements = {
        "__STYLE__": REPORT_STYLE,
        "__CANDIDATE_PROFILE__": html.escape(str(candidate["profile"])),
        "__CANDIDATE_COMMIT__": html.escape(str(candidate["git_commit"])),
        "__BASELINE_PROVENANCE__": provenance,
        "__EXECUTION_ENVIRONMENTS__": _html_table(
            EXECUTION_HEADERS, execution_rows(report)
        ),
        "__MASTER_TABLE__": _master_table(report),
        "__READING_GUIDE__": _explanation_html(_READING_GUIDE),
        "__SUITE_DETAILS__": "".join(figures),
    }
    for marker, replacement in replacements.items():
        document = document.replace(marker, replacement)
    return document


def render_candidate_html_report(
    report: Mapping[str, Any],
) -> tuple[str, dict[PurePosixPath, bytes]]:
    """
    Render a validated candidate payload into standalone HTML and PNG bytes.

    Each observable is drawn by how many points and items it has: cumulative
    curves for many items, one row per item for a few, step by step for a few
    runs, or a table. Its channels carry the axis frozen in the baseline, the
    limits, and each run's per-cluster ``points`` in ``cluster_ids`` order. A
    suite with ``gate_curves`` gets one gate figure and table per system
    instead of charts of its gate observables.

    Args:
        report: Candidate report mapping with per-cluster points and limits.

    Returns:
        The standalone HTML document and PNG files keyed by relative path.

    """
    images: dict[PurePosixPath, bytes] = {}
    suite_sections = []
    for suite in report["suites"]:
        suite_name = str(suite["suite"])
        gates = suite["gate_curves"]
        gate_figures, gate_observables = [], set()
        if gates is not None:
            gate_figures.append("<h3>Gates by system</h3>")
            for system in gates["systems"]:
                gate_observables.update(system["observables"].values())
                png = _plot_gates(
                    system,
                    candidate=str(report["candidate"]["profile"]),
                    status=str(suite["status"]),
                )
                images[
                    PurePosixPath(
                        "gates", _slug(suite_name), f"{_slug(system['name'])}.png"
                    )
                ] = png
                encoded = base64.b64encode(png).decode("ascii")
                caption = f"{system['name']} gates"
                gate_figures.append(
                    "<figure>"
                    f"<figcaption>{html.escape(caption)}</figcaption>"
                    f'<img alt="{html.escape(caption)}" '
                    f'src="data:image/png;base64,{encoded}">'
                    f"{_gate_caption(system)}"
                    f"{_gate_table(suite, system)}"
                    "</figure>"
                )
        figures = []
        skipped = set(gate_observables)
        if gates is not None:
            skipped.update(o["name"] for o in suite["observables"] if o["report_only"])
        if suite_name == "inference" and report.get("performance"):
            # Without a comparison there is no performance table to defer to.
            for baseline in report["performance"].get("baselines", {}).values():
                skipped.update(baseline.get("metrics", {}))
        for observable in suite["observables"]:
            observable_name = str(observable["name"])
            kind = _chart_kind(observable, skipped)
            if kind == "none":
                continue
            status = (
                _report_only_label(observable)
                if observable["report_only"]
                else observable["status"]
            )
            caption = " · ".join(
                part
                for part in (
                    display_name(observable_name),
                    status,
                    _systems_title(observable),
                )
                if part
            )
            body = ""
            if kind in ("cumulative", "items", "runs"):
                plot = {
                    "cumulative": _plot_cumulative,
                    "items": _plot_items,
                    "runs": _plot_runs,
                }[kind]
                png = plot(observable, candidate=str(report["candidate"]["profile"]))
                images[
                    PurePosixPath(
                        "charts", _slug(suite_name), f"{_slug(observable_name)}.png"
                    )
                ] = png
                encoded = base64.b64encode(png).decode("ascii")
                body = f'<img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">'
            body += _chart_caption(observable, kind)
            if kind in ("runs", "table"):
                body += _gap_table(observable)
            elif observable["report_only"]:
                body += (
                    '<details class="exact"><summary>Exact gaps</summary>'
                    f"{_gap_table(observable)}</details>"
                )
            else:
                body += "".join(
                    _html_table(
                        STATISTICS_HEADERS,
                        statistics_rows(channel),
                        container="table-scroll compact",
                    )
                    for channel in observable["channels"]
                )
            figures.append(
                f"<figure><figcaption>{html.escape(caption)}</figcaption>{body}</figure>"
            )
        note = suite["note"]
        note_html = "" if not note else f'<p class="suite-note">{html.escape(note)}</p>'

        if not figures:
            figures.append('<p class="suite-note">No charted observables.</p>')
        performance = (
            _performance_html(report.get("performance"))
            if suite_name == "inference"
            else ""
        )
        suite_sections.append(
            "<details>"
            f"<summary>{html.escape(display_name(suite_name))} — "
            f"{html.escape(suite['status'])}</summary>"
            f'<div class="suite-body">{note_html}'
            f"{_explanation_html(suite['explanation'])}{''.join(gate_figures)}"
            f'{"".join(figures)}{performance}</div>'
            "</details>"
        )
    return _html_document(report, suite_sections), images


def _is_candidate_report_directory(path: Path) -> bool:
    stored = path / "report.json"
    try:
        payload = json.loads(stored.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(payload, dict) and payload.get("schema") == CANDIDATE_REPORT_SCHEMA
    )


def write_candidate_html_report(
    report: Mapping[str, Any],
    report_dir: str | Path,
) -> Path:
    """
    Atomically publish one complete candidate HTML report directory.

    Rendering and all file writes occur in a sibling temporary directory,
    which is renamed into place only after every output has succeeded. A
    previous candidate report in the destination is replaced; any other
    existing directory is left untouched.

    Args:
        report: Candidate report mapping enriched with compact plot payloads.
        report_dir: Directory that will contain the report.

    Returns:
        Path to the published ``index.html``.

    Raises:
        FileExistsError: If ``report_dir`` exists and is not a candidate report.
    """
    destination = Path(report_dir).expanduser().resolve()
    if destination.exists() and not _is_candidate_report_directory(destination):
        raise FileExistsError(
            f"report directory already exists and is not a candidate report: "
            f"{destination}"
        )
    document, images = render_candidate_html_report(report)
    with staged_directory(destination, replace=True) as temporary:
        (temporary / "index.html").write_text(document, encoding="utf-8")
        write_json(temporary / "report.json", report)
        for relative_path, content in images.items():
            output = temporary.joinpath(*relative_path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
    return destination / "index.html"
