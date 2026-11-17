"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.spec.schema import (
    CHANNEL_FIELDS,
    DECISION_STATISTICS,
    PRECISION_TIERS,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _channel_ratios(
    candidate: Mapping[str, Any],
    limits: Mapping[str, float],
    channel: str,
    *,
    smoke: bool,
    statistics: Sequence[str],
) -> dict[str, float]:
    statistics_key, value_key = CHANNEL_FIELDS[channel]
    if smoke:
        # A few smoke clusters are each held to the loosest statistic's limit.
        observed = max(
            float(cluster[value_key]) for cluster in candidate["decision_clusters"]
        )
        return {"maximum_cluster_mean": observed / limits[statistics[-1]]}
    return {
        statistic: float(candidate[statistics_key][statistic])
        / float(limits[statistic])
        for statistic in statistics
    }


def _observable_result(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    statistics = baseline.get("decision_statistics", DECISION_STATISTICS)
    ratios = {
        tier: {
            channel: _channel_ratios(
                candidate,
                baseline["limits"][tier][channel],
                channel,
                smoke=smoke,
                statistics=statistics,
            )
            for channel in decision_channels(baseline["comparison"])
        }
        for tier in PRECISION_TIERS
    }
    scores = {
        tier: {channel: max(values.values()) for channel, values in channels.items()}
        for tier, channels in ratios.items()
    }
    return {
        "report_only": False,
        "tier_pass": {
            tier: min(score.values()) <= 1.0 for tier, score in scores.items()
        },
        "channel_pass": {
            tier: {channel: value <= 1.0 for channel, value in score.items()}
            for tier, score in scores.items()
        },
        "selected_channel": {
            tier: min(score, key=score.__getitem__) for tier, score in scores.items()
        },
        "limit_ratios": ratios,
    }


def tier_status(tier: str | None) -> str:
    """
    Name the status of passing ``tier``, or FAIL when no tier passed.
    """
    return {None: "FAIL", "FP64": "PASS"}.get(tier, f"PASS@{tier}")


def tightest_tier(tier_pass: Mapping[str, bool]) -> str | None:
    """
    Return the tightest precision tier marked as passed, or None.
    """
    return next((tier for tier in PRECISION_TIERS if tier_pass[tier]), None)


def limiting_check(result: Mapping[str, Any], tier: str) -> tuple[str, str, float]:
    """
    Return an observable's selected channel at ``tier``, the statistic closest
    to its limit there, and that statistic's ratio to the limit.
    """
    channel = result["selected_channel"][tier]
    ratios = result["limit_ratios"][tier][channel]
    statistic = max(ratios, key=ratios.__getitem__)
    return channel, statistic, ratios[statistic]


def qualify_suite(
    candidate_summary: Mapping[str, Any],
    baseline_suite: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Return the tightest calibrated tier passed by every decision observable.

    A full run compares its graded statistics, median and p95 unless the suite
    sets others, with the calibrated limits. A smoke run covers only a few
    cases, so each of its cluster means must instead stay below the loosest
    graded statistic's limit; that is not a qualification of the whole
    distribution.
    """
    candidate_observables = candidate_summary["observables"]
    baseline_observables = baseline_suite["observables"]
    missing = {
        name
        for name, observable in baseline_observables.items()
        if not observable["report_only"]
    } - set(candidate_observables)
    if missing:
        raise RegressionSchemaError(f"decision observables are missing: {missing}")
    observable_results = {
        name: (
            {"report_only": True}
            if baseline_observables[name]["report_only"]
            else _observable_result(candidate, baseline_observables[name], smoke=smoke)
        )
        for name, candidate in candidate_observables.items()
    }
    decisions = {
        name: result
        for name, result in observable_results.items()
        if not result["report_only"]
    }
    tier_pass = {
        tier: all(result["tier_pass"][tier] for result in decisions.values())
        for tier in PRECISION_TIERS
    }
    tightest = tightest_tier(tier_pass)
    status = ("SMOKE_" if smoke else "") + tier_status(tightest)
    qualification = {
        "status": status,
        "tightest_tier": tightest,
        "tier_pass": tier_pass,
        "observables": observable_results,
    }
    if smoke:
        return qualification
    tier = tightest or PRECISION_TIERS[-1]
    name, channel, statistic, ratio = max(
        ((name, *limiting_check(result, tier)) for name, result in decisions.items()),
        key=lambda item: item[3],
    )
    qualification["limiting"] = {
        "observable": name,
        "channel": channel,
        "statistic": statistic,
        "ratio_to_limit": ratio,
        "tier": tier,
    }
    return qualification


def decision_channels(comparison: str) -> tuple[str, ...]:
    """
    Return the channels that can qualify an observable.

    A relative observable also passes on its absolute error, which protects
    small reference values from ill-conditioned ratios.
    """
    if comparison == "relative_with_floor":
        return ("primary", "absolute")
    return ("primary",)


def report_channels(comparison: str) -> tuple[str, ...]:
    """
    Return the channels a report shows; only absolute observables have one.
    """
    return ("primary",) if comparison == "absolute" else ("primary", "absolute")
