"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# ``MeasurementRecord.values`` is a NumPy array, not a pandas object.
# ruff: noqa: PD011
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core.regression_suite.artifacts.baselines import (
    baseline_provenance,
    load_baseline,
)
from fairchem.core.regression_suite.artifacts.storage import (
    ArtifactPath,
    baseline_run_directory,
    fragment_path,
    load_reference,
    open_artifact,
)
from fairchem.core.regression_suite.evaluation.comparison import (
    _primary_difference,
    summarize_run,
)
from fairchem.core.regression_suite.evaluation.decision import (
    decision_channels,
    limiting_check,
    qualify_suite,
    report_channels,
    tier_status,
    tightest_tier,
)
from fairchem.core.regression_suite.evaluation.identity import (
    suite_expectation,
    validate_identity,
)
from fairchem.core.regression_suite.evaluation.performance import (
    compare_performance,
    load_performance_sidecar,
)
from fairchem.core.regression_suite.spec.hashing import canonical_json
from fairchem.core.regression_suite.spec.profiles import (
    profile_config,
    unsupported_suites_for_profile,
)
from fairchem.core.regression_suite.spec.schema import (
    CANDIDATE_REPORT_SCHEMA,
    CANDIDATE_REPORT_SCHEMA_VERSION,
    CHANNEL_FIELDS,
    DECISION_STATISTICS,
    PRECISION_TIERS,
    PROFILE_TIERS,
)
from fairchem.core.regression_suite.suites import SUITES, load_suite

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fairchem.core.regression_suite.suites.base import (
        ObservableDescription,
        Suite,
    )


def _execution_environment_rows(
    baseline: Mapping[str, Any],
    selected_suites: Sequence[str],
    candidate_runs: Sequence[tuple[str, Mapping[str, Any]]],
    role: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(
        source: str,
        role: str,
        profile: str,
        suite: str,
        environment: Mapping[str, Any],
    ) -> None:
        entry = grouped.setdefault(
            (source, role, profile, canonical_json(environment)),
            {
                "source": source,
                "role": role,
                "profile": profile,
                "suites": [],
                "environment": dict(environment),
            },
        )
        entry["suites"].append(suite)

    for suite in selected_suites:
        baseline_suite = baseline["suites"][suite]
        add(
            "baseline",
            "reference",
            "fp64-eager",
            suite,
            baseline_suite["reference"]["execution_environment"],
        )
        for profile, _tier in PROFILE_TIERS:
            add(
                "baseline",
                "calibration",
                profile,
                suite,
                baseline_suite["profiles"][profile]["execution_environment"],
            )
            add(
                "baseline",
                "verification",
                profile,
                suite,
                baseline_suite["verifications"][profile]["execution_environment"],
            )
    for suite, identity in candidate_runs:
        add(
            "candidate",
            role,
            str(identity["profile"]),
            suite,
            identity["execution_environment"],
        )
    order = {"reference": 0, "calibration": 1, "verification": 2, "comparison": 3}
    rows = list(grouped.values())
    for row in rows:
        row["suites"] = sorted(row["suites"])
    return sorted(
        rows,
        key=lambda row: (
            0 if row["source"] == "baseline" else 1,
            order[row["role"]],
            row["profile"],
            canonical_json(row["environment"]),
        ),
    )


def _qualify_candidate(
    summary: Mapping[str, Any],
    *,
    suite: str,
    profile: str,
    baseline: Mapping[str, Any],
    expectation: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    baseline_suite = baseline["suites"][suite]
    validate_identity(
        summary["identity"],
        {
            "suite": suite,
            "profile": profile,
            "role": role,
            "plan_hash": baseline_suite["plan_hash"],
            "suite_contract_hash": baseline_suite["suite_contract_hash"],
            "checkpoint": baseline["checkpoint"],
            "seed": baseline["seed"],
            "git_dirty": False,
            "device": baseline["device"],
            "predictor_scope": expectation["predictor_scope"],
            "inference_settings": expectation["inference_settings"],
            "case_count": baseline_suite["profiles"]["fp64-eager"]["completion"][
                "requested_cases"
            ],
            "reference_hash": baseline_suite["reference"]["reference_hash"],
        },
        label=f"{suite} candidate",
    )
    return qualify_suite(summary, baseline_suite)


def _cluster_values(
    summary: Mapping[str, Any] | None, value_key: str, cluster_ids: Sequence[str]
) -> list[float | None] | None:
    """
    Return one run's cluster means in ``cluster_ids`` order, None where absent.
    """
    if summary is None:
        return None
    values = {c["cluster_id"]: c[value_key] for c in summary["decision_clusters"]}
    return [
        None if cluster_id not in values else float(f"{values[cluster_id]:.4g}")
        for cluster_id in cluster_ids
    ]


def _channels(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    qualification: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    result = []
    cluster_ids = baseline["decision_cluster_ids"]
    for channel in report_channels(baseline["comparison"]):
        statistics_key, value_key = CHANNEL_FIELDS[channel]
        decides = not baseline["report_only"] and channel in decision_channels(
            baseline["comparison"]
        )
        graded = decides and qualification is not None
        channel_report = {
            "name": channel,
            "statistics": (
                None if candidate is None else dict(candidate[statistics_key])
            ),
            "calibration": {
                tier: dict(baseline["profiles"][profile][statistics_key])
                for profile, tier in PROFILE_TIERS
            },
            "limits": (
                {tier: baseline["limits"][tier][channel] for tier in PRECISION_TIERS}
                if decides
                else None
            ),
            "channel_pass": (
                {
                    tier: qualification["channel_pass"][tier][channel]
                    for tier in PRECISION_TIERS
                }
                if graded
                else None
            ),
            "limit_ratios": (
                {
                    tier: qualification["limit_ratios"][tier][channel]
                    for tier in PRECISION_TIERS
                }
                if graded
                else None
            ),
            # Axes are frozen in the baseline, so a candidate cannot rescale
            # its own charts.
            "plot": {"axis": dict(baseline["report_axes"][channel])},
            "points": {
                **{
                    tier: _cluster_values(
                        baseline["profiles"][profile], value_key, cluster_ids
                    )
                    for profile, tier in PROFILE_TIERS
                },
                "candidate": _cluster_values(candidate, value_key, cluster_ids),
            },
        }
        result.append(channel_report)
    return result


def _observable_judgment(
    qualification: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Return an observable's status and its limiting check at the tier it is
    judged at: its tightest passing tier, or the loosest tier when it fails.
    """
    if qualification is None or qualification["report_only"]:
        return ("REPORT ONLY" if qualification else None), None
    tightest = tightest_tier(qualification["tier_pass"])
    tier = tightest or PRECISION_TIERS[-1]
    channel, statistic, ratio = limiting_check(qualification, tier)
    return tier_status(tightest), {
        "tier": tier,
        "channel": channel,
        "statistic": statistic,
        "ratio_to_limit": ratio,
    }


def _cluster_rows(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    limiting: Mapping[str, Any] | None,
    statistic: str,
) -> list[dict[str, Any]]:
    """
    Return every decision cluster's error, worst first.

    For decision observables, each cluster's mean in the limiting channel is
    divided by that channel's ``statistic`` limit at the judged tier.
    """
    if candidate is None:
        return []
    channels = report_channels(baseline["comparison"])
    rows = []
    for cluster in candidate["decision_clusters"]:
        values = {channel: cluster[CHANNEL_FIELDS[channel][1]] for channel in channels}
        ratio = None
        if limiting is not None:
            limit = baseline["limits"][limiting["tier"]][limiting["channel"]][statistic]
            ratio = values[limiting["channel"]] / limit
        rows.append(
            {"cluster_id": cluster["cluster_id"], "values": values, "ratio": ratio}
        )
    channel = limiting["channel"] if limiting else channels[0]
    return sorted(
        rows,
        key=lambda row: (
            -(row["ratio"] if row["ratio"] is not None else row["values"][channel]),
            row["cluster_id"],
        ),
    )


def _observable_report(
    name: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    qualification: Mapping[str, Any] | None,
    description: ObservableDescription | None,
) -> dict[str, Any]:
    status, limiting = _observable_judgment(qualification)
    # Clusters are compared with the loosest graded statistic's limit.
    cluster_statistic = baseline.get("decision_statistics", DECISION_STATISTICS)[-1]
    return {
        "name": name,
        "description": None if description is None else description._asdict(),
        "cluster_ids": list(baseline["decision_cluster_ids"]),
        "items": None,
        "systems": None,
        "series": None,
        "units": baseline["units"],
        "comparison": baseline["comparison"],
        "floor": baseline["floor"],
        "report_only": baseline["report_only"],
        "population": None if candidate is None else dict(candidate["population"]),
        "tier_pass": None if qualification is None else qualification.get("tier_pass"),
        "selected_channel": (
            None if qualification is None else qualification.get("selected_channel")
        ),
        "channels": _channels(candidate, baseline, qualification),
        "status": status,
        "limiting": limiting,
        "cluster_statistic": None if limiting is None else cluster_statistic,
        "clusters": _cluster_rows(candidate, baseline, limiting, cluster_statistic),
    }


def _performance_comparison(
    baseline_dir: ArtifactPath,
    candidate_dir: ArtifactPath,
    profile: str,
    baseline: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    candidate_path = (
        fragment_path(candidate_dir, "inference", profile, role).parent
        / "performance.json"
    )
    if not candidate_path.is_file():
        return {
            "status": "NOT AVAILABLE",
            "reason": "candidate performance sidecar not found",
        }
    baseline_suite = baseline["suites"]["inference"]
    baselines = {}
    for baseline_profile, tier in PROFILE_TIERS:
        paths = [
            (
                baseline_dir / baseline_suite[collection][baseline_profile]["fragment"]
            ).parent.parent
            / "performance.json"
            for collection in ("profiles", "verifications")
        ]
        if not all(path.is_file() for path in paths):
            return {
                "status": "NOT AVAILABLE",
                "reason": (
                    "main performance sidecars are incomplete; run a new "
                    "baseline campaign"
                ),
            }
        baselines[tier] = [load_performance_sidecar(path) for path in paths]
    return compare_performance(load_performance_sidecar(candidate_path), baselines)


def _energy_deviations(path: ArtifactPath) -> dict[str, np.ndarray]:
    """
    Return one run's NVE energy-deviation series by case.
    """
    _manifest, records = load_reference(path)
    return {
        record.case_id: record.values
        for record in records
        if record.observable == "energy_deviation"
    }


def _item_key(cluster_id: str) -> str:
    """
    Return the case of a cluster, without its orientation.
    """
    return re.sub(r"/orientation-\d+$", "", cluster_id.split("::")[0])


def _systems(definition: Suite) -> dict[str, dict[str, Any]]:
    """
    Describe each item of a suite's plan: its name, formula, and atom count.
    """
    systems: dict[str, dict[str, Any]] = {}
    for case in definition.plan().cases:
        key = _item_key(case.case_id)
        if key not in systems:
            atoms = definition.source_atoms(case)
            systems[key] = {
                "label": key.split("/", 1)[-1],
                "formula": atoms.get_chemical_formula(),
                "atoms": len(atoms),
            }
    return systems


# Observables with at most this many clusters are shown run by run.
FEW_CLUSTERS = 4


def _short_series(
    baseline_root: ArtifactPath,
    suite: Suite,
    candidate_records: ArtifactPath | None,
    names: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Return each named observable's FP64 reference values by cluster, and each
    sample's gap to them for the FP32 and TF32 calibration runs and the
    candidate, as the observable compares them.
    """
    runs = {
        "FP32": fragment_path(baseline_root, suite.name, "fp32-eager", "calibration"),
        "TF32": fragment_path(baseline_root, suite.name, "tf32-eager", "calibration"),
    }
    if candidate_records is not None:
        runs["candidate"] = candidate_records
    _manifest, references = load_reference(
        fragment_path(baseline_root, suite.name, "fp64-eager", "reference")
    )
    reference = {
        record.key: record
        for record in references
        if record.observable in names and record.values.ndim == 1
    }
    series: dict[str, dict[str, dict[str, Any]]] = {}
    for record in reference.values():
        series.setdefault(record.observable, {})[record.cluster_id] = {
            "reference": [float(f"{value:.6g}") for value in record.values],
            "gaps": {},
        }
    for label, path in runs.items():
        _manifest, records = load_reference(path)
        for record in records:
            if record.key in reference:
                primary, _absolute = _primary_difference(
                    reference[record.key].values,
                    record.values,
                    suite.observables[record.observable],
                )
                entry = series[record.observable][record.cluster_id]
                entry["gaps"][label] = _significant(primary)
    return series


def _significant(values: np.ndarray) -> list[float]:
    # Gate curves span many decades, so they keep significant digits.
    return [float(f"{value:.4g}") for value in values]


def _nve_gate_curves(
    baseline_root: ArtifactPath, candidate_records: ArtifactPath | None
) -> dict[str, Any]:
    """
    Return each NVE system's gate curves for the FP64, FP32, and TF32
    calibration runs and the candidate, with the candidate's per-seed curves.

    Every curve compares a run with the FP64 reference of the same seeds, and
    each graded gate value is its curve's time average.
    """
    # Suite modules are imported only when their suite is used.
    from fairchem.core.regression_suite.suites.nve_md.suite import (
        DATASETS,
        GAP_SMOOTHING_FRAMES,
        GATES,
        SAMPLE_INTERVAL,
        SYSTEMS,
        atom_count,
        drift_window,
        gate_curves,
        gate_observable,
    )

    reference = _energy_deviations(
        fragment_path(baseline_root, "nve-md", "fp64-eager", "reference")
    )
    runs = {
        tier: _energy_deviations(
            fragment_path(baseline_root, "nve-md", profile, "calibration")
        )
        for profile, tier in PROFILE_TIERS
    }
    if candidate_records is not None:
        runs["candidate"] = _energy_deviations(candidate_records)
    systems = []
    for system in SYSTEMS:
        case_ids = sorted(c for c in reference if c.split("/")[2] == system.name)
        if not case_ids:
            continue
        frame_ps = SAMPLE_INTERVAL * DATASETS[system.dataset].timestep_fs / 1000.0
        window = drift_window(system)
        references = np.stack([reference[case_id] for case_id in case_ids])
        curves = {
            label: gate_curves(
                np.stack([run[case_id] for case_id in case_ids]), references, window
            )
            for label, run in runs.items()
        }
        frames = references.shape[1]
        systems.append(
            {
                "name": system.name,
                "dataset": system.dataset,
                "timestep_fs": DATASETS[system.dataset].timestep_fs,
                "temperature_k": system.kelvin,
                "steps": system.steps,
                "atoms": atom_count(system),
                "seeds": len(case_ids),
                "sample_interval": SAMPLE_INTERVAL,
                "frame_ps": frame_ps,
                "gap_frames": GAP_SMOOTHING_FRAMES,
                "drift_frames": window,
                "gap_times_ps": _significant(
                    (
                        np.arange(frames - GAP_SMOOTHING_FRAMES + 1)
                        + (GAP_SMOOTHING_FRAMES - 1) / 2
                    )
                    * frame_ps
                ),
                "drift_times_ps": _significant(
                    (np.arange(frames - window + 1) + (window - 1) / 2) * frame_ps
                ),
                "observables": {
                    gate: gate_observable(system.name, gate) for gate in GATES
                },
                "curves": {
                    label: {gate: _significant(values[gate]) for gate in GATES}
                    for label, values in curves.items()
                },
                "candidate_seeds": None
                if "candidate" not in curves
                else {
                    gate: [
                        _significant(row)
                        for row in curves["candidate"][f"{gate}_seeds"]
                    ]
                    for gate in ("fp64_gap", "drift")
                },
            }
        )
    return {"systems": systems}


def build_candidate_report(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    candidate_profile: str,
    suites: Sequence[str] | None = None,
    role: str = "comparison",
) -> dict[str, Any]:
    """
    Build a complete, artifact-only candidate regression report.

    Args:
        baseline_dir: Sealed baseline file, or a finalized baseline directory.
        candidate_dir: Directory or sealed file containing the reported records.
        candidate_profile: Profile label or Hydra profile config name.
        suites: Optional subset of suites to include in the report.
        role: Role of the reported runs; ``verification`` reports a baseline's
            held-out runs, which live in the baseline itself.

    Returns:
        A report containing every selected suite and observable.
    """
    baseline_root = open_artifact(baseline_dir)
    candidate_root = open_artifact(candidate_dir)
    baseline = load_baseline(baseline_root / "baseline.json")
    profile = candidate_profile.replace("_", "-")
    config = profile_config(profile.replace("-", "_"))
    unsupported = unsupported_suites_for_profile(profile)
    requested = None if suites is None else {s.replace("_", "-") for s in suites}
    selected_suites = tuple(
        suite
        for suite in SUITES
        if suite in baseline["suites"] and (requested is None or suite in requested)
    )
    suite_reports = []
    candidate_runs: list[tuple[str, Mapping[str, Any]]] = []
    for suite in selected_suites:
        definition = load_suite(suite)
        baseline_suite = baseline["suites"][suite]
        baseline_observables = baseline_suite["observables"]
        records_path = fragment_path(candidate_root, suite, profile, role)
        summary = qualification = None
        if records_path.is_dir():
            summary = summarize_run(
                records_path,
                (baseline_root / baseline_suite["reference"]["manifest"]).parent,
                definition,
            )
            qualification = _qualify_candidate(
                summary,
                suite=suite,
                profile=profile,
                baseline=baseline,
                expectation=suite_expectation(suite, config),
                role=role,
            )
            candidate_runs.append((suite, summary["identity"]))
        observables = [
            _observable_report(
                name,
                baseline_observables[name],
                None if summary is None else summary["observables"][name],
                None if qualification is None else qualification["observables"][name],
                definition.observable_descriptions.get(name),
            )
            for name in sorted(baseline_observables)
        ]
        if suite != "nve-md":
            systems = _systems(definition)
            for observable in observables:
                keys = [_item_key(c) for c in observable["cluster_ids"]]
                observable["items"] = keys
                observable["systems"] = {
                    key: systems[key] for key in dict.fromkeys(keys)
                }
        few = {
            name
            for name, observable in baseline_observables.items()
            if len(observable["decision_cluster_ids"]) <= FEW_CLUSTERS
        }
        if few and suite != "nve-md":
            series = _short_series(
                baseline_root,
                definition,
                records_path if records_path.is_dir() else None,
                few,
            )
            for observable in observables:
                observable["series"] = series.get(observable["name"])
        gate_curves = (
            _nve_gate_curves(
                baseline_root, records_path if records_path.is_dir() else None
            )
            if suite == "nve-md"
            else None
        )
        if summary is None:
            suite_reports.append(
                {
                    "suite": suite,
                    "status": "UNSUPPORTED" if suite in unsupported else "NOT RUN",
                    "completion": None,
                    "population": None,
                    "tier_pass": None,
                    "tightest_tier": None,
                    "limiting": None,
                    "note": "unsupported by this profile"
                    if suite in unsupported
                    else None,
                    "observables": observables,
                    "gate_curves": gate_curves,
                    "explanation": [list(part) for part in definition.explanation],
                }
            )
            continue
        decisions = [
            summary["observables"][name]
            for name in sorted(summary["observables"])
            if not summary["observables"][name]["report_only"]
        ]
        suite_reports.append(
            {
                "suite": suite,
                "status": qualification["status"],
                "completion": dict(summary["completion"]),
                "population": {
                    "clusters": len(
                        {
                            cluster["cluster_id"]
                            for observable in decisions
                            for cluster in observable["decision_clusters"]
                        }
                    ),
                    "values": sum(
                        int(observable["population"]["values"])
                        for observable in decisions
                    ),
                },
                "tier_pass": qualification["tier_pass"],
                "tightest_tier": qualification["tightest_tier"],
                "limiting": dict(qualification["limiting"]),
                "note": None,
                "observables": observables,
                "gate_curves": gate_curves,
                "explanation": [list(part) for part in definition.explanation],
            }
        )
    if not candidate_runs:
        raise FileNotFoundError(
            f"candidate profile {profile!r} has no saved records in {candidate_root}"
        )
    identity = candidate_runs[0][1]
    report = {
        "schema": CANDIDATE_REPORT_SCHEMA,
        "schema_version": CANDIDATE_REPORT_SCHEMA_VERSION,
        "baseline": {
            **baseline_provenance(baseline, baseline_dir, selected_suites),
            "seed": baseline["seed"],
        },
        "candidate": {
            key: identity[key]
            for key in ("profile", "checkpoint", "seed", "git_commit", "git_dirty")
        },
        "executions": _execution_environment_rows(
            baseline, selected_suites, candidate_runs, role
        ),
        "suites": suite_reports,
    }
    if "inference" in selected_suites:
        report["performance"] = _performance_comparison(
            baseline_root, candidate_root, profile, baseline, role
        )
    return report


def verification_report(
    baseline_dir: str | Path, profile: str
) -> tuple[dict[str, Any], Path]:
    """
    Build the report of a baseline's held-out verification run of one main
    profile, and return it with its report directory.
    """
    report = build_candidate_report(
        baseline_dir, baseline_dir, candidate_profile=profile, role="verification"
    )
    reports = baseline_run_directory(Path(baseline_dir).expanduser().resolve())
    return report, reports / "reports" / f"verification-{profile.replace('_', '-')}"


def candidate_report_exit_code(report: Mapping[str, Any]) -> int:
    """
    Return the process status implied by suite qualification results.

    Unsupported profile capabilities remain informational. A numerical failure
    or a suite that did not run makes the complete candidate report fail.
    """
    passing = {"PASS", "PASS@FP32", "PASS@TF32", "UNSUPPORTED"}
    return int(any(suite["status"] not in passing for suite in report["suites"]))
