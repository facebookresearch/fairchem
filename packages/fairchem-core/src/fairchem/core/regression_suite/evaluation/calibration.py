"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import datetime
import itertools
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fairchem.core import __version__ as fairchem_version
from fairchem.core.regression_suite.artifacts.baselines import load_baseline
from fairchem.core.regression_suite.artifacts.storage import (
    fragment_path,
    load_reference_manifest,
    pack_baseline,
    write_json_atomic,
)
from fairchem.core.regression_suite.evaluation.comparison import summarize_run
from fairchem.core.regression_suite.evaluation.decision import (
    decision_channels,
    limiting_check,
    qualify_suite,
    report_channels,
)
from fairchem.core.regression_suite.evaluation.identity import (
    run_scope,
    validate_identity,
)
from fairchem.core.regression_suite.spec.hashing import file_sha256, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    BASELINE_SCHEMA,
    BASELINE_SCHEMA_VERSION,
    CHANNEL_FIELDS,
    DECISION_STATISTICS,
    PROFILE_TIERS,
    HistogramSpec,
    RegressionSchemaError,
)
from fairchem.core.regression_suite.suites import CAMPAIGN_SUITES, load_suite

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


# Differences below this value, in an observable's primary-channel units, are
# numerically identical to FP64. Repeated FP64 GPU executions differ by
# roughly 1e-16 to 1e-11, so limits scaled from that rounding noise alone
# would be unstable; FP32 errors are at least 1e-8 for every current observable.
EQUIVALENCE_FLOOR = 1.0e-10
_FP64_SETTINGS = {
    "base_precision_dtype": "float64",
    "tf32": False,
    "compile": False,
    "execution_mode": "general",
    "merge_mole": False,
    "activation_checkpointing": False,
}
# Report axes span these quantiles of the main runs' cluster means, padded
# by whole decades.
_AXIS_QUANTILES = (0.01, 0.99)
_AXIS_PADDING_DECADES = 1.0
# Fields copied from a run's identity, and an observable's contract fields.
_RUN_FIELDS = (
    "run_id",
    "git_commit",
    "git_dirty",
    "checkpoint",
    "inference_settings",
    "predictor_scope",
    "execution_environment",
)
_CONTRACT_FIELDS = (
    "units",
    "comparison",
    "floor",
    "report_only",
    "population",
)


def _rounded_outward(value: float, *, lower: bool) -> int:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        return int(rounded)
    return math.floor(value) if lower else math.ceil(value)


def _report_axis(
    histogram: HistogramSpec,
    runs: list[Mapping[str, Any]],
    channel: str,
) -> dict[str, Any]:
    """
    Fit and freeze a log report axis to main-branch decision values.

    The axis spans the ``_AXIS_QUANTILES`` of the positive cluster means and every
    run's median, padded and rounded outward to whole decades. Exact zeros are
    shown as a separate count; an all-zero population keeps the observable's
    histogram bounds.
    """
    statistics_key, value_key = CHANNEL_FIELDS[channel]
    values = np.asarray(
        [cluster[value_key] for run in runs for cluster in run["decision_clusters"]],
        dtype=np.float64,
    )
    medians = np.asarray(
        [run[statistics_key]["median_absolute"] for run in runs], dtype=np.float64
    )
    minimum, maximum = histogram.minimum, histogram.maximum
    if (values > 0.0).any():
        lower, upper = np.quantile(
            np.log10(values[values > 0.0]), _AXIS_QUANTILES, method="linear"
        )
        if (medians > 0.0).any():
            median_logs = np.log10(medians[medians > 0.0])
            lower = min(float(lower), float(np.min(median_logs)))
            upper = max(float(upper), float(np.max(median_logs)))
        minimum = 10.0 ** _rounded_outward(
            float(lower) - _AXIS_PADDING_DECADES, lower=True
        )
        maximum = 10.0 ** _rounded_outward(
            float(upper) + _AXIS_PADDING_DECADES, lower=False
        )
    return {"minimum": minimum, "maximum": maximum}


def _expected_profile_settings(
    reference_settings: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    expected = dict(reference_settings)
    if profile in {"fp32-eager", "tf32-eager"}:
        expected["base_precision_dtype"] = "float32"
    expected["tf32"] = profile == "tf32-eager"
    return expected


def _campaign_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the fields every run of one baseline shares, which the baseline
    records at its top level.
    """
    return {
        "checkpoint": identity["checkpoint"],
        "seed": identity["seed"],
        "device": identity["device"],
        "scope": run_scope(identity),
    }


def _validate_run(
    summary: Mapping[str, Any],
    *,
    suite: str,
    profile: str,
    role: str,
    reference: Mapping[str, Any],
) -> None:
    identity = reference["identity"]
    run_identity = summary["identity"]
    validate_identity(
        {**run_identity, "scope": run_scope(run_identity)},
        {
            "suite": suite,
            "profile": profile,
            "role": role,
            "git_dirty": False,
            "inference_settings": _expected_profile_settings(
                identity["inference_settings"], profile
            ),
            "predictor_scope": identity["predictor_scope"],
            "git_commit": identity["git_commit"],
            **_campaign_fields(identity),
        },
        label=f"{suite}/{profile}/{role}",
    )


def _limits(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    safety_factor: float,
    comparison: str = "absolute",
    floor: float = 0.0,
    statistics: Sequence[str] = DECISION_STATISTICS,
    merge_unresolved: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Scale each profile's decision ``statistics`` into monotonic tier limits.

    The absolute channel of a relative observable uses the equivalence floor
    multiplied by the observable's denominator floor. A log-ratio observable's
    floor is the smallest log ratio its primary limits resolve, so runs that
    reproduce one another almost exactly do not calibrate a vanishing limit.
    With ``merge_unresolved``, a tier whose looser neighbour's calibration run
    stays within the tier's limit takes that neighbour's limit, since the
    statistic cannot tell the two precisions apart.
    """
    channel_floors = {
        "primary": max(EQUIVALENCE_FLOOR, floor)
        if comparison == "log_ratio"
        else EQUIVALENCE_FLOOR,
        "absolute": EQUIVALENCE_FLOOR * floor,
    }
    limits: dict[str, dict[str, dict[str, float]]] = {}
    previous = None
    for profile, tier in PROFILE_TIERS:
        limits[tier] = {
            channel: {
                statistic: max(
                    0.0 if previous is None else previous[channel][statistic],
                    safety_factor
                    * max(
                        channel_floors[channel],
                        float(profiles[profile][CHANNEL_FIELDS[channel][0]][statistic]),
                    ),
                )
                for statistic in statistics
            }
            for channel in decision_channels(comparison)
        }
        previous = limits[tier]
    if merge_unresolved:
        # From the loosest pair down, so a merged tier passes its limit on.
        for (_, tighter), (looser_profile, looser) in reversed(
            list(itertools.pairwise(PROFILE_TIERS))
        ):
            for channel, values in limits[tighter].items():
                statistics_key = CHANNEL_FIELDS[channel][0]
                for statistic, limit in values.items():
                    observed = profiles[looser_profile][statistics_key][statistic]
                    if float(observed) <= limit:
                        values[statistic] = limits[looser][channel][statistic]
    return limits


def _software_and_hardware(
    environments: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """
    Describe the fairchem and torch versions and the device of the reference
    runs; readers treat these fields as optional.
    """

    def joined(values: Iterable[str]) -> str:
        return ", ".join(sorted(set(values)))

    return {
        "fairchem_version": fairchem_version,
        "torch_version": joined(env["torch_version"] for env in environments),
        "device_name": joined(
            env["accelerator_name"] or f"CPU ({env['cpu_architecture']})"
            for env in environments
        ),
    }


def finalize_baseline(
    artifact_dir: str | Path,
    *,
    campaign: str = "standard",
    safety_factor: float = 2.0,
    overwrite: bool = False,
    output: str = "baseline.json",
    sealed: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate calibration and independent verification runs and write a baseline.

    An existing baseline is only extended, for example from the standard to the
    full campaign, unless ``overwrite`` is set. Given ``sealed``, the baseline
    and every run it references are also packed into that single file.
    """
    artifact_root = Path(artifact_dir).resolve()
    output_path = artifact_root / output
    suites = CAMPAIGN_SUITES[campaign]
    existing_baseline = None
    if output_path.exists() and not overwrite:
        existing_baseline = load_baseline(output_path)
        if not set(existing_baseline["suites"]) < set(suites):
            raise FileExistsError(f"baseline already exists: {output_path}")

    suite_entries = {}
    verification_failures = []
    campaign_identity = None
    for suite in suites:
        definition = load_suite(suite)
        reference_path = fragment_path(artifact_root, suite, "fp64-eager", "reference")
        reference = load_reference_manifest(reference_path)
        identity = reference["identity"]
        validate_identity(
            identity,
            {
                "suite": suite,
                "profile": "fp64-eager",
                "role": "reference",
                "git_dirty": False,
            },
            label=f"{suite} reference",
        )
        validate_identity(
            identity["inference_settings"],
            _FP64_SETTINGS,
            label=f"{suite} FP64 reference settings",
        )
        reference_campaign = _campaign_fields(identity)
        campaign_identity = campaign_identity or reference_campaign
        if reference_campaign != campaign_identity:
            raise RegressionSchemaError(
                "all baseline references must share one checkpoint, seed, "
                "device, and scope"
            )

        # Summaries and baseline entries by collection, then profile.
        runs = {"profiles": {}, "verifications": {}}
        entries = {"profiles": {}, "verifications": {}}
        run_ids = {identity["run_id"]}
        for profile, _tier in PROFILE_TIERS:
            for role, collection in (
                ("calibration", "profiles"),
                ("verification", "verifications"),
            ):
                records_path = fragment_path(artifact_root, suite, profile, role)
                summary = summarize_run(records_path, reference_path, definition)
                _validate_run(
                    summary,
                    suite=suite,
                    profile=profile,
                    role=role,
                    reference=reference,
                )
                if summary["identity"]["run_id"] in run_ids:
                    raise RegressionSchemaError(
                        f"{suite} profile runs must have distinct run IDs"
                    )
                run_ids.add(summary["identity"]["run_id"])
                runs[collection][profile] = summary
                entries[collection][profile] = {
                    **{key: summary["identity"][key] for key in _RUN_FIELDS},
                    "fragment": str(
                        (records_path / "manifest.json").relative_to(artifact_root)
                    ),
                    "fragment_sha256": file_sha256(records_path / "manifest.json"),
                    "completion": summary["completion"],
                }

        statistics = definition.decision_statistics or DECISION_STATISTICS
        suite_safety_factor = definition.safety_factor or safety_factor
        observables = {}
        for name in sorted(runs["profiles"]["fp64-eager"]["observables"]):
            profile_values, verification_values = (
                {
                    profile: summary["observables"][name]
                    for profile, summary in runs[collection].items()
                }
                for collection in ("profiles", "verifications")
            )
            first = profile_values["fp64-eager"]
            observables[name] = {
                **{key: first[key] for key in _CONTRACT_FIELDS},
                "decision_cluster_ids": [
                    cluster["cluster_id"] for cluster in first["decision_clusters"]
                ],
                "profiles": profile_values,
                "verification_profiles": verification_values,
                "report_axes": {
                    channel: _report_axis(
                        definition.observables[name].histogram,
                        [*profile_values.values(), *verification_values.values()],
                        channel,
                    )
                    for channel in report_channels(first["comparison"])
                },
                "decision_statistics": list(statistics),
                "limits": None
                if first["report_only"]
                else _limits(
                    profile_values,
                    safety_factor=suite_safety_factor,
                    comparison=first["comparison"],
                    floor=float(first["floor"]),
                    statistics=statistics,
                    merge_unresolved=definition.merge_unresolved_tiers,
                ),
            }
        suite_entry = {
            "git_commit": identity["git_commit"],
            "plan_hash": identity["plan_hash"],
            "suite_contract_hash": identity["suite_contract_hash"],
            "reference": {
                "run_id": identity["run_id"],
                "reference_hash": reference["reference_hash"],
                "manifest": str(
                    (reference_path / "manifest.json").relative_to(artifact_root)
                ),
                "values": str(
                    (reference_path / reference["values_file"]).relative_to(
                        artifact_root
                    )
                ),
                "values_sha256": reference["values_sha256"],
                "record_count": reference["record_count"],
                "predictor_scope": identity["predictor_scope"],
                "execution_environment": identity["execution_environment"],
            },
            **entries,
            "observables": observables,
        }
        for profile, required_tier in PROFILE_TIERS:
            result = qualify_suite(
                {
                    "observables": {
                        name: observable["verification_profiles"][profile]
                        for name, observable in observables.items()
                    }
                },
                suite_entry,
            )
            for name, observable in sorted(result["observables"].items()):
                if observable["report_only"] or observable["tier_pass"][required_tier]:
                    continue
                channel, statistic, ratio = limiting_check(observable, required_tier)
                verification_failures.append(
                    f"{suite}/{profile} {required_tier}: {name}/{channel}/"
                    f"{statistic} is {ratio:.6g}x its limit"
                )
        suite_entries[suite] = suite_entry

    if verification_failures:
        raise RegressionSchemaError(
            f"independent verification failed {len(verification_failures)} "
            "check(s):\n" + "\n".join(verification_failures)
        )
    payload = {
        "schema": BASELINE_SCHEMA,
        "schema_version": BASELINE_SCHEMA_VERSION,
        "campaign": campaign,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        **campaign_identity,
        **_software_and_hardware(
            [
                entry["reference"]["execution_environment"]
                for entry in suite_entries.values()
            ]
        ),
        # Extending a baseline requires the same default safety factor.
        "calibration_policy": {"safety_factor": safety_factor},
        "suites": suite_entries,
    }
    if existing_baseline is not None and (
        _campaign_fields(existing_baseline),
        existing_baseline["calibration_policy"]["safety_factor"],
    ) != (campaign_identity, safety_factor):
        raise RegressionSchemaError(
            "an existing baseline can only be extended with matching provenance"
        )
    payload["baseline_id"] = stable_hash(payload)
    write_json_atomic(output_path, payload)
    if sealed is not None:
        pack_baseline(artifact_root, sealed)
    return payload
