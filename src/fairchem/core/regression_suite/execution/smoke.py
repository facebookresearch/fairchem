"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from fairchem.core.regression_suite.artifacts.baselines import (
    check_baseline,
    load_baseline,
)
from fairchem.core.regression_suite.artifacts.storage import (
    fragment_path,
    open_artifact,
    write_json_atomic,
)
from fairchem.core.regression_suite.evaluation.comparison import summarize_run
from fairchem.core.regression_suite.evaluation.decision import (
    qualify_suite,
    tightest_tier,
)
from fairchem.core.regression_suite.evaluation.identity import check_baseline_plans
from fairchem.core.regression_suite.execution.environment import (
    prepare_compilation_cache,
)
from fairchem.core.regression_suite.execution.runner import (
    resolve_baseline_checkpoint,
    resolve_checkpoint,
    run_suite,
    task_config,
)
from fairchem.core.regression_suite.reporting.check_report import finite_or_none
from fairchem.core.regression_suite.spec.profiles import profile_config
from fairchem.core.regression_suite.spec.schema import CHANNEL_FIELDS
from fairchem.core.regression_suite.suites import load_suite

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class SuiteCheck(TypedDict):
    """
    One suite's quick or fast check on its smoke cases.

    ``reference`` names what the candidate was compared with: the reference
    profile of a quick check, or the baseline of a fast check. Only a fast
    check grades, so only it has a ``qualification``.
    """

    mode: Literal["quick", "fast"]
    suite: str
    profile: str
    reference: str
    checkpoint: str
    records_path: str
    reference_path: str
    summary: dict[str, Any]
    status: str
    passed: bool
    verdicts: dict[str, str]
    qualification: dict[str, Any] | None
    elapsed_seconds: float


def _run_smoke_cases(
    suite: str, profile: str, output_dir: Path, **values: Any
) -> tuple[Path, str]:
    """
    Run one profile on a suite's smoke cases and return its records directory
    and profile label.
    """
    destination = output_dir.expanduser().resolve()
    config = task_config(
        suite,
        "comparison",
        profile,
        {"scope": "smoke", "output_dir": str(destination), **values},
    )
    identity = run_suite(config)["identity"]
    return (
        fragment_path(destination, suite, identity["profile"], "comparison"),
        identity["profile"],
    )


def _quick_suite(
    suite: str,
    output_dir: Path,
    *,
    profile: str,
    reference_profile: str,
    checkpoint: str,
    checkpoint_id: str | None,
    device: str,
) -> SuiteCheck:
    """
    Run the reference and the candidate on one suite's smoke cases and
    summarize the candidate's differences to the reference.
    """
    started = time.perf_counter()
    checkpoint, checkpoint_id = resolve_checkpoint(checkpoint, checkpoint_id)
    model = {"checkpoint": checkpoint, "checkpoint_id": checkpoint_id, "device": device}
    # Separate directories keep the two runs apart when both use one profile.
    reference_path, reference_label = _run_smoke_cases(
        suite, reference_profile, output_dir / "reference", **model
    )
    records_path, label = _run_smoke_cases(
        suite, profile, output_dir / "candidate", **model
    )
    definition = load_suite(suite)
    summary = summarize_run(records_path, reference_path, definition)
    verdicts = {
        name: "ok"
        if all(
            math.isfinite(value)
            for key, _ in CHANNEL_FIELDS.values()
            for value in observable[key].values()
        )
        else "NON-FINITE"
        for name, observable in summary["observables"].items()
    }
    for name, observable in definition.observables.items():
        if not observable.report_only and name not in verdicts:
            verdicts[name] = "MISSING"
    problems = set(verdicts.values()) - {"ok"}
    status = "+".join(sorted(problems)) if problems else "OK"
    return {
        "mode": "quick",
        "suite": suite,
        "profile": label,
        "reference": reference_label,
        "checkpoint": checkpoint_id,
        "records_path": str(records_path),
        "reference_path": str(reference_path),
        "summary": summary,
        "status": status,
        "passed": status == "OK",
        "verdicts": verdicts,
        "qualification": None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _fast_suite(
    suite: str,
    output_dir: Path,
    *,
    baseline_dir: Path,
    profile: str,
    device: str,
    checkpoint: str,
    checkpoint_id: str,
) -> SuiteCheck:
    """
    Run and qualify one suite's smoke cases against a finalized baseline.
    """
    started = time.perf_counter()
    root = open_artifact(baseline_dir)
    baseline = load_baseline(root / "baseline.json")
    records_path, label = _run_smoke_cases(
        suite,
        profile,
        output_dir,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        seed=int(baseline["seed"]),
        device=device,
    )
    baseline_suite = baseline["suites"][suite]
    reference_path = (root / baseline_suite["reference"]["manifest"]).parent
    summary = summarize_run(records_path, reference_path, load_suite(suite))
    qualification = qualify_suite(summary, baseline_suite, smoke=True)
    verdicts = {}
    for name in summary["observables"]:
        decision = qualification["observables"].get(name)
        if decision is None or decision["report_only"]:
            verdicts[name] = "REPORT ONLY"
        else:
            verdicts[name] = tightest_tier(decision["tier_pass"]) or "BEYOND_TF32"
    return {
        "mode": "fast",
        "suite": suite,
        "profile": label,
        "reference": baseline["baseline_id"],
        "checkpoint": checkpoint_id,
        "records_path": str(records_path),
        "reference_path": str(reference_path),
        "summary": summary,
        "status": qualification["status"],
        "passed": qualification["status"].startswith("SMOKE_PASS"),
        "verdicts": verdicts,
        "qualification": qualification,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _check_suites(
    suites: Sequence[str],
    output_dir: str | Path,
    check: Callable[[str, Path], SuiteCheck],
) -> list[SuiteCheck]:
    """
    Check each suite in turn and save each result as ``<suite>/result.json``.

    JSON has no NaN or infinity, so non-finite statistics are saved as null;
    their verdicts still name them.
    """
    prepare_compilation_cache()
    results = []
    for suite in suites:
        suite_dir = Path(output_dir) / suite
        result = check(suite, suite_dir)
        write_json_atomic(suite_dir / "result.json", finite_or_none(result))
        results.append(result)
    return results


def run_quick(
    *,
    profile: str,
    reference_profile: str,
    suites: Sequence[str],
    output_dir: str | Path,
    checkpoint: str,
    checkpoint_id: str | None = None,
    device: str = "cuda",
) -> list[SuiteCheck]:
    """
    Compare a profile with a reference profile on each suite's smoke cases,
    without a baseline.
    """
    return _check_suites(
        suites,
        output_dir,
        functools.partial(
            _quick_suite,
            profile=profile,
            reference_profile=reference_profile,
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
            device=device,
        ),
    )


def run_fast(
    *,
    baseline_dir: str | Path,
    profile: str,
    suites: Sequence[str],
    output_dir: str | Path,
    device: str = "cuda",
    checkpoint: str | None = None,
    checkpoint_id: str | None = None,
) -> list[SuiteCheck]:
    """
    Run and qualify the suite-owned smoke cases of each suite in turn on
    ``device``, which should be the baseline's.

    The checkpoint is the baseline's unless given; a checkpoint path needs
    ``checkpoint_id``, and either must have the baseline's identity.
    """
    baseline_root = Path(baseline_dir).expanduser().resolve()
    baseline = check_baseline(baseline_root)
    checkpoint, checkpoint_id = resolve_baseline_checkpoint(
        baseline, checkpoint, checkpoint_id
    )
    check_baseline_plans(baseline, suites, profile_config(profile))
    return _check_suites(
        suites,
        output_dir,
        functools.partial(
            _fast_suite,
            baseline_dir=baseline_root,
            profile=profile,
            device=device,
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
        ),
    )
