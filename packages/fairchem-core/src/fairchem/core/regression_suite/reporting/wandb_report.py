"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.artifacts.sources import BASELINE_ARTIFACT_TYPE
from fairchem.core.regression_suite.artifacts.storage import (
    baseline_run_directory,
    write_json_atomic,
)
from fairchem.core.regression_suite.reporting.check_report import render_check_html
from fairchem.core.regression_suite.reporting.formatting import (
    observable_rows,
    performance_highlights,
)
from fairchem.core.regression_suite.reporting.html_report import (
    write_candidate_html_report,
)
from fairchem.core.regression_suite.reporting.report import candidate_report_exit_code
from fairchem.core.regression_suite.spec.hashing import file_sha256

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


_JOB_TYPE = "candidate-regression-report"
_INFERENCE_HISTORY_METRICS = (
    ("performance_ms_per_step", "inference/ms_per_step"),
    ("performance_ns_per_day", "inference/ns_per_day"),
    ("performance_peak_gpu_memory", "inference/peak_gpu_memory_mb"),
)


@dataclass(frozen=True)
class WandBReportSettings:
    """
    Settings for publishing one regression report to W&B.

    Authentication is intentionally absent. The W&B client obtains credentials
    through its standard environment or login configuration, so secrets cannot
    become run configuration or artifact metadata through this interface.
    """

    project: str
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()


def _wandb_module() -> Any:
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError(
            "W&B report publication requires the optional 'wandb' package"
        ) from error


def _inference_performance_history(
    report: Mapping[str, Any],
) -> list[dict[str, float | int]]:
    """
    Build one W&B scalar-history row per inference system size.
    """
    history_names = dict(_INFERENCE_HISTORY_METRICS)
    rows: dict[int, dict[str, float | int]] = {}
    for highlight in performance_highlights(report):
        history_name = history_names.get(highlight["metric"])
        if history_name is None:
            continue
        atom_count = int(highlight["coordinates"]["atom_count"])
        rows.setdefault(atom_count, {"inference/num_atoms": atom_count})[
            history_name
        ] = float(highlight["candidate"])
    return [rows[atom_count] for atom_count in sorted(rows)]


def _publish_run(
    root: Path,
    settings: WandBReportSettings,
    *,
    job_type: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    exit_code: int,
    log: Callable[[Any], None] = lambda _run: None,
) -> str:
    """
    Publish one local report directory as a new W&B run and return its URL.

    The run logs ``index.html``, whatever ``log`` adds, and ``summary``, and
    uploads the directory as an artifact named and typed ``job_type``.

    Raises:
        RuntimeError: If W&B is unavailable or does not return a run URL.
    """
    html = (root / "index.html").read_text(encoding="utf-8")
    wandb = _wandb_module()
    run = wandb.init(
        project=settings.project,
        entity=settings.entity,
        name=settings.run_name,
        group=settings.group,
        tags=list(settings.tags),
        job_type=job_type,
        config=dict(config),
        save_code=False,
        dir=str(root.parent),
    )
    if run is None:
        raise RuntimeError("W&B did not create a report run")

    try:
        run.log({job_type.replace("-", "_"): wandb.Html(html, inject=False)})
        log(run)
        for key, value in summary.items():
            run.summary[key] = value
        artifact = wandb.Artifact(name=job_type, type=job_type, metadata=dict(config))
        artifact.add_dir(str(root))
        run.log_artifact(artifact)
        url = getattr(run, "url", None)
        if not isinstance(url, str) or not url:
            raise RuntimeError("W&B report run did not provide a URL")
        run.summary["publication/status"] = "SUCCEEDED"
        run.finish(exit_code=exit_code)
    except BaseException:
        with suppress(Exception):
            run.summary["publication/status"] = "FAILED"
        with suppress(Exception):
            run.finish(exit_code=1)
        raise
    return url


def _record_publication(
    root: Path,
    settings: WandBReportSettings | None,
    publish: Callable[[], str],
) -> str | None:
    """
    Run ``publish`` given W&B settings and record its outcome in ``wandb.json``.

    A failed publication keeps the local report and re-raises the error.
    """
    if settings is None:
        return None
    record = {"run_name": settings.run_name, "group": settings.group}
    try:
        url = publish()
    except Exception as error:
        write_json_atomic(
            root / "wandb.json",
            {**record, "status": "FAILED", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    write_json_atomic(
        root / "wandb.json", {**record, "status": "PUBLISHED", "url": url}
    )
    return url


def _publish_candidate_report(
    root: Path, report: Mapping[str, Any], settings: WandBReportSettings
) -> str:
    """
    Publish one written candidate report, with its per-suite qualification
    summary and inference performance history, to W&B.
    """
    baseline, candidate = report["baseline"], report["candidate"]
    metadata = {
        "baseline_id": baseline["baseline_id"],
        "baseline_name": baseline["name"],
        "baseline_created_at": baseline["created_at"],
        "baseline_git_commits": sorted(baseline["commits"]),
        "candidate_git_commit": candidate["git_commit"],
        "candidate_profile": candidate["profile"],
        "checkpoint": candidate["checkpoint"],
    }
    summary: dict[str, Any] = {}
    for suite in report["suites"]:
        summary[f"qualification/{suite['suite']}/status"] = suite["status"]
        summary[f"qualification/{suite['suite']}/tier"] = (
            suite["tightest_tier"] or "NONE"
        )
        for observable in observable_rows(suite):
            key = f"qualification/{suite['suite']}/{observable['name']}"
            summary[f"{key}/status"] = observable["status"]
            if observable["limiting"] is not None:
                summary[f"{key}/ratio"] = observable["limiting"]["ratio_to_limit"]
    for row in performance_highlights(report):
        group = "_".join(f"{key}{value}" for key, value in row["coordinates"].items())
        summary[f"performance/{row['metric']}/{group}/factor_vs_fp32"] = row["factor"]
    qualification_exit_code = candidate_report_exit_code(report)
    summary["qualification/passed"] = qualification_exit_code == 0
    summary["qualification/exit_code"] = qualification_exit_code

    def log_inference_history(run: Any) -> None:
        inference_history = _inference_performance_history(report)
        if not inference_history:
            return
        run.define_metric("inference/num_atoms", hidden=True)
        for _performance_name, history_name in _INFERENCE_HISTORY_METRICS:
            if any(history_name in row for row in inference_history):
                run.define_metric(
                    history_name,
                    step_metric="inference/num_atoms",
                    summary="none",
                )
        for row in inference_history:
            run.log(row)

    return _publish_run(
        root,
        settings,
        job_type=_JOB_TYPE,
        config=metadata,
        summary=summary,
        exit_code=qualification_exit_code,
        log=log_inference_history,
    )


def write_report(
    report: Mapping[str, Any],
    report_dir: str | Path,
    settings: WandBReportSettings | None = None,
) -> str | None:
    """
    Write a local report and, given W&B settings, publish it as a new run.

    The publication outcome is recorded in ``wandb.json`` beside the report; a
    local-only rewrite replaces the report and drops any earlier record, which
    described other content. A failed publication keeps the local report and
    re-raises the error.
    """
    root = Path(report_dir).expanduser().resolve()
    write_candidate_html_report(report, root)
    return _record_publication(
        root, settings, lambda: _publish_candidate_report(root, report, settings)
    )


def write_check_report(
    report: Mapping[str, Any],
    report_dir: str | Path,
    settings: WandBReportSettings | None = None,
) -> str | None:
    """
    Write a quick or fast check's ``report.json`` and ``index.html`` and, given
    W&B settings, publish the check directory as a new run.

    The publication outcome is recorded in ``wandb.json`` beside the report. A
    failed publication keeps the local files and re-raises the error.
    """
    root = Path(report_dir).expanduser().resolve()
    write_json_atomic(root / "report.json", report)
    (root / "index.html").write_text(render_check_html(report), encoding="utf-8")
    mode = report["mode"]
    config = {
        key: report[key]
        for key in ("mode", "profile", "reference", "checkpoint", "created_at")
    }
    if report["baseline"] is not None:
        config["baseline"] = report["baseline"]
    summary: dict[str, Any] = {}
    for suite in report["suites"]:
        summary[f"{mode}/{suite['suite']}/status"] = suite["status"]
        if suite["qualification"] is not None:
            summary[f"{mode}/{suite['suite']}/tier"] = (
                suite["qualification"]["tightest_tier"] or "NONE"
            )
    summary[f"{mode}/passed"] = report["exit_code"] == 0
    summary[f"{mode}/exit_code"] = report["exit_code"]
    return _record_publication(
        root,
        settings,
        lambda: _publish_run(
            root,
            settings,
            job_type=f"{mode}-regression-check",
            config=config,
            summary=summary,
            exit_code=report["exit_code"],
        ),
    )


def baseline_artifact_name(checkpoint: str) -> str:
    """
    Return the W&B artifact that holds every baseline of one checkpoint.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint) + "-baseline"


def publish_baseline(
    sealed: str | Path, baseline: Mapping[str, Any], settings: WandBReportSettings
) -> str:
    """
    Upload a sealed baseline file as a new version of its checkpoint's artifact.

    The version is aliased ``latest`` and with the baseline's name, and the
    outcome is recorded in ``wandb.json`` in the baseline's run directory.

    Smoke baselines are refused, since they cover too few cases to qualify
    a candidate.

    Returns:
        The qualified artifact name, ``entity/project/name:vN``.
    """
    if baseline["scope"] != "full":
        raise ValueError("only full-scope baselines are published")
    path = Path(sealed).resolve()
    run_directory = baseline_run_directory(path)
    name = run_directory.name
    record_path = run_directory / "wandb.json"
    metadata = {
        "baseline_id": baseline["baseline_id"],
        "baseline_name": name,
        "checkpoint": baseline["checkpoint"],
        "campaign": baseline["campaign"],
        "git_commits": sorted(
            {suite["git_commit"] for suite in baseline["suites"].values()}
        ),
        "sha256": file_sha256(path),
    }
    record = {"artifact": baseline_artifact_name(str(baseline["checkpoint"]))}
    try:
        wandb = _wandb_module()
        run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            name=f"{name}-baseline",
            group=settings.group or name,
            tags=list(settings.tags),
            job_type=BASELINE_ARTIFACT_TYPE,
            config=metadata,
            save_code=False,
            dir=str(record_path.parent),
        )
        try:
            artifact = wandb.Artifact(
                record["artifact"], type=BASELINE_ARTIFACT_TYPE, metadata=metadata
            )
            artifact.add_file(str(path))
            run.log_artifact(artifact, aliases=["latest", name])
            artifact.wait()
            qualified = artifact.qualified_name
            run.finish()
        except BaseException:
            with suppress(Exception):
                run.finish(exit_code=1)
            raise
    except Exception as error:
        write_json_atomic(
            record_path,
            {**record, "status": "FAILED", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    write_json_atomic(
        record_path, {**record, "status": "PUBLISHED", "version": qualified}
    )
    return qualified
