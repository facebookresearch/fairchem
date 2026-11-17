"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.regression_suite.artifacts.baselines import (
    BaselineUnavailableError,
    baseline_provenance,
    check_baseline,
    load_baseline,
    point_current_baseline,
)
from fairchem.core.regression_suite.artifacts.sources import (
    add_no_proxy_hosts,
    resolve_baseline,
)
from fairchem.core.regression_suite.artifacts.storage import (
    artifact_root,
    baseline_run_directory,
    checkout_root,
    fragment_path,
    load_json,
    open_artifact,
    pack_baseline,
    report_directory,
    sealed_baseline_path,
)
from fairchem.core.regression_suite.evaluation.calibration import finalize_baseline
from fairchem.core.regression_suite.execution.campaign import (
    active_slurm_job_ids,
    baseline_status,
    candidate_report_directory,
    candidate_run_report,
    launch_baseline,
    launch_candidate,
    load_candidate_launch_manifest,
    manifest_job_ids,
)
from fairchem.core.regression_suite.execution.executors import create_slurm_executor
from fairchem.core.regression_suite.execution.runner import (
    compose_run_config,
    run_suite,
)
from fairchem.core.regression_suite.execution.settings import (
    DeviceUnavailableError,
    load_settings,
    resolve_device,
    resolve_run_on,
    selected_suites,
    slurm_resources,
)
from fairchem.core.regression_suite.execution.smoke import run_fast, run_quick
from fairchem.core.regression_suite.execution.source import (
    call_in_source,
    git_provenance,
    prepend_pythonpath,
)
from fairchem.core.regression_suite.reporting.check_report import (
    build_check_report,
    render_check_summary,
    render_smoke_result,
)
from fairchem.core.regression_suite.reporting.formatting import provenance_lines
from fairchem.core.regression_suite.reporting.report import (
    candidate_report_exit_code,
    verification_report,
)
from fairchem.core.regression_suite.reporting.text_report import render_candidate_report
from fairchem.core.regression_suite.reporting.wandb_report import (
    WandBReportSettings,
    publish_baseline,
    write_check_report,
    write_report,
)
from fairchem.core.regression_suite.spec.profiles import resolve_profile
from fairchem.core.regression_suite.spec.schema import (
    PROFILE_TIERS,
    CommandError,
    RegressionSchemaError,
)
from fairchem.core.regression_suite.suites import STANDARD_SUITES, SUITES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from omegaconf import DictConfig

    from fairchem.core.regression_suite.artifacts.sources import ResolvedBaseline
    from fairchem.core.regression_suite.execution.smoke import SuiteCheck


def _timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def _short_commit(repo_dir: Path) -> str:
    return git_provenance(repo_dir)[0][:8]


def _launched_runs(directory: Path) -> dict[Path, dict[str, Any]]:
    """
    Map each launched run directory under ``directory`` to its latest launch
    manifest, oldest launch first.

    Launch IDs start with their UTC launch time, so they order launches;
    directory times do not, since writing a report bumps them.
    """
    latest: dict[Path, Path] = {}
    for manifest in directory.glob("*/launches/*/launch.json"):
        run = manifest.parents[2]
        if not run.is_symlink():
            latest[run] = max(latest.get(run, manifest), manifest)
    return {
        run: load_json(manifest)
        for run, manifest in sorted(
            latest.items(), key=lambda item: item[1].parent.name
        )
    }


def _wait_for_jobs(job_ids: set[str]) -> None:
    active = active_slurm_job_ids(job_ids)
    if active:
        print(f"Waiting for SLURM jobs {', '.join(sorted(active))}", flush=True)
    while active:
        time.sleep(30)
        active = active_slurm_job_ids(active)


def _publishes(args: argparse.Namespace, settings: DictConfig) -> bool:
    return settings.wandb.auto_publish if args.wandb is None else args.wandb


def _wandb_settings(
    settings: DictConfig,
    *,
    group: str,
    run_name: str | None = None,
    tags: Sequence[str] | None = None,
    enabled: bool = True,
) -> WandBReportSettings | None:
    """
    Return W&B publication settings, grouping every run of one baseline.
    """
    if not enabled:
        return None
    return WandBReportSettings(
        project=str(settings.wandb.project),
        entity=settings.wandb.entity,
        run_name=run_name,
        group=group,
        tags=tuple(tags or ()),
    )


def _resolved_baseline(
    settings: DictConfig, reference: str | None, checkpoint: str | None
) -> tuple[ResolvedBaseline, dict[str, Any]]:
    """
    Resolve ``--baseline``, or the ``baseline`` setting, and load it on the
    submitting host.

    The setting's ``{checkpoint}`` is ``checkpoint``, or the checkpoint setting
    when none is given; a given ``checkpoint`` must be the baseline's.
    """
    wandb = settings.wandb
    name = Path(checkpoint or str(settings.checkpoint)).stem
    resolved = resolve_baseline(
        reference or str(settings.baseline).replace("{checkpoint}", name),
        artifact_root() / "baselines",
        wandb_project="/".join(
            str(part) for part in (wandb.entity, wandb.project) if part
        ),
        wandb_no_proxy_hosts=tuple(wandb.no_proxy_hosts),
    )
    baseline = check_baseline(resolved.path, checkpoint=checkpoint)
    return resolved, baseline


def _print_baseline(resolved: ResolvedBaseline, baseline: Mapping[str, Any]) -> None:
    """
    Print the baseline a check grades against and where it came from.
    """
    lines = provenance_lines(baseline_provenance(baseline, resolved.path))
    lines.append(f"  source {resolved.source}")
    uncovered = [suite for suite in SUITES if suite not in baseline["suites"]]
    if uncovered:
        lines.append(f"  not in this baseline, so not checked: {', '.join(uncovered)}")
    print("\n".join(lines), flush=True)


def _checkpoint(args: argparse.Namespace, settings: DictConfig) -> str:
    """
    Return the checkpoint identity a fast or full check asks for.
    """
    return args.checkpoint_id or args.checkpoint or str(settings.checkpoint)


def _baseline_device(baseline: Mapping[str, Any], run_on: str) -> str:
    """
    Return the baseline's device, which its checks run on, if this launch can.
    """
    device = baseline["device"]
    return resolve_device(device, run_on, source=f"a baseline built on {device}")


def _publication(report_dir: Path) -> str:
    record_path = report_dir / "wandb.json"
    if not record_path.is_file():
        return "not published"
    record = load_json(record_path)
    return record.get("url") or f"W&B {record['status']}: {record.get('error')}"


def _print_launch(kind: str, directory: Path, manifest: Mapping[str, Any]) -> None:
    print(f"{kind.capitalize()}: {directory}")
    print(f"Status: {manifest.get('status')}")
    jobs = sorted(manifest_job_ids(manifest))
    if jobs:
        print(f"Jobs: {', '.join(jobs)}")
    blocked = manifest.get("blocked_suites")
    if blocked:
        print(f"Blocked suites (pinned to other commits): {dict(blocked)}")


def _baseline(args: argparse.Namespace, settings: DictConfig) -> int:
    repo_dir = checkout_root()
    checkpoint = args.checkpoint or str(settings.checkpoint)
    name = args.name or f"{Path(checkpoint).stem}-{_short_commit(repo_dir)}"
    smoke = args.scope == "smoke"
    if smoke and not name.endswith("-smoke"):
        name += "-smoke"
    baselines = artifact_root() / "baselines"
    artifact_dir = baselines / name
    sealed = sealed_baseline_path(artifact_dir)
    # Smoke baselines cover too few cases to be any check's default.
    current = None if smoke else baselines / "current"
    sealed_status = "is sealed" if smoke else "is now the current baseline"
    campaign = "full" if args.full else "standard"
    if args.finalize:
        # Re-finalizing must not drop suites that an earlier --full launch ran.
        if any(
            load_json(manifest)["campaign"] == "full"
            for manifest in artifact_dir.glob("launches/*/launch.json")
        ):
            campaign = "full"
        finalize_baseline(
            artifact_dir, campaign=campaign, overwrite=True, sealed=sealed
        )
        if current is not None:
            point_current_baseline(sealed, current)
        print(f"Finalized; {sealed} {sealed_status}.")
        return 0
    if args.publish:
        if smoke:
            raise CommandError("smoke baselines are never published")
        if not sealed.is_file():
            pack_baseline(artifact_dir, sealed)
        add_no_proxy_hosts(tuple(settings.wandb.no_proxy_hosts))
        version = publish_baseline(
            sealed,
            load_baseline(open_artifact(sealed) / "baseline.json"),
            _wandb_settings(settings, group=name),
        )
        print(f"Published {sealed} as {version}")
        return 0
    manifest = launch_baseline(
        artifact_dir=artifact_dir,
        campaign=campaign,
        checkpoint=checkpoint,
        checkpoint_id=args.checkpoint_id,
        repo_dir=repo_dir,
        run_name=f"regression-baseline-{name}",
        submit=not args.dry_run,
        executor=args.run_on,
        device=resolve_device(args.device, args.run_on),
        scope=args.scope,
        current_link=None if current is None else str(current),
        parallelism=int(settings.slurm.parallelism),
        resources=slurm_resources(settings, gpu=True),
        finalize_resources=slurm_resources(settings, gpu=False),
        wandb=_wandb_settings(
            settings,
            group=name,
            enabled=_publishes(args, settings) and not smoke,
        ),
    )
    _print_launch("baseline", artifact_dir, manifest)
    if args.dry_run:
        return 0
    if manifest.get("status") == "already_complete":
        # Baselines finalized before sealing are packed on first use.
        if not sealed.is_file():
            pack_baseline(artifact_dir, sealed)
        if current is not None:
            point_current_baseline(sealed, current)
        print(f"Already finalized; {sealed} {sealed_status}.")
        return 0
    if args.run_on == "slurm":
        if not args.wait:
            print("Next: fairchem-regression status")
            return 0
        _wait_for_jobs(manifest_job_ids(manifest))
    if sealed.is_file():
        print(f"Finalized; {sealed} {sealed_status}.")
        for report_dir in sorted((artifact_dir / "reports").glob("verification-*")):
            print(f"  {report_dir.name}: {_publication(report_dir)}")
        return 0
    print(f"Not finalized; see the finalize log under {artifact_dir / 'launches'}")
    return 1


def _check_command(
    command: Callable[[argparse.Namespace, DictConfig, TextIO], int],
) -> Callable[[argparse.Namespace, DictConfig], int]:
    """
    Give a check command the stream its report prints to.

    Under ``--format json`` that stream is standard output and everything else
    the command prints goes to standard error, so standard output holds only
    the JSON report.
    """

    @functools.wraps(command)
    def run(args: argparse.Namespace, settings: DictConfig) -> int:
        output = sys.stdout
        with (
            contextlib.redirect_stdout(sys.stderr)
            if args.format == "json"
            else contextlib.nullcontext()
        ):
            return command(args, settings, output)

    return run


def _check(
    kind: str,
    args: argparse.Namespace,
    settings: DictConfig,
    function: Callable[..., list[SuiteCheck]],
    output: TextIO,
    *,
    run_on: str,
    device: str,
    group: str,
    baseline: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> int:
    """
    Run a quick or fast check on ``device`` here or in one blocking SLURM job,
    print each suite's result, and write the check's JSON and HTML reports.

    With W&B publication on, the reports are published as a run in ``group``;
    a failed publication keeps them and is reported.
    """
    run = _timestamp()
    output_dir = artifact_root() / kind / f"{args.profile}-{run}"
    report_dir = report_directory(artifact_root(), args.profile, kind, run)
    if run_on == "local":
        results = function(output_dir=output_dir, device=device, **kwargs)
    else:
        repo_dir = checkout_root()
        executor = create_slurm_executor(
            output_dir / "slurm",
            name=f"regression-{kind}-{args.profile}",
            resources=slurm_resources(settings, gpu=True),
            gpus_per_node=1,
            device=device,
        )
        prepend_pythonpath(str(repo_dir / "src"))
        job = executor.submit(
            functools.partial(
                call_in_source,
                str(repo_dir),
                function,
                output_dir=str(output_dir),
                device=device,
                **kwargs,
            )
        )
        print(f"Submitted {kind} job {job.job_id}; waiting.", flush=True)
        results = job.result()
    report = build_check_report(results, baseline)
    if args.format == "table":
        for result in results:
            print(render_smoke_result(result), end="\n\n")
    if args.format != "json":
        print(render_check_summary(report))
    print(f"Results: {output_dir}")
    wandb = _wandb_settings(
        settings,
        group=group,
        run_name=f"{kind}-{output_dir.name}",
        tags=args.tag,
        enabled=_publishes(args, settings),
    )
    if wandb is not None:
        add_no_proxy_hosts(tuple(settings.wandb.no_proxy_hosts))
    try:
        write_check_report(report, report_dir, wandb)
    except Exception:
        # A failed publication keeps the reports and is recorded in wandb.json,
        # which the W&B line below prints.
        if not (report_dir / "wandb.json").is_file():
            raise
    print(f"JSON report: {report_dir / 'report.json'}")
    print(f"HTML report: {report_dir / 'index.html'}")
    print(f"W&B: {_publication(report_dir)}")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    return report["exit_code"]


@_check_command
def _quick(args: argparse.Namespace, settings: DictConfig, output: TextIO) -> int:
    profile = resolve_profile(args.profile)
    reference = resolve_profile(args.reference)
    suites = selected_suites(
        args.suite, profile, SUITES if args.suite else STANDARD_SUITES
    )
    run_on = resolve_run_on(args.run_on)
    return _check(
        "quick",
        args,
        settings,
        run_quick,
        output,
        run_on=run_on,
        device=resolve_device(args.device, run_on),
        group="quick",
        profile=profile,
        reference_profile=reference,
        suites=selected_suites(args.suite, reference, suites),
        checkpoint=args.checkpoint or str(settings.checkpoint),
        checkpoint_id=args.checkpoint_id,
    )


@_check_command
def _fast(args: argparse.Namespace, settings: DictConfig, output: TextIO) -> int:
    profile = resolve_profile(args.profile)
    resolved, baseline = _resolved_baseline(
        settings, args.baseline, _checkpoint(args, settings)
    )
    _print_baseline(resolved, baseline)
    default = SUITES if args.suite else STANDARD_SUITES
    suites = selected_suites(
        args.suite, profile, [suite for suite in default if suite in baseline["suites"]]
    )
    run_on = resolve_run_on(args.run_on)
    return _check(
        "fast",
        args,
        settings,
        run_fast,
        output,
        run_on=run_on,
        device=_baseline_device(baseline, run_on),
        group=baseline_run_directory(resolved.path).name,
        baseline=baseline_provenance(baseline, resolved.path, suites),
        baseline_dir=resolved.path,
        profile=profile,
        suites=suites,
        checkpoint=args.checkpoint,
        checkpoint_id=args.checkpoint_id,
    )


@_check_command
def _full(args: argparse.Namespace, settings: DictConfig, output: TextIO) -> int:
    profile = resolve_profile(args.profile)
    resolved, baseline = _resolved_baseline(
        settings, args.baseline, _checkpoint(args, settings)
    )
    _print_baseline(resolved, baseline)
    repo_dir = checkout_root()
    run_on = resolve_run_on(args.run_on)
    _baseline_device(baseline, run_on)
    run = f"{_short_commit(repo_dir)}-{_timestamp()}"
    name = f"{args.profile}-{run}"
    candidate_dir = artifact_root() / "candidates" / name
    report_dir = report_directory(artifact_root(), args.profile, "full", run)
    try:
        manifest = launch_candidate(
            baseline_dir=resolved.path,
            baseline_source=resolved.source,
            candidate_dir=candidate_dir,
            report_dir=report_dir,
            checkpoint=args.checkpoint,
            checkpoint_id=args.checkpoint_id,
            repo_dir=repo_dir,
            profile=profile,
            suites=args.suite,
            run_name=f"regression-candidate-{name}",
            submit=not args.dry_run,
            executor=run_on,
            parallelism=int(settings.slurm.parallelism),
            resources=slurm_resources(settings, gpu=True),
            finalize_resources=slurm_resources(settings, gpu=False),
            wandb=_wandb_settings(
                settings,
                group=baseline_run_directory(resolved.path).name,
                run_name=name,
                tags=args.tag,
                enabled=_publishes(args, settings),
            ),
        )
    except CommandError as error:
        # Local launches raise failed tasks after the report marks them.
        if not (report_dir / "report.json").is_file():
            raise
        print(f"Error: {error}", file=sys.stderr)
        manifest = load_candidate_launch_manifest(candidate_dir)
    _print_launch("candidate", candidate_dir, manifest)
    if args.dry_run:
        return 0
    if run_on == "slurm":
        if not args.wait:
            print("The report follows the jobs; see: fairchem-regression status")
            return 0
        _wait_for_jobs(manifest_job_ids(manifest))
    if not (report_dir / "report.json").is_file():
        print(f"No report; see the finalize log under {candidate_dir / 'launches'}")
        return 1
    report = load_json(report_dir / "report.json")
    if args.format != "json":
        print(render_candidate_report(report, summary=args.format == "summary"))
    print(f"JSON report: {report_dir / 'report.json'}")
    print(f"HTML report: {report_dir / 'index.html'}\nW&B: {_publication(report_dir)}")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    return candidate_report_exit_code(report)


def _candidate_dir(configured: str | None) -> Path:
    candidates = artifact_root() / "candidates"
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else candidates / configured
    # Dry runs record a planned launch but have nothing to report.
    runs = [
        run
        for run, manifest in _launched_runs(candidates).items()
        if manifest["status"] != "planned"
    ]
    if not runs:
        raise CommandError(f"no candidate runs in {candidates}")
    return runs[-1]


def _publish_report(
    report: Mapping[str, Any],
    report_dir: Path,
    wandb: WandBReportSettings | None,
    settings: DictConfig,
    output_format: str,
    output: TextIO,
) -> int:
    if wandb is not None:
        add_no_proxy_hosts(tuple(settings.wandb.no_proxy_hosts))
    url = write_report(report, report_dir, wandb)
    if output_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    else:
        rendered = render_candidate_report(report, summary=output_format == "summary")
        print(rendered, file=output)
    print(f"JSON report: {report_dir / 'report.json'}", file=sys.stderr)
    print(f"HTML report: {report_dir / 'index.html'}", file=sys.stderr)
    if url:
        print(f"W&B report: {url}", file=sys.stderr)
    return candidate_report_exit_code(report)


def _report(args: argparse.Namespace, settings: DictConfig) -> int:
    """
    Regenerate reports; ``--wandb`` publishes each as a new W&B run.

    The rendered reports go to ``--output`` or standard output, and anything
    printed while building them goes to standard error. Messages printed while
    ``fairchem.core`` is imported, such as NVIDIA Warp's banner, precede this,
    so only ``--output`` guarantees a file that holds nothing else.
    """
    with contextlib.ExitStack() as stack:
        output = (
            sys.stdout
            if args.output is None
            else stack.enter_context(args.output.open("w", encoding="utf-8"))
        )
        stack.enter_context(contextlib.redirect_stdout(sys.stderr))
        return _build_reports(args, settings, output)


def _build_reports(
    args: argparse.Namespace, settings: DictConfig, output: TextIO
) -> int:
    if args.verification:
        baseline_dir = _resolved_baseline(settings, args.baseline, None)[0].path
        profiles = (
            [profile for profile, _tier in PROFILE_TIERS]
            if args.verification == "all"
            else [resolve_profile(args.verification)]
        )
        exit_codes = []
        for profile in profiles:
            report, report_dir = verification_report(baseline_dir, profile)
            name = baseline_run_directory(baseline_dir).name
            wandb = _wandb_settings(
                settings,
                group=name,
                run_name=f"{name}-{report_dir.name}",
                tags=args.tag,
                enabled=args.wandb,
            )
            exit_codes.append(
                _publish_report(
                    report, report_dir, wandb, settings, args.format, output
                )
            )
        return max(exit_codes)
    candidate_dir = _candidate_dir(args.run)
    manifest = load_candidate_launch_manifest(candidate_dir)
    wandb = _wandb_settings(
        settings,
        group=baseline_run_directory(manifest["baseline_dir"]).name,
        run_name=candidate_dir.name,
        tags=args.tag,
        enabled=args.wandb,
    )
    return _publish_report(
        candidate_run_report(candidate_dir),
        candidate_report_directory(candidate_dir, manifest),
        wandb,
        settings,
        args.format,
        output,
    )


def _status(_args: argparse.Namespace, settings: DictConfig) -> int:
    root = artifact_root()
    print(f"Root: {root}")
    try:
        baseline_path = resolve_baseline("current", root / "baselines").path
    except BaselineUnavailableError:
        print("Current baseline: none")
    else:
        baseline = load_baseline(open_artifact(baseline_path) / "baseline.json")
        lines = provenance_lines(baseline_provenance(baseline, baseline_path))
        lines[0] = lines[0].replace("Baseline:", "Current baseline:", 1)
        print("\n".join(lines))
        run_directory = baseline_run_directory(baseline_path)
        for report_dir in sorted((run_directory / "reports").glob("verification-*")):
            print(f"  {report_dir.name}: {_publication(report_dir)}")
        record = run_directory / "wandb.json"
        if record.is_file():
            published = load_json(record)
            print(f"  W&B baseline: {published.get('version') or published}")
    baselines = list(_launched_runs(root / "baselines").items())
    for path, launch in baselines[-3:]:
        # Launches recorded before scopes existed were full.
        status = baseline_status(path, launch["campaign"], launch.get("scope", "full"))
        counts = {
            role: f"{sum(item['state'] == 'complete' for item in states.values())}"
            f"/{len(states)}"
            for role, states in status["fragments"].items()
        }
        finalized = (
            "sealed"
            if sealed_baseline_path(path).is_file()
            else "finalized"
            if (path / "baseline.json").is_file()
            else "running"
        )
        print(
            f"  baseline {path.name}: {finalized}; references "
            f"{counts['reference']}, calibration {counts['calibration']}, "
            f"verification {counts['verification']}"
        )
    candidates = list(_launched_runs(root / "candidates").items())
    for path, manifest in candidates[-5:]:
        suites = manifest["selected_suites"]
        done = sum(
            fragment_path(path, suite, manifest["profile"], "comparison").is_dir()
            for suite in suites
        )
        report_dir = candidate_report_directory(path, manifest)
        report = (
            f"report ready at {report_dir}, {_publication(report_dir)}"
            if (report_dir / "report.json").is_file()
            else "no report yet"
        )
        print(f"  candidate {path.name}: {done}/{len(suites)} suites, {report}")
    return 0


def _run(args: argparse.Namespace, _settings: DictConfig) -> int:
    identity = run_suite(compose_run_config(args.overrides))["identity"]
    print(json.dumps(identity, indent=2, sort_keys=True, default=str))
    return 0


_DEVICES = ("auto", "cpu", "cuda")
_DEVICE_HELP = (
    "where tasks run (default auto: cuda on SLURM, or locally when a GPU is visible)"
)
_BASELINE_HELP = (
    "baseline reference: hf:ORG/REPO/PATH[@REVISION], "
    "wandb:[ENTITY/PROJECT/]NAME[:ALIAS], a local name or path, or current "
    "(default: the baseline setting, the checkpoint's published baseline)"
)


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "summary", "json"),
        default="table",
        help="detailed tables, a compact overview, or the JSON report",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairchem-regression",
        description=(
            "Numerical regression suite. Artifacts live in regression-artifacts/ "
            "of this checkout; see docs/core/common_tasks/regression_suite.md."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser(
        "baseline",
        help="build or resume the FP64 baseline and its FP64/FP32/TF32 calibration",
    )
    baseline.add_argument("--checkpoint", help="checkpoint name or path")
    baseline.add_argument("--checkpoint-id", help="identity for a checkpoint path")
    baseline.add_argument("--name", help="baseline name (default: checkpoint-commit)")
    baseline.add_argument("--full", action="store_true", help="include NVE MD")
    baseline.add_argument(
        "--scope",
        choices=("full", "smoke"),
        default="full",
        help="smoke runs only each suite's smoke cases, under a -smoke name that "
        "never becomes current or is published",
    )
    baseline.add_argument("--run-on", choices=("slurm", "local"), default="slurm")
    baseline.add_argument(
        "--device", choices=_DEVICES, default="auto", help=_DEVICE_HELP
    )
    baseline.add_argument("--wait", action="store_true", help="wait for SLURM jobs")
    baseline.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        help="publish the baseline and its reports to W&B (default: the "
        "wandb.auto_publish setting)",
    )
    baseline.add_argument(
        "--publish", action="store_true", help="only upload the sealed baseline to W&B"
    )
    baseline.add_argument(
        "--finalize", action="store_true", help="only re-finalize existing runs"
    )
    baseline.add_argument("--dry-run", action="store_true")

    checks = {}
    for command, help_text in (
        ("quick", "compare a profile with a reference profile on smoke cases"),
        ("fast", "qualify a profile against the baseline on smoke cases"),
        ("full", "run a profile on every supported suite and qualify it"),
    ):
        checks[command] = commands.add_parser(command, help=help_text)
        checks[command].add_argument(
            "profile", help="umas-fast, tf32, fp32, fp64, or a profile name"
        )
        checks[command].add_argument(
            "--suite", action="append", help="limit to a suite; repeatable"
        )
        checks[command].add_argument(
            "--run-on", choices=("auto", "slurm", "local"), default="auto"
        )
    checks["quick"].add_argument(
        "--reference",
        default="fp32",
        help="profile to compare with (default: fp32)",
    )
    for command in ("fast", "full"):
        checks[command].add_argument("--baseline", help=_BASELINE_HELP)
        checks[command].add_argument(
            "--checkpoint", help="checkpoint name or path (default: the setting)"
        )
        checks[command].add_argument(
            "--checkpoint-id", help="baseline checkpoint ID for a checkpoint path"
        )
    checks["quick"].add_argument(
        "--checkpoint", help="checkpoint name or path (default: the setting)"
    )
    checks["quick"].add_argument(
        "--checkpoint-id", help="identity for a checkpoint path"
    )
    checks["quick"].add_argument(
        "--device", choices=_DEVICES, default="auto", help=_DEVICE_HELP
    )
    for check in checks.values():
        check.add_argument(
            "--wandb",
            action=argparse.BooleanOptionalAction,
            help="publish the report to W&B (default: the wandb.auto_publish "
            "setting); JSON and HTML reports are always written",
        )
        check.add_argument("--tag", action="append", help="W&B tag; repeatable")
        _add_format(check)
    full = checks["full"]
    full.add_argument("--wait", action="store_true", help="wait and print the report")
    full.add_argument("--dry-run", action="store_true")

    report = commands.add_parser("report", help="render a candidate report")
    report.add_argument(
        "run", nargs="?", help="candidate name or path (default: latest)"
    )
    report.add_argument(
        "--verification",
        nargs="?",
        const="all",
        metavar="PROFILE",
        help="report the baseline's held-out verification runs: all, or one of "
        "fp64, fp32, and tf32",
    )
    report.add_argument("--baseline", help=f"for --verification, {_BASELINE_HELP}")
    report.add_argument(
        "--wandb", action="store_true", help="publish each report as a new W&B run"
    )
    report.add_argument("--tag", action="append", help="W&B tag; repeatable")
    _add_format(report)
    report.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="write the rendered report to PATH instead of standard output",
    )

    commands.add_parser("status", help="show the current baseline and recent runs")

    run = commands.add_parser("run", help="run one suite task (debugging)")
    run.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    return parser


_COMMANDS = {
    "baseline": _baseline,
    "quick": _quick,
    "fast": _fast,
    "full": _full,
    "report": _report,
    "status": _status,
    "run": _run,
}


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the regression-suite command-line interface.
    """
    args = _parser().parse_args(argv)
    # Jobs run in the source checkout, so a relative path must not reach them.
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint and checkpoint not in pretrained_mlip.available_models:
        args.checkpoint = str(Path(checkpoint).expanduser().resolve())
    try:
        return _COMMANDS[args.command](args, load_settings())
    except DeviceUnavailableError as error:
        print(f"{error}; nothing was started.", file=sys.stderr)
        return 1
    except BaselineUnavailableError as error:
        print(f"Baseline unavailable; nothing was started.\n{error}", file=sys.stderr)
        return 1
    except (CommandError, RegressionSchemaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
