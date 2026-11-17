"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.artifacts.baselines import (
    check_baseline,
    load_baseline,
    point_current_baseline,
)
from fairchem.core.regression_suite.artifacts.storage import (
    fragment_path,
    load_json,
    load_reference_manifest,
    sealed_baseline_path,
    write_json_atomic,
)
from fairchem.core.regression_suite.evaluation.calibration import finalize_baseline
from fairchem.core.regression_suite.evaluation.identity import (
    check_baseline_plans,
    run_scope,
)
from fairchem.core.regression_suite.execution.executors import EXECUTORS, Rerunnable
from fairchem.core.regression_suite.execution.runner import (
    ROLES,
    merge_suite_shards,
    resolve_baseline_checkpoint,
    resolve_checkpoint,
    run_suite,
    task_config,
)
from fairchem.core.regression_suite.execution.settings import selected_suites
from fairchem.core.regression_suite.execution.source import (
    call_in_source,
    clean_commit,
    create_source_snapshot,
    prepend_pythonpath,
)
from fairchem.core.regression_suite.reporting.report import (
    build_candidate_report,
    verification_report,
)
from fairchem.core.regression_suite.reporting.wandb_report import (
    publish_baseline,
    write_report,
)
from fairchem.core.regression_suite.spec.profiles import profile_config
from fairchem.core.regression_suite.spec.schema import (
    PROFILE_TIERS,
    CommandError,
    RegressionSchemaError,
)
from fairchem.core.regression_suite.suites import (
    CAMPAIGN_SUITES,
    SUITES,
    load_suite,
)
from fairchem.core.regression_suite.suites.base import suite_contract_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fairchem.core.launchers.api import SlurmConfig
    from fairchem.core.regression_suite.reporting.wandb_report import (
        WandBReportSettings,
    )


# Every baseline run uses this model seed; candidates use their baseline's.
_BASELINE_SEED = 42


@dataclass(frozen=True)
class SuiteTask:
    """
    One suite/profile/role execution, optionally restricted to one plan case.
    """

    suite: str
    profile: str
    role: str
    shard_index: int | None = None
    shard_count: int = 1


def baseline_tasks(role: str, suites: Sequence[str]) -> list[SuiteTask]:
    """
    Return the FP64 reference tasks or the per-profile tasks of one role.
    """
    profiles = (
        ["fp64-eager"]
        if role == "reference"
        else [profile for profile, _tier in PROFILE_TIERS]
    )
    return [SuiteTask(suite, profile, role) for profile in profiles for suite in suites]


def _slurm_run_id(role: str) -> str:
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    # Local tasks inherit their allocation's job ID, which they all share.
    suffix = f"{array_job}-{array_task}" if array_job and array_task else None
    return f"{role}-{suffix or uuid.uuid4().hex}"


def _run_task(
    task: SuiteTask,
    *,
    artifact_dir: str,
    checkpoint: str,
    checkpoint_id: str,
    seed: int,
    device: str,
    scope: str,
    staging_dir: str,
    merge: bool = False,
    baseline: str | None = None,
    baseline_id: str | None = None,
) -> str:
    """
    Run one task, or merge one run's case shards, and return its run ID.

    Candidate tasks first confirm that their baseline is readable from this
    node, so an inaccessible baseline stops the run before any GPU work.
    """
    if baseline is not None:
        check_baseline(baseline, baseline_id)
    shard_root = Path(staging_dir) / task.role / task.suite / task.profile
    values = {
        "checkpoint": checkpoint,
        "checkpoint_id": checkpoint_id,
        "output_dir": artifact_dir,
        "run_id": _slurm_run_id(task.role),
        "device": device,
        "scope": scope,
        "seed": seed,
    }
    if task.shard_index is not None:
        values |= {
            "+case_shard_index": task.shard_index,
            "+case_shard_count": task.shard_count,
            "+case_shard_output": str(shard_root / f"{task.shard_index:04d}"),
        }
    config = task_config(task.suite, task.role, task.profile, values)
    if merge:
        result = merge_suite_shards(config, shard_root, task.shard_count)
    else:
        result = run_suite(config)
    return str(result["identity"]["run_id"])


def _finalize_baseline_job(
    artifact_dir: str,
    *,
    campaign: str,
    current_link: str | None,
    wandb: WandBReportSettings | None = None,
) -> str:
    sealed = sealed_baseline_path(artifact_dir)
    baseline = finalize_baseline(artifact_dir, campaign=campaign, sealed=sealed)
    if current_link:
        point_current_baseline(sealed, current_link)
    for profile, _tier in PROFILE_TIERS:
        report, report_dir = verification_report(sealed, profile)
        settings = (
            None
            if wandb is None
            else dataclasses.replace(
                wandb, run_name=f"{Path(artifact_dir).name}-{report_dir.name}"
            )
        )
        # The baseline is final; a failed report is recorded and republishable.
        try:
            write_report(report, report_dir, settings)
        except Exception as error:
            print(f"{report_dir.name} report failed: {error}", flush=True)
    if wandb is not None:
        try:
            print(f"Published {publish_baseline(sealed, baseline, wandb)}", flush=True)
        except Exception as error:
            print(f"baseline upload failed: {error}", flush=True)
    return str(baseline["baseline_id"])


def candidate_run_report(candidate_dir: str | Path) -> dict[str, Any]:
    """
    Build the report of a launched candidate from its launch manifest.
    """
    manifest = load_candidate_launch_manifest(candidate_dir)
    selected = list(manifest["selected_suites"])
    return build_candidate_report(
        manifest["baseline_dir"],
        candidate_dir,
        candidate_profile=manifest["profile_config"],
        suites=selected if selected != manifest["supported_suites"] else None,
    )


def candidate_report_directory(
    candidate_dir: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """
    Return where a launched candidate's report is written.
    """
    # Candidates launched before reports/ existed kept their report inside.
    return Path(manifest.get("report_dir") or Path(candidate_dir) / "report")


def _finalize_candidate_job(
    candidate_dir: str, *, report_dir: str, wandb: WandBReportSettings | None
) -> str | None:
    return write_report(candidate_run_report(candidate_dir), report_dir, wandb)


@dataclass(frozen=True)
class ArrayPhase:
    """
    One array of independent tasks, submitted after its dependency phase.
    """

    name: str
    tasks: tuple[SuiteTask, ...]
    dependency: str | None = None
    gpus_per_node: int = 1


def _phases(role: str, tasks: Sequence[SuiteTask], scope: str) -> list[ArrayPhase]:
    """
    Split opted-in suites into one GPU task per plan case plus a CPU merge;
    smoke-scoped runs are never split.
    """
    sharded, regular, merges = [], [], []
    for task in tasks:
        suite = load_suite(task.suite)
        if scope == "smoke" or not suite.slurm_case_parallel:
            regular.append(task)
            continue
        count = len(suite.plan().cases)
        merges.append(dataclasses.replace(task, shard_count=count))
        sharded.extend(
            dataclasses.replace(task, shard_index=index, shard_count=count)
            for index in range(count)
        )
    return [
        ArrayPhase(role, (*sharded, *regular)),
        ArrayPhase(f"{role}_merge", tuple(merges), dependency=role, gpus_per_node=0),
    ]


def _fragment_state(path: Path, task: SuiteTask, scope: str) -> dict[str, Any]:
    """
    Inspect one saved run, reusable only by launches of its scope;
    finalization validates its values against the FP64 reference.
    """
    if not path.exists():
        return {"state": "missing"}
    try:
        identity = load_reference_manifest(path)["identity"]
        suite = load_suite(task.suite)
        expected = {
            "suite": task.suite,
            "profile": task.profile,
            "role": task.role,
            "plan_hash": suite.plan().plan_hash,
            "suite_contract_hash": suite_contract_hash(suite),
            "scope": scope,
        }
        actual = {**identity, "scope": run_scope(identity)}
        differing = sorted(
            key for key, value in expected.items() if actual.get(key) != value
        )
        if differing:
            raise RegressionSchemaError(
                f"{differing} differ from this launch; remove {path} to recompute it"
            )
    except (KeyError, OSError, ValueError) as error:
        return {"state": "invalid", "error": str(error)}
    return {"state": "complete", "identity": identity}


def baseline_status(
    artifact_dir: str | Path, campaign: str, scope: str = "full"
) -> dict[str, Any]:
    """
    Inspect a baseline's saved runs without submitting or running inference.
    """
    root = Path(artifact_dir).expanduser().resolve()
    suites = CAMPAIGN_SUITES[campaign]
    fragments = {
        role: {
            f"{task.suite}/{task.profile}": _fragment_state(
                fragment_path(root, task.suite, task.profile, task.role),
                task,
                scope,
            )
            for task in baseline_tasks(role, suites)
        }
        for role in ROLES
    }
    baseline: dict[str, Any] = {"state": "missing"}
    if (root / "baseline.json").exists():
        try:
            payload = load_baseline(root / "baseline.json")
            baseline = {
                "state": (
                    "complete" if set(suites) <= set(payload["suites"]) else "partial"
                ),
                "baseline_id": payload["baseline_id"],
                "checkpoint": payload["checkpoint"],
                "seed": payload["seed"],
                "device": payload["device"],
                "scope": payload["scope"],
            }
        except (KeyError, OSError, ValueError) as error:
            baseline = {"state": "invalid", "error": str(error)}
    return {"fragments": fragments, "baseline": baseline}


def manifest_job_ids(manifest: Mapping[str, Any]) -> set[str]:
    """
    Return the scheduler job IDs recorded in a launch manifest.
    """
    return {
        str(phase[key])
        for phase in manifest.get("phases", {}).values()
        for key in ("array_job_id", "job_id")
        if phase.get(key)
    }


def active_slurm_job_ids(job_ids: set[str]) -> set[str]:
    job_ids = {job_id for job_id in job_ids if job_id.isdigit()}
    if not job_ids:
        return set()
    if shutil.which("squeue") is None:
        raise CommandError("squeue is required to check SLURM jobs")
    output = subprocess.run(
        ["squeue", "--noheader", f"--user={os.getuid()}", "--format=%A"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line.strip() for line in output.splitlines()} & job_ids


def _launch(
    root: Path,
    phases: Sequence[ArrayPhase],
    fields: dict[str, Any],
    *,
    submit: bool,
    repo_dir: Path,
    resources: SlurmConfig,
    parallelism: int,
    finalize: Callable[[], str] | None = None,
    finalize_resources: SlurmConfig | None = None,
    finalize_after_failures: bool = False,
) -> dict[str, Any]:
    """
    Record a launch and, unless it is a dry run, snapshot the source and submit.

    Non-empty phases are submitted in order. SLURM phases wait on their
    dependency with ``afterok`` and the finalizer waits on every submitted
    array; local phases block, so order suffices. With
    ``finalize_after_failures`` the finalizer also runs after failed tasks
    (``afterany``; local launches finalize, then raise the failure). A
    timed-out phase task reruns from the start, which is safe because it
    publishes its records by one atomic rename at the end; the finalizer is
    not rerun.
    """
    executor = fields["executor"]
    if submit and executor == "slurm" and shutil.which("sbatch") is None:
        raise CommandError("sbatch is required for a submitted launch")
    now = datetime.datetime.now(datetime.UTC)
    launch_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    launch_dir = root / "launches" / launch_id
    launch_dir.mkdir(parents=True)
    manifest_path = launch_dir / "launch.json"
    manifest = {
        "launch_id": launch_id,
        "created_at": now.isoformat(),
        "source_dir": None,
        **fields,
        "phases": {
            phase.name: {
                "tasks": [dataclasses.asdict(task) for task in phase.tasks],
                "array_job_id": None,
                "dependency": None,
            }
            for phase in phases
        },
    }
    if finalize is not None:
        manifest["phases"]["finalize"] = {"job_id": None, "dependency": None}
    if not submit:
        write_json_atomic(manifest_path, manifest)
        return manifest
    source_dir = create_source_snapshot(
        repo_dir, fields["git_commit"], launch_dir / "source"
    )
    manifest["source_dir"] = str(source_dir)
    prepend_pythonpath(str(source_dir / "src"))
    job = functools.partial(
        call_in_source,
        str(source_dir),
        _run_task,
        artifact_dir=str(root),
        checkpoint=fields["checkpoint"],
        checkpoint_id=fields["checkpoint_id"],
        seed=int(fields["seed"]),
        device=fields["device"],
        scope=fields["scope"],
        staging_dir=str(launch_dir / "staging"),
        **(
            {"baseline": fields["baseline_dir"], "baseline_id": fields["baseline_id"]}
            if "baseline_dir" in fields
            else {}
        ),
    )
    create_executor = EXECUTORS[executor]
    job_ids: dict[str, str] = {}
    manifest["status"] = "submitting"
    try:
        failure = None
        for phase in phases:
            if not phase.tasks:
                continue
            dependency = job_ids.get(phase.dependency)
            phase_executor = create_executor(
                launch_dir / phase.name,
                name=f"{fields['run_name']}-{phase.name}",
                resources=resources,
                gpus_per_node=phase.gpus_per_node,
                device=fields["device"],
                array_parallelism=min(parallelism, len(phase.tasks)),
                dependency=dependency,
            )
            phase_job = Rerunnable(job, merge=phase.name.endswith("_merge"))
            try:
                jobs = phase_executor.map_array(phase_job, phase.tasks)
            except CommandError as error:
                if not finalize_after_failures:
                    raise
                failure = error
                break
            job_ids[phase.name] = str(jobs[0].job_id).rsplit("_", 1)[0]
            manifest["phases"][phase.name].update(
                array_job_id=job_ids[phase.name], dependency=dependency
            )
            write_json_atomic(manifest_path, manifest)
        if finalize is not None:
            dependency = ":".join(job_ids.values()) or None
            finalize_executor = create_executor(
                launch_dir / "finalize",
                name=f"{fields['run_name']}-finalize",
                resources=finalize_resources,
                gpus_per_node=0,
                dependency=dependency,
                after="afterany" if finalize_after_failures else "afterok",
            )
            finalize_job = functools.partial(call_in_source, str(source_dir), finalize)
            manifest["phases"]["finalize"] = {
                "job_id": str(finalize_executor.submit(finalize_job).job_id),
                "dependency": dependency,
            }
        if failure is not None:
            raise failure
        manifest["status"] = "submitted"
    except Exception as error:
        manifest["status"] = "submission_failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json_atomic(manifest_path, manifest)
    return manifest


def launch_baseline(
    *,
    artifact_dir: str | Path,
    repo_dir: Path,
    checkpoint: str,
    resources: SlurmConfig,
    finalize_resources: SlurmConfig,
    run_name: str,
    campaign: str = "standard",
    checkpoint_id: str | None = None,
    executor: str = "slurm",
    device: str = "cuda",
    scope: str = "full",
    parallelism: int = 64,
    submit: bool = True,
    current_link: str | None = None,
    wandb: WandBReportSettings | None = None,
) -> dict[str, Any]:
    """
    Submit or dry-run every missing task of a resumable baseline campaign.

    Completed runs are never resubmitted. A suite whose completed runs come
    from another commit is pinned to it: this launch skips the suite, and the
    finalizer is withheld while a pinned suite still has missing work. The
    finalizer writes the verification reports and, given ``wandb``, publishes
    them. With ``executor="local"`` every phase runs on this machine before
    returning. Every task runs on ``device``, and a ``smoke`` scope runs
    only each suite's smoke cases.
    """
    commit = clean_commit(repo_dir, "baseline")
    checkpoint, checkpoint_id = resolve_checkpoint(checkpoint, checkpoint_id)
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = baseline_status(root, campaign, scope)
    baseline = status["baseline"]
    if baseline["state"] in {"complete", "partial"}:
        requested = {
            "checkpoint": checkpoint_id,
            "seed": _BASELINE_SEED,
            "device": device,
            "scope": scope,
        }
        mismatches = [key for key, value in requested.items() if baseline[key] != value]
        if mismatches:
            raise RegressionSchemaError(
                "completed baseline uses different launch provenance or policy: "
                f"{mismatches}"
            )
        if baseline["state"] == "complete":
            return {
                "status": "already_complete",
                "artifact_dir": str(root),
                "baseline_id": baseline["baseline_id"],
            }
    fragments = {
        f"{role}/{key}": state
        for role, states in status["fragments"].items()
        for key, state in states.items()
    }
    invalid = [key for key, state in fragments.items() if state["state"] == "invalid"]
    if baseline["state"] == "invalid":
        invalid.append("baseline.json")
    if invalid:
        raise RegressionSchemaError(
            f"invalid artifacts must be resolved before resume: {invalid}"
        )
    mismatched = []
    suite_commits: dict[str, set[Any]] = {}
    for key, state in fragments.items():
        if state["state"] != "complete":
            continue
        identity = state["identity"]
        if (
            identity.get("git_dirty") is not False
            or identity.get("checkpoint") != checkpoint_id
            or identity.get("seed") != _BASELINE_SEED
            or identity.get("device") != device
        ):
            mismatched.append(key)
        suite_commits.setdefault(key.split("/")[1], set()).add(identity["git_commit"])
    if mismatched:
        raise RegressionSchemaError(
            f"completed fragments use different launch provenance: {mismatched}"
        )
    mixed = sorted(
        suite for suite, commits in suite_commits.items() if len(commits) > 1
    )
    if mixed:
        raise RegressionSchemaError(f"suite fragments mix source commits: {mixed}")
    pinned_suites = {
        suite: next(iter(commits))
        for suite, commits in sorted(suite_commits.items())
        if commits != {commit}
    }
    missing = {key for key, state in fragments.items() if state["state"] == "missing"}
    blocked_suites = {
        suite: pinned
        for suite, pinned in pinned_suites.items()
        if any(key.split("/")[1] == suite for key in missing)
    }
    phases = [
        phase
        for role in ROLES
        for phase in _phases(
            role,
            [
                task
                for task in baseline_tasks(role, CAMPAIGN_SUITES[campaign])
                if f"{role}/{task.suite}/{task.profile}" in missing
                and task.suite not in pinned_suites
            ],
            scope,
        )
    ]
    if submit:
        job_ids = {
            job_id
            for path in root.glob("launches/*/launch.json")
            for job_id in manifest_job_ids(load_json(path))
        }
        active = active_slurm_job_ids(job_ids)
        if active:
            raise CommandError(
                "an earlier baseline launch is still active; refusing duplicate "
                f"submission for SLURM jobs {sorted(active)}"
            )
    finalize = not blocked_suites
    has_work = finalize or any(phase.tasks for phase in phases)
    return _launch(
        root,
        phases,
        {
            "run_name": run_name,
            "campaign": campaign,
            "status": "planned" if has_work else "blocked",
            "executor": executor,
            "device": device,
            "scope": scope,
            "git_commit": commit,
            "pinned_suites": pinned_suites,
            "blocked_suites": blocked_suites,
            "artifact_dir": str(root),
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "seed": _BASELINE_SEED,
        },
        submit=submit and has_work,
        repo_dir=repo_dir,
        resources=resources,
        parallelism=parallelism,
        finalize=(
            functools.partial(
                _finalize_baseline_job,
                str(root),
                campaign=campaign,
                current_link=current_link,
                wandb=wandb,
            )
            if finalize
            else None
        ),
        finalize_resources=finalize_resources,
    )


def load_candidate_launch_manifest(candidate_dir: str | Path) -> dict[str, Any]:
    """
    Load the launch manifest of a candidate directory.
    """
    manifests = sorted(Path(candidate_dir).expanduser().glob("launches/*/launch.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"expected one launch manifest in {candidate_dir}")
    return load_json(manifests[0])


def launch_candidate(
    *,
    baseline_dir: str | Path,
    baseline_source: str,
    candidate_dir: str | Path,
    report_dir: str | Path,
    repo_dir: Path,
    profile: str,
    resources: SlurmConfig,
    finalize_resources: SlurmConfig,
    run_name: str,
    suites: Sequence[str] | None = None,
    checkpoint: str | None = None,
    checkpoint_id: str | None = None,
    executor: str = "slurm",
    parallelism: int = 64,
    submit: bool = True,
    wandb: WandBReportSettings | None = None,
) -> dict[str, Any]:
    """
    Submit or dry-run one candidate profile on the selected baseline suites.

    A finalizer writes the report to ``report_dir`` after every task, failed
    ones included, and, given ``wandb``, publishes it. With ``executor="local"`` every task runs on this machine
    before returning. Tasks run on the baseline's device and cases. The
    manifest records ``baseline_source``, the pinned reference of the baseline.
    """
    commit = clean_commit(repo_dir, "candidate")
    baseline_root = Path(baseline_dir).expanduser().resolve()
    baseline = check_baseline(baseline_root)
    config = profile_config(profile)
    baseline_suites = tuple(suite for suite in SUITES if suite in baseline["suites"])
    selected = selected_suites(suites, profile, baseline_suites)
    checkpoint, checkpoint_id = resolve_baseline_checkpoint(
        baseline, checkpoint, checkpoint_id
    )
    check_baseline_plans(baseline, selected, config)
    label = str(config.label)
    root = Path(candidate_dir).expanduser().resolve()
    reports = Path(report_dir).expanduser().resolve()
    return _launch(
        root,
        _phases(
            "comparison",
            [SuiteTask(suite, label, "comparison") for suite in selected],
            baseline["scope"],
        ),
        {
            "run_name": run_name,
            "status": "planned",
            "executor": executor,
            "device": baseline["device"],
            "scope": baseline["scope"],
            "git_commit": commit,
            "baseline_dir": str(baseline_root),
            "baseline_id": baseline["baseline_id"],
            "baseline_source": baseline_source,
            "candidate_dir": str(root),
            "report_dir": str(reports),
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "seed": baseline["seed"],
            "profile": label,
            "profile_config": profile,
            "selected_suites": list(selected),
            "supported_suites": list(selected_suites(None, profile, baseline_suites)),
        },
        submit=submit,
        repo_dir=repo_dir,
        resources=resources,
        parallelism=parallelism,
        finalize=functools.partial(
            _finalize_candidate_job, str(root), report_dir=str(reports), wandb=wandb
        ),
        finalize_resources=finalize_resources,
        # The report marks suites without records as failed.
        finalize_after_failures=True,
    )
