"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from submitit.core.job_environment import SignalHandler
from submitit.core.utils import DelayedSubmission, JobPaths, UncompletedJobError

from fairchem.core.launchers.api import SlurmConfig
from fairchem.core.regression_suite.execution.campaign import (
    baseline_status,
    launch_baseline,
    launch_candidate,
)
from fairchem.core.regression_suite.spec.schema import CommandError
from fairchem.core.regression_suite.suites import SUITES

_CAMPAIGN = "fairchem.core.regression_suite.execution.campaign"
_REPO = Path(__file__).resolve().parents[3]
_NVE_SHARDS = 200


class FakeExecutor:
    """
    Record submitted work and return SLURM-style job IDs.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.submitted = None
        self.tasks = None

    def map_array(self, callable_, tasks):
        self.submitted, self.tasks = callable_, list(tasks)
        return [
            SimpleNamespace(job_id=f"{self.job_id}_{i}") for i, _ in enumerate(tasks)
        ]

    def submit(self, callable_):
        self.submitted = callable_
        return SimpleNamespace(job_id=self.job_id)


@pytest.fixture()
def slurm(monkeypatch):
    """
    Stub the source snapshot, clean provenance, sbatch, and executors; job IDs
    count up by 100 per executor.
    """
    calls, executors = [], []

    def factory(folder, **options):
        calls.append(options)
        executors.append(FakeExecutor(str(100 * len(calls))))
        return executors[-1]

    def snapshot(_repo_dir, _commit, destination):
        (Path(destination) / "src").mkdir(parents=True)
        return Path(destination)

    monkeypatch.setattr(f"{_CAMPAIGN}.create_source_snapshot", snapshot)
    monkeypatch.setattr(
        "fairchem.core.regression_suite.execution.source.git_provenance",
        lambda _repo_dir: ("commit", False),
    )
    monkeypatch.setattr(f"{_CAMPAIGN}.EXECUTORS", {"slurm": factory})
    monkeypatch.setattr(f"{_CAMPAIGN}.shutil.which", lambda _name: "/usr/bin/sbatch")
    monkeypatch.setenv("PYTHONPATH", "")
    return SimpleNamespace(calls=calls, executors=executors)


def _launch(tmp_path, **overrides):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    return launch_baseline(
        **{
            "artifact_dir": tmp_path / "artifacts",
            "campaign": "full",
            "checkpoint": str(checkpoint),
            "checkpoint_id": "checkpoint-id",
            "repo_dir": _REPO,
            "run_name": "test-regression",
            "submit": True,
            "parallelism": 15,
            "resources": SlurmConfig(partition="g3", mem_gb=120),
            "finalize_resources": SlurmConfig(partition="g3", mem_gb=16),
            **overrides,
        }
    )


def test_baseline_phases_chain_and_seal_only_after_every_phase_succeeds(
    tmp_path, slurm
):
    manifest = _launch(tmp_path)

    phases = manifest["phases"]
    assert [phases[name]["array_job_id"] for name in list(phases)[:-1]] == [
        "100",
        "200",
        "300",
        "400",
        "500",
        "600",
    ]
    # Each role's merge waits for its shards; roles themselves run in parallel.
    assert [call.get("dependency") for call in slurm.calls[:6]] == [
        None,
        "100",
        None,
        "300",
        None,
        "500",
    ]
    assert phases["finalize"] == {
        "job_id": "700",
        "dependency": "100:200:300:400:500:600",
    }
    assert slurm.calls[6]["after"] == "afterok"
    assert [call["gpus_per_node"] for call in slurm.calls] == [1, 0, 1, 0, 1, 0, 0]
    source = manifest["source_dir"]
    assert {executor.submitted.args[0] for executor in slurm.executors} == {source}


def test_resume_submits_only_missing_runs(tmp_path, slurm, monkeypatch):
    status = baseline_status(tmp_path / "artifacts", "full")
    complete = {
        "state": "complete",
        "identity": {
            "git_commit": "commit",
            "git_dirty": False,
            "checkpoint": "checkpoint-id",
            "seed": 42,
            "device": "cuda",
        },
    }
    for key in status["fragments"]["reference"]:
        status["fragments"]["reference"][key] = complete
    status["fragments"]["calibration"]["elastic/fp32-eager"] = complete
    monkeypatch.setattr(f"{_CAMPAIGN}.baseline_status", lambda *_args: status)

    manifest = _launch(tmp_path)

    assert manifest["phases"]["reference"]["array_job_id"] is None
    tasks = manifest["phases"]["calibration"]["tasks"]
    assert len(tasks) == 3 * (len(SUITES) + _NVE_SHARDS - 1) - 1
    assert ("elastic", "fp32-eager") not in {
        (task["suite"], task["profile"]) for task in tasks
    }
    assert manifest["phases"]["finalize"]["dependency"] == "100:200:300:400"


def test_overlapping_launch_is_refused(tmp_path, slurm, monkeypatch):
    _launch(tmp_path)
    monkeypatch.setattr(
        f"{_CAMPAIGN}.active_slurm_job_ids", lambda job_ids: {"400"} & job_ids
    )

    with pytest.raises(CommandError, match="earlier baseline launch is still active"):
        _launch(tmp_path)


def test_timed_out_array_tasks_are_requeued_but_the_finalizer_is_not(tmp_path, slurm):
    _launch(tmp_path)

    def time_out(function, arguments, folder):
        delayed = DelayedSubmission(function, *arguments)
        delayed.set_timeout(timeout_min=0, max_num_timeout=3)
        folder.mkdir()
        paths = JobPaths(folder, job_id="1_0")
        requeued = []
        environment = SimpleNamespace(global_rank=0, _requeue=requeued.append)
        with pytest.raises(SystemExit):
            SignalHandler(environment, paths, delayed).checkpoint_and_try_requeue(
                signal.SIGUSR2
            )
        assert requeued == [2]
        return DelayedSubmission.load(paths.submitted_pickle)

    *phases, finalizer = slurm.executors
    task = phases[0].tasks[0]
    rerun = time_out(phases[0].submitted, (task,), tmp_path / "phase")
    assert rerun.args == (task,)
    assert rerun.function.keywords == phases[0].submitted.keywords
    with pytest.raises(UncompletedJobError, match="not checkpointable"):
        time_out(finalizer.submitted, (), tmp_path / "finalize")


def test_candidate_report_follows_its_tasks_even_when_some_fail(
    tmp_path, slurm, monkeypatch
):
    baseline = {
        "baseline_id": "baseline-id",
        "checkpoint": "test-model",
        "seed": 42,
        "device": "cuda",
        "scope": "full",
        "suites": {
            suite: {"git_commit": "c", "plan_hash": "p", "suite_contract_hash": "h"}
            for suite in SUITES
        },
    }
    root = "fairchem.core.regression_suite"
    monkeypatch.setattr(f"{_CAMPAIGN}.load_baseline", lambda _path: baseline)
    monkeypatch.setattr(
        f"{root}.artifacts.baselines.load_baseline", lambda _p: baseline
    )
    monkeypatch.setattr(
        f"{root}.execution.runner.pretrained_mlip.available_models", {"test-model": {}}
    )
    monkeypatch.setattr(
        f"{root}.evaluation.identity.suite_expectation",
        lambda _suite, _profile: {"plan_hash": "p", "suite_contract_hash": "h"},
    )

    manifest = launch_candidate(
        baseline_dir=tmp_path / "baseline",
        baseline_source="hf:facebook/UMA/regression_baselines/test.baseline@abc",
        candidate_dir=tmp_path / "candidate",
        report_dir=tmp_path / "report",
        repo_dir=_REPO,
        profile="fp32_eager",
        suites=["edge-cases", "inference"],
        run_name="candidate-test",
        submit=True,
        parallelism=4,
        resources=SlurmConfig(partition="g3"),
        finalize_resources=SlurmConfig(partition="g3", mem_gb=16),
    )

    assert manifest["phases"]["finalize"] == {"job_id": "200", "dependency": "100"}
    assert slurm.calls[1]["after"] == "afterany"
    keywords = slurm.executors[0].submitted.keywords
    assert keywords["baseline_id"] == "baseline-id"
    assert [task.suite for task in slurm.executors[0].tasks] == [
        "edge-cases",
        "inference",
    ]
