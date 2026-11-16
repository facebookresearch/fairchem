"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

from fairchem.core.regression_suite import cli
from fairchem.core.regression_suite.artifacts import storage
from fairchem.core.regression_suite.artifacts.storage import (
    load_json,
    open_artifact,
    write_json_atomic,
)
from fairchem.core.regression_suite.execution.source import SOURCE_MARKER
from fairchem.core.regression_suite.suites import STANDARD_SUITES
from tests.core.regression_suite.tiny_model import write_tiny_checkpoint

# Calibrate a tiny random-weight UMA end to end on CPU. One sealed smoke-scope
# baseline per module grades every candidate. On CPU, FP64 reruns are bitwise
# identical and a TF32 profile computes in plain FP32, so FP64 candidates must
# reach the FP64 tier and FP32 and TF32 candidates exactly the FP32 tier. GPU
# FP32 is not bitwise reproducible, so a smoke baseline does not seal reliably
# there; the TF32 tier is covered by real-model campaigns.
pytestmark = pytest.mark.serial

# Workers record this commit only if they read it from the snapshot marker.
_SNAPSHOT_COMMIT = "e2e0" * 10
_PASSING = {"PASS", "PASS@FP32", "PASS@TF32"}
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _link_farm(source: Path, destination: Path, copied: Path) -> None:
    """
    Mirror ``source`` with symlinks, except for real directories leading to
    ``copied`` and a copy of it.
    """
    destination.mkdir(parents=True)
    for entry in source.iterdir():
        if entry == copied:
            shutil.copy2(entry, destination / entry.name)
        elif copied.is_relative_to(entry):
            _link_farm(entry, destination / entry.name, copied)
        else:
            (destination / entry.name).symlink_to(entry)


def _working_tree_snapshot(_repo_dir, commit, destination) -> Path:
    """
    Stand in for ``git archive`` with the imported ``fairchem`` package, so a
    dirty checkout or an installed copy tests its own code.

    Workers look for the marker at the root above the resolved ``storage.py``,
    which follows symlinks, so that file is copied and the rest is linked.
    """
    root = Path(destination)
    copied = Path(storage.__file__).resolve()
    _link_farm(copied.parents[3], root / "src" / "fairchem", copied)
    write_json_atomic(root / SOURCE_MARKER, {"git_commit": commit})
    return root


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """
    Point the CLI at a scratch artifact root and a tiny random-weight UMA,
    launch from a clean snapshot of the working tree, and seal one CPU smoke
    baseline for the module.
    """
    scratch = tmp_path_factory.mktemp("tiny")
    root = scratch / "root"
    threads = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "artifact_root", lambda: root)
        patch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *_args, **_options: pytest.fail("unexpected Hugging Face download"),
        )
        patch.setattr(
            "fairchem.core.regression_suite.execution.source.git_provenance",
            lambda _repo_dir: (_SNAPSHOT_COMMIT, False),
        )
        patch.setattr(
            "fairchem.core.regression_suite.execution.campaign.create_source_snapshot",
            _working_tree_snapshot,
        )
        # CI imports an installed copy, so stand in for the checkout with this one.
        patch.setattr(cli, "checkout_root", lambda: _REPO_ROOT)
        runs = itertools.count()
        patch.setattr(cli, "_timestamp", lambda: f"run{next(runs)}")
        patch.delenv("FAIRCHEM_REGRESSION_CONFIG", raising=False)
        # Launches prepend their snapshot to PYTHONPATH; checks set compile caches.
        patch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        for variable in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            patch.setenv(variable, str(scratch / variable.lower()))
        torch.set_num_threads(4)
        try:
            checkpoint = write_tiny_checkpoint(scratch / "tiny.pt")
            command = ["baseline", "--scope", "smoke", "--device", "cpu", "--no-wandb"]
            assert cli.main([*command, "--name", "tiny", *_model(checkpoint)]) == 0
            yield SimpleNamespace(
                root=root,
                checkpoint=checkpoint,
                sealed=root / "baselines" / "tiny-smoke.baseline",
            )
        finally:
            torch.set_num_threads(threads)


def _model(checkpoint: Path) -> list[str]:
    model = ["--checkpoint", str(checkpoint), "--checkpoint-id", "tiny-uma"]
    return [*model, "--run-on", "local"]


_CHECK_REPORT_KEYS = {
    "mode",
    "profile",
    "reference",
    "checkpoint",
    "created_at",
    "baseline",
    "suites",
    "exit_code",
}


def _check_report(tiny, mode: str, profile: str) -> dict:
    """
    Return the only ``mode`` check's JSON report, filed under the run's
    timestamp beside the other reports, after checking that its HTML report
    names every suite the JSON report holds.
    """
    (results,) = tiny.root.glob(f"{mode}/*/")
    stamp = results.name.removeprefix(f"{profile}-")
    run = tiny.root / "reports" / profile / mode / stamp
    assert [*tiny.root.glob(f"reports/*/{mode}/*/")] == [run]
    report = json.loads((run / "report.json").read_text())
    assert set(report) == _CHECK_REPORT_KEYS
    page = (run / "index.html").read_text()
    assert all(suite["suite"] in page for suite in report["suites"])
    return report


def _full(
    tiny, profile: str, checkpoint: Path, *options: str
) -> tuple[int, Path, Path]:
    """
    Grade ``profile`` against the sealed baseline; return the exit code, the
    candidate directory, and the report directory this run wrote.
    """
    before = set(tiny.root.glob("candidates/*"))
    command = ["full", profile, "--baseline", str(tiny.sealed), "--no-wandb"]
    code = cli.main([*command, *options, *_model(checkpoint)])
    (candidate,) = set(tiny.root.glob("candidates/*")) - before
    run = candidate.name.removeprefix(f"{profile}-")
    assert not (candidate / "report").exists()
    return code, candidate, tiny.root / "reports" / profile / "full" / run


def test_quick_compares_fp32_with_fp64_on_every_standard_suite(tiny, capsys):
    command = ["quick", "fp32", "--reference", "fp64", "--device", "cpu"]
    capsys.readouterr()

    assert cli.main([*command, *_model(tiny.checkpoint)]) == 0

    results = [load_json(path) for path in tiny.root.glob("quick/*/*/result.json")]
    assert sorted(result["suite"] for result in results) == sorted(STANDARD_SUITES)
    assert {result["status"] for result in results} == {"OK"}
    assert any(
        (observable["statistics"]["p95"] or 0.0) > 0.0
        for result in results
        for observable in result["summary"]["observables"].values()
    )
    report = _check_report(tiny, "quick", "fp32")
    assert (report["mode"], report["reference"], report["baseline"]) == (
        "quick",
        "fp64-eager",
        None,
    )
    assert sorted(suite["suite"] for suite in report["suites"]) == sorted(
        STANDARD_SUITES
    )
    # The detailed blocks come first and the overview table ends the output.
    printed = capsys.readouterr().out.splitlines()
    artifact = max(i for i, line in enumerate(printed) if line.startswith("Artifact"))
    separator = next(i for i in range(artifact, len(printed)) if "-+-" in printed[i])
    total = next(i for i, line in enumerate(printed) if line.startswith("Total"))
    rows = [line.split(" | ") for line in printed[separator + 1 : total]]
    assert sorted(tuple(cell.strip() for cell in row[:3]) for row in rows) == sorted(
        (suite["suite"], suite["status"], name)
        for suite in report["suites"]
        for name in suite["verdicts"]
    )


def test_smoke_baseline_separates_fp32_limits_from_fp64(tiny):
    baseline = load_json(open_artifact(tiny.sealed) / "baseline.json")

    assert (baseline["scope"], baseline["device"]) == ("smoke", "cpu")
    assert sorted(baseline["suites"]) == sorted(STANDARD_SUITES)
    runs = [
        run
        for suite in baseline["suites"].values()
        for collection in ("profiles", "verifications")
        for run in suite[collection].values()
    ]
    assert {(run["git_commit"], run["git_dirty"]) for run in runs} == {
        (_SNAPSHOT_COMMIT, False)
    }
    verification = tiny.root / "baselines" / "tiny-smoke" / "reports"
    assert {path.parent.name for path in verification.glob("*/index.html")} == {
        f"verification-{profile}-eager" for profile in ("fp64", "fp32", "tf32")
    }
    assert not (tiny.root / "baselines" / "current").exists()

    # Limits never decrease toward lower tiers by construction, so only strict
    # separation shows that FP32 noise was measured above the FP64 floor.
    separated = {}
    for name, suite in baseline["suites"].items():
        separated[name] = [
            fp32[statistic] > limits["FP64"][channel][statistic]
            for observable in suite["observables"].values()
            if (limits := observable["limits"]) is not None
            for channel, fp32 in limits["FP32"].items()
            for statistic in fp32
        ]
    assert all(any(flags) for flags in separated.values()), separated
    flags = list(itertools.chain.from_iterable(separated.values()))
    assert sum(flags) >= 0.8 * len(flags)


@pytest.mark.parametrize(
    ("profile", "status", "output_format"),
    [
        ("fp64", "PASS", "table"),
        ("fp32", "PASS@FP32", "summary"),
        ("tf32", "PASS@FP32", "table"),
    ],
)
def test_full_candidates_land_on_their_precision_tier(
    tiny, capsys, profile, status, output_format
):
    capsys.readouterr()
    options = ("--format", output_format)
    code, _candidate, report_dir = _full(tiny, profile, tiny.checkpoint, *options)

    assert code == 0
    printed = capsys.readouterr().out
    overview = printed.rindex("Suite qualification")
    details = printed.rfind("observable statistics")
    assert details < 0 if output_format == "summary" else 0 < details < overview
    assert printed[overview:].count(status) >= len(STANDARD_SUITES)
    report = load_json(report_dir / "report.json")
    assert (report_dir / "index.html").is_file()
    assert {suite["suite"]: suite["status"] for suite in report["suites"]} == (
        dict.fromkeys(STANDARD_SUITES, status)
    )


def test_fast_fp32_smoke_passes_at_fp32(tiny, capsys):
    command = ["fast", "fp32", "--baseline", str(tiny.sealed), "--format", "json"]
    capsys.readouterr()

    assert cli.main([*command, *_model(tiny.checkpoint)]) == 0

    results = [load_json(path) for path in tiny.root.glob("fast/*/*/result.json")]
    assert {result["suite"]: result["status"] for result in results} == (
        dict.fromkeys(STANDARD_SUITES, "SMOKE_PASS@FP32")
    )
    report = _check_report(tiny, "fast", "fp32")
    assert (report["mode"], report["exit_code"]) == ("fast", 0)
    assert report["baseline"]["name"] == "tiny-smoke"
    assert json.loads(capsys.readouterr().out) == report
    assert {suite["suite"]: suite["status"] for suite in report["suites"]} == (
        dict.fromkeys(STANDARD_SUITES, "SMOKE_PASS@FP32")
    )


def test_full_fails_a_model_with_different_weights(tiny):
    changed = write_tiny_checkpoint(tiny.checkpoint.with_name("changed.pt"), seed=1)

    code, _candidate, report_dir = _full(tiny, "fp32", changed)

    assert code == 1
    statuses = {
        suite["status"] for suite in load_json(report_dir / "report.json")["suites"]
    }
    assert statuses - _PASSING


def test_report_json_output_parses_despite_output_while_building(
    tiny, monkeypatch, capsys, tmp_path
):
    capsys.readouterr()
    code, candidate, report_dir = _full(
        tiny, "fp64", tiny.checkpoint, "--format", "json"
    )
    assert json.loads(capsys.readouterr().out) == load_json(report_dir / "report.json")
    build = cli.candidate_run_report

    def noisy_build(candidate_dir):
        # GPU libraries such as NVIDIA Warp print a banner when first used.
        print("Warp initialized:")
        return build(candidate_dir)

    monkeypatch.setattr(cli, "candidate_run_report", noisy_build)
    capsys.readouterr()

    assert cli.main(["report", "--format", "json", str(candidate)]) == code
    printed = capsys.readouterr()
    assert json.loads(printed.out) == load_json(report_dir / "report.json")
    assert "Warp initialized:" in printed.err

    output = tmp_path / "report.json"
    command = ["report", "--format", "json", "--output", str(output)]
    assert cli.main([*command, str(candidate)]) == code
    assert json.loads(output.read_text()) == load_json(report_dir / "report.json")
    assert capsys.readouterr().out == ""
