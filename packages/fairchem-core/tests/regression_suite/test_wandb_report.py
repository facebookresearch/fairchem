"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from fairchem.core.regression_suite.artifacts.storage import load_json
from fairchem.core.regression_suite.reporting.check_report import build_check_report
from fairchem.core.regression_suite.reporting.wandb_report import (
    WandBReportSettings,
    publish_baseline,
    write_check_report,
)
from fairchem.core.regression_suite.spec.hashing import file_sha256


def test_publish_baseline_uploads_a_versioned_full_scope_artifact(
    tmp_path, monkeypatch
):
    artifact = Mock(qualified_name="fairchem/uma-regression/uma-s-1p2p1-baseline:v4")
    run = Mock()
    module = ModuleType("wandb")
    module.init = Mock(return_value=run)
    module.Artifact = Mock(return_value=artifact)
    monkeypatch.setitem(sys.modules, "wandb", module)
    (tmp_path / "v12").mkdir()
    sealed = tmp_path / "v12.baseline"
    sealed.write_bytes(b"sealed")
    baseline = {
        "baseline_id": "id",
        "checkpoint": "uma-s-1p2p1",
        "campaign": "standard",
        "scope": "full",
        "suites": {"a": {"git_commit": "c1"}, "b": {"git_commit": "c1"}},
    }
    settings = WandBReportSettings(project="uma-regression")

    with pytest.raises(ValueError, match="only full-scope baselines"):
        publish_baseline(sealed, {**baseline, "scope": "smoke"}, settings)
    module.init.assert_not_called()

    assert publish_baseline(sealed, baseline, settings) == artifact.qualified_name
    assert module.Artifact.call_args.args == ("uma-s-1p2p1-baseline",)
    metadata = module.Artifact.call_args.kwargs["metadata"]
    # Downloads verify the file against this hash and name it after the baseline.
    assert metadata["sha256"] == file_sha256(sealed)
    assert metadata["baseline_name"] == "v12"
    assert metadata["git_commits"] == ["c1"]
    artifact.add_file.assert_called_once_with(str(sealed))
    run.log_artifact.assert_called_once_with(artifact, aliases=["latest", "v12"])
    assert load_json(tmp_path / "v12" / "wandb.json") == {
        "artifact": "uma-s-1p2p1-baseline",
        "status": "PUBLISHED",
        "version": artifact.qualified_name,
    }


def _fast_result() -> dict:
    return {
        "mode": "fast",
        "suite": "inference",
        "profile": "nitro",
        "reference": "baseline-id",
        "checkpoint": "uma-s-1p2p1",
        "records_path": "/records",
        "reference_path": "/reference",
        "summary": {
            "completion": {"completed_cases": 2, "requested_cases": 2},
            "observables": {
                "energy": {"statistics": {"median_absolute": 1e-6, "p95": float("nan")}}
            },
        },
        "status": "SMOKE_PASS@FP32",
        "passed": True,
        "verdicts": {"energy": "FP32"},
        "qualification": {"status": "SMOKE_PASS@FP32", "tightest_tier": "FP32"},
        "elapsed_seconds": 3.0,
    }


def test_check_report_publishes_its_summary_and_keeps_files_on_failure(
    tmp_path, monkeypatch
):
    run = Mock(url="https://wandb.example/run", summary={})
    module = ModuleType("wandb")
    module.init = Mock(return_value=run)
    module.Html = Mock()
    module.Artifact = Mock()
    monkeypatch.setitem(sys.modules, "wandb", module)
    provenance = {
        "name": "v12",
        "baseline_id": "baseline-id",
        "created_at": None,
        "checkpoint": "uma-s-1p2p1",
        "campaign": "standard",
        "scope": "full",
        "device": "cuda",
        "commits": {"c1": ["inference"]},
    }
    report = build_check_report([_fast_result()], provenance)
    settings = WandBReportSettings(project="uma-regression", run_name="fast-run")
    published, failed = tmp_path / "published", tmp_path / "failed"

    assert write_check_report(report, published, settings) == run.url
    assert run.summary["fast/inference/status"] == "SMOKE_PASS@FP32"
    assert run.summary["fast/inference/tier"] == "FP32"
    assert module.init.call_args.kwargs["config"]["baseline"] == provenance
    assert load_json(published / "wandb.json") == {
        "run_name": "fast-run",
        "group": None,
        "status": "PUBLISHED",
        "url": run.url,
    }

    module.init.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        write_check_report(report, failed, settings)
    stored = load_json(failed / "report.json")
    # JSON has no NaN, so the non-finite statistic is stored as null.
    observable = stored["suites"][0]["summary"]["observables"]["energy"]
    assert observable["statistics"]["p95"] is None
    assert "inference" in (failed / "index.html").read_text()
    assert load_json(failed / "wandb.json")["status"] == "FAILED"
