"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import shutil

import huggingface_hub
import pytest

from fairchem.core.regression_suite import cli
from fairchem.core.regression_suite.artifacts.baselines import point_current_baseline
from tests.core.regression_suite.artifact_factories import sealed_baseline

HF_FILE = "regression_baselines/uma-s-1p2p1-f00dcafe.baseline"
HF_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _settings_file(monkeypatch, tmp_path, text):
    override = tmp_path / "settings.yaml"
    override.write_text(text, encoding="utf-8")
    monkeypatch.setenv("FAIRCHEM_REGRESSION_CONFIG", str(override))


@pytest.fixture()
def root(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "artifact_root", lambda: tmp_path / "root")
    monkeypatch.delenv("FAIRCHEM_REGRESSION_CONFIG", raising=False)
    monkeypatch.setattr(cli, "checkout_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(cli, "_short_commit", lambda _repo: "abc12345")
    monkeypatch.setattr(cli, "_timestamp", lambda: "20260929T000000Z")
    return tmp_path / "root"


@pytest.fixture()
def hub(root, monkeypatch, tmp_path):
    """
    Serve a sealed ``uma-s-1p2p1`` baseline as any file of a fake Hugging Face
    cache; return the download calls.
    """
    sealed = sealed_baseline(tmp_path / "built", "uma-s-1p2p1-f00dcafe")
    snapshot = tmp_path / "hub" / "models--facebook--UMA" / "snapshots" / HF_COMMIT
    calls = []

    def hf_hub_download(repo_id, filename, *, revision=None, **_options):
        calls.append((repo_id, filename, revision))
        cached = snapshot / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sealed, cached)
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return calls


@pytest.fixture()
def launches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "launch_candidate",
        lambda **config: calls.append(config) or {"status": "submitted"},
    )
    return calls


@pytest.mark.parametrize(
    ("settings", "checkpoint"),
    [
        (None, []),
        ("checkpoint: uma-m-1p1\n", ["--checkpoint", "uma-s-1p2p1"]),
        (
            "checkpoint: uma-m-1p1\n",
            ["--checkpoint", "/models/x.pt", "--checkpoint-id", "uma-s-1p2p1"],
        ),
    ],
)
def test_checks_default_to_the_checkpoints_published_baseline(
    monkeypatch, tmp_path, hub, launches, settings, checkpoint
):
    if settings is not None:
        _settings_file(monkeypatch, tmp_path, settings)
    filename = "regression_baselines/uma-s-1p2p1.baseline"

    assert cli.main(["full", "multigpu", "--run-on", "slurm", *checkpoint]) == 0

    assert hub == [("facebook/UMA", filename, None)]
    assert launches[0]["baseline_source"] == f"hf:facebook/UMA/{filename}@{HF_COMMIT}"


def test_baseline_flag_overrides_the_setting_which_is_pinned_when_used(
    root, monkeypatch, tmp_path, hub, launches
):
    local = sealed_baseline(root / "baselines", "uma-s-1p2p1-abc12345")
    point_current_baseline(local, root / "baselines" / "current")
    _settings_file(monkeypatch, tmp_path, f"baseline: hf:facebook/UMA/{HF_FILE}\n")
    command = ["full", "multigpu", "--run-on", "slurm"]

    assert cli.main(command) == 0
    assert cli.main([*command, "--baseline", "current"]) == 0

    assert hub == [("facebook/UMA", HF_FILE, None)]
    assert launches[0]["baseline_source"] == f"hf:facebook/UMA/{HF_FILE}@{HF_COMMIT}"
    assert launches[1]["baseline_dir"] == local.resolve()


@pytest.mark.parametrize("command", ["fast", "full"])
def test_baselines_of_another_checkpoint_stop_before_anything_starts(
    monkeypatch, capsys, hub, command
):
    started = []
    for name in ("launch_candidate", "run_fast", "create_slurm_executor"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: started.append(args))

    arguments = [command, "tf32", "--run-on", "slurm", "--checkpoint", "uma-m-1p1"]
    assert cli.main(arguments) == 1

    assert hub == [("facebook/UMA", "regression_baselines/uma-m-1p1.baseline", None)]
    assert started == []
    error = capsys.readouterr().err
    assert "built from checkpoint uma-s-1p2p1, not the requested uma-m-1p1" in error


def test_full_publishes_to_wandb_only_when_asked(monkeypatch, tmp_path, hub, launches):
    command = ["full", "fp32", "--run-on", "slurm"]

    assert cli.main(command) == 0
    assert cli.main([*command, "--wandb", "--tag", "t"]) == 0
    _settings_file(monkeypatch, tmp_path, "wandb:\n  auto_publish: true\n")
    assert cli.main(command) == 0
    assert cli.main([*command, "--no-wandb"]) == 0

    default, asked, automatic, declined = (launch["wandb"] for launch in launches)
    assert default is None
    assert asked.run_name == "fp32-abc12345-20260929T000000Z"
    assert asked.group == f"uma-s-1p2p1@{HF_COMMIT[:8]}"
    assert asked.tags == ("t",)
    assert automatic is not None
    assert declined is None


def test_smoke_baselines_are_never_published(root, monkeypatch, capsys):
    (root / "baselines" / "v12-smoke").mkdir(parents=True)
    published = []
    monkeypatch.setattr(cli, "publish_baseline", lambda *args: published.append(args))

    command = ["baseline", "--name", "v12", "--scope", "smoke", "--publish"]
    assert cli.main(command) == 1

    assert published == []
    assert "smoke baselines are never published" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "headline", "cause"),
    [
        (["quick", "tf32", "--suite", "bogus"], "Error: unknown suites", None),
        (["quick", "bf16"], "Error: unknown profile 'bf16'", None),
        (
            ["fast", "fp32", "--baseline", "hf:facebook/UMA"],
            "Baseline unavailable; nothing was started.",
            "malformed baseline reference",
        ),
        (
            ["full", "fp32"],
            "Baseline unavailable; nothing was started.",
            "uma-s-1p2p1.baseline is not available",
        ),
    ],
)
def test_expected_errors_print_a_short_message_and_start_nothing(
    root, monkeypatch, capsys, arguments, headline, cause
):
    def missing(*_args, **_options):
        raise huggingface_hub.errors.EntryNotFoundError("404 Entry Not Found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    started = []
    for name in ("run_quick", "run_fast", "launch_candidate"):
        monkeypatch.setattr(cli, name, lambda **kwargs: started.append(kwargs))

    assert cli.main([*arguments, "--run-on", "local"]) == 1

    assert started == []
    first, *rest = capsys.readouterr().err.splitlines()
    assert first.startswith(headline)
    if cause is None:
        assert rest == []
    else:
        assert cause in "\n".join(rest)
