"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import shutil
import sys
from types import ModuleType
from unittest.mock import Mock

import huggingface_hub
import pytest

from fairchem.core.regression_suite.artifacts.baselines import (
    BaselineUnavailableError,
    check_baseline,
    point_current_baseline,
)
from fairchem.core.regression_suite.artifacts.sources import (
    HuggingFaceReference,
    LocalReference,
    WandBReference,
    parse_baseline_reference,
    resolve_baseline,
)
from fairchem.core.regression_suite.spec.hashing import file_sha256
from tests.core.regression_suite.artifact_factories import sealed_baseline

COMMIT = "0123456789abcdef0123456789abcdef01234567"
HF_FILE = "regression_baselines/uma-s-1p2p1-c0ffee00.baseline"


@pytest.mark.parametrize(
    ("reference", "parsed"),
    [
        (
            f"hf:facebook/UMA/{HF_FILE}",
            HuggingFaceReference("facebook/UMA", HF_FILE),
        ),
        (
            f"hf:facebook/UMA/{HF_FILE}@{COMMIT}",
            HuggingFaceReference("facebook/UMA", HF_FILE, COMMIT),
        ),
        (
            "hf:org/repo/top.baseline@main",
            HuggingFaceReference("org/repo", "top.baseline", "main"),
        ),
        ("wandb:uma-s-1p2p1-baseline", WandBReference("uma-s-1p2p1-baseline")),
        (
            "wandb:uma-s-1p2p1-baseline:v3",
            WandBReference("uma-s-1p2p1-baseline", "v3"),
        ),
        (
            "wandb:team/project/uma-s-1p2p1-baseline",
            WandBReference("uma-s-1p2p1-baseline", "latest", "team/project"),
        ),
        (
            "wandb:team/project/uma-s-1p2p1-baseline:v3",
            WandBReference("uma-s-1p2p1-baseline", "v3", "team/project"),
        ),
        ("current", LocalReference("current")),
        ("uma-s-1p2p1-c0ffee00", LocalReference("uma-s-1p2p1-c0ffee00")),
        ("/data/x.baseline", LocalReference("/data/x.baseline")),
        ("./a:b.baseline", LocalReference("./a:b.baseline")),
    ],
)
def test_references_parse_by_prefix(reference, parsed):
    assert parse_baseline_reference(reference) == parsed


@pytest.mark.parametrize(
    ("reference", "form"),
    [
        ("hf:facebook/UMA", "hf:ORG/REPO/PATH[@REVISION]"),
        ("hf:facebook//x.baseline", "hf:ORG/REPO/PATH[@REVISION]"),
        (f"hf:facebook/UMA/{HF_FILE}@", "hf:ORG/REPO/PATH[@REVISION]"),
        ("wandb:", "wandb:[ENTITY/PROJECT/]NAME[:ALIAS]"),
        ("wandb:project/name", "wandb:[ENTITY/PROJECT/]NAME[:ALIAS]"),
        ("wandb:name:", "wandb:[ENTITY/PROJECT/]NAME[:ALIAS]"),
    ],
)
def test_malformed_references_name_the_expected_form(tmp_path, reference, form):
    with pytest.raises(BaselineUnavailableError) as error:
        resolve_baseline(reference, tmp_path)
    assert reference in str(error.value)
    assert form in str(error.value)


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """
    Serve one sealed baseline from a fake Hugging Face cache at ``COMMIT``.
    """
    sealed = sealed_baseline(tmp_path / "built", "uma-s-1p2p1-c0ffee00")
    cached = tmp_path / "hub" / "models--facebook--UMA" / "snapshots" / COMMIT
    cached = cached / HF_FILE
    cached.parent.mkdir(parents=True)
    shutil.copyfile(sealed, cached)
    calls = []

    def hf_hub_download(repo_id, filename, *, revision=None, **_options):
        calls.append((repo_id, filename, revision))
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return calls, sealed


def test_huggingface_baselines_are_copied_beside_local_ones_and_pinned(tmp_path, hub):
    calls, sealed = hub
    baselines = tmp_path / "root" / "baselines"

    resolved = resolve_baseline(f"hf:facebook/UMA/{HF_FILE}", baselines)
    again = resolve_baseline(resolved.source, baselines)

    assert calls == [
        ("facebook/UMA", HF_FILE, None),
        ("facebook/UMA", HF_FILE, COMMIT),
    ]
    assert resolved.source == f"hf:facebook/UMA/{HF_FILE}@{COMMIT}"
    assert again == resolved
    assert resolved.path == baselines / "uma-s-1p2p1-c0ffee00@01234567.baseline"
    assert file_sha256(resolved.path) == file_sha256(sealed)
    assert check_baseline(resolved.path) == check_baseline(sealed)


def test_huggingface_revisions_never_share_a_local_file(tmp_path, monkeypatch):
    baselines = tmp_path / "baselines"
    local = sealed_baseline(baselines, "uma-s-1p2p1")
    local_bytes = local.read_bytes()
    published = {
        commit: sealed_baseline(tmp_path / commit, "built", device)
        for commit, device in (("a" * 40, "cpu"), ("b" * 40, "cuda"))
    }
    filename = "regression_baselines/uma-s-1p2p1.baseline"

    def hf_hub_download(_repo_id, filename, *, revision, **_options):
        cached = tmp_path / "hub" / "snapshots" / revision / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(published[revision], cached)
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)

    paths = {
        commit: resolve_baseline(f"hf:facebook/UMA/{filename}@{commit}", baselines).path
        for commit in published
    }

    assert paths == {
        commit: baselines / f"uma-s-1p2p1@{commit[:8]}.baseline" for commit in published
    }
    for commit, path in paths.items():
        assert path.read_bytes() == published[commit].read_bytes()
    assert local.read_bytes() == local_bytes


def test_unreachable_huggingface_baselines_are_unavailable(tmp_path, monkeypatch):
    def offline(*_args, **_options):
        raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)

    with pytest.raises(BaselineUnavailableError, match="not in the Hugging Face cache"):
        resolve_baseline(f"hf:facebook/UMA/{HF_FILE}", tmp_path)


def _fake_wandb(monkeypatch, sealed, baseline_id):
    downloads = []

    def download(root):
        downloads.append(root)
        shutil.copyfile(sealed, os.path.join(root, sealed.name))
        return root

    remote = Mock(
        qualified_name="fairchem/uma-regression/uma-s-1p2p1-baseline:latest",
        version="v4",
        metadata={
            "baseline_name": sealed.stem,
            "baseline_id": baseline_id,
            "sha256": file_sha256(sealed),
        },
        download=download,
    )
    api = Mock(artifact=Mock(return_value=remote))
    module = ModuleType("wandb")
    module.Api = Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return api, downloads


def test_wandb_baselines_download_once_verify_and_pin_the_version(
    tmp_path, monkeypatch
):
    sealed = sealed_baseline(tmp_path / "built", "uma-s-1p2p1-c0ffee00")
    baseline_id = check_baseline(sealed)["baseline_id"]
    api, downloads = _fake_wandb(monkeypatch, sealed, baseline_id)
    monkeypatch.setenv("NO_PROXY", "localhost")
    baselines = tmp_path / "root" / "baselines"
    options = {
        "wandb_project": "fairchem/uma-regression",
        "wandb_no_proxy_hosts": ("store.example",),
    }

    first = resolve_baseline("wandb:uma-s-1p2p1-baseline", baselines, **options)
    second = resolve_baseline("wandb:team/other/name:v2", baselines, **options)

    assert [entry.args[0] for entry in api.artifact.call_args_list] == [
        "fairchem/uma-regression/uma-s-1p2p1-baseline:latest",
        "team/other/name:v2",
    ]
    assert first == second
    assert first.path == baselines / "uma-s-1p2p1-c0ffee00@v4.baseline"
    assert first.source == "wandb:fairchem/uma-regression/uma-s-1p2p1-baseline:v4"
    assert file_sha256(first.path) == file_sha256(sealed)
    assert len(downloads) == 1
    assert os.environ["NO_PROXY"] == "localhost,store.example"

    api.artifact.return_value.metadata["baseline_id"] = "other"
    with pytest.raises(BaselineUnavailableError, match="is not baseline other"):
        resolve_baseline("wandb:uma-s-1p2p1-baseline", baselines, **options)


def test_local_baselines_resolve_by_path_name_or_current(tmp_path):
    baselines = tmp_path / "baselines"
    first = sealed_baseline(baselines, "first")
    second = sealed_baseline(baselines, "second")
    point_current_baseline(second, baselines / "current")

    for reference, expected in (
        ("current", second),
        ("first", first),
        ("first.baseline", first),
        (str(first), first),
        ("second", second),
    ):
        resolved = resolve_baseline(reference, baselines)
        assert resolved.path == expected.resolve()
        assert resolved.source == str(expected.resolve())


def test_missing_local_baseline_explains_how_to_create_one(tmp_path):
    with pytest.raises(BaselineUnavailableError, match="fairchem-regression baseline"):
        resolve_baseline("current", tmp_path / "baselines")
