"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

from fairchem.core.common.utils import (
    get_branch_for_repo,
    get_deep,
    get_package_version,
    safe_extract_tar,
)


def test_get_deep() -> None:
    d = {"oc20": {"energy": 1.5}}
    assert get_deep(d, "oc20.energy") == 1.5
    assert get_deep(d, "oc20.force", 0.9) == 0.9
    assert get_deep(d, "omol.energy") is None


def _git(repo: Path, *args: str) -> None:
    subprocess.check_call(["git", "-C", str(repo), *args], stdout=subprocess.DEVNULL)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A repo with one commit on a branch named "a-branch"."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "a-branch")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "f.txt").write_text("hello")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def test_get_branch_for_repo(git_repo: Path) -> None:
    assert get_branch_for_repo(str(git_repo)) == "a-branch"


def test_get_branch_for_repo_detached_head(git_repo: Path) -> None:
    # a detached HEAD has no branch; the commit hash still identifies the code
    _git(git_repo, "checkout", "-q", "--detach", "HEAD")
    assert get_branch_for_repo(str(git_repo)) is None


def test_get_branch_for_repo_outside_a_repo(tmp_path: Path) -> None:
    # how a released (non-editable) install looks
    assert get_branch_for_repo(str(tmp_path)) is None


def test_get_package_version() -> None:
    # fairchem-core is installed to run this, so a version must resolve
    assert get_package_version()


class TestSafeExtractTar:
    """Downloaded tar archives must be validated against path traversal."""

    def _make_tar(self, tar_path: Path, arcname: str):
        payload = tar_path.parent / "payload"
        payload.write_bytes(b"x")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(payload, arcname=arcname)

    def test_extracts_safe_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "ok.tar.gz"
            self._make_tar(tar_path, "pkg/bin/prometheus")
            dest = root / "out"
            dest.mkdir()
            with tarfile.open(tar_path) as tar:
                safe_extract_tar(tar, str(dest))
            assert (dest / "pkg" / "bin" / "prometheus").exists()

    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "evil.tar.gz"
            self._make_tar(tar_path, "../evil")
            dest = root / "out"
            dest.mkdir()
            with tarfile.open(tar_path) as tar, pytest.raises(ValueError):
                safe_extract_tar(tar, str(dest))
            assert not (root / "evil").exists()

    def test_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "evil.tar.gz"
            # tar.add() strips leading "/", so craft the absolute entry directly.
            info = tarfile.TarInfo("/tmp/evil_abs")
            info.size = 0
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.addfile(info)
            dest = root / "out"
            dest.mkdir()
            with tarfile.open(tar_path) as tar, pytest.raises(ValueError):
                safe_extract_tar(tar, str(dest))

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "evil.tar.gz"
            info = tarfile.TarInfo("pkg/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../etc/passwd"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.addfile(info)
            dest = root / "out"
            dest.mkdir()
            with tarfile.open(tar_path) as tar, pytest.raises(ValueError):
                safe_extract_tar(tar, str(dest))
