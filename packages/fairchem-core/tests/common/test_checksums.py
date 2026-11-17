"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

import pytest

from fairchem.core.common import checksums
from fairchem.core.common.checksums import cached_file_md5


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _forbid_hashing(monkeypatch) -> None:
    def fail(path):
        raise AssertionError(f"{path} was hashed again")

    monkeypatch.setattr(checksums, "file_md5", fail)


@pytest.fixture()
def data_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights" * 1000)
    return path


def test_writes_md5sum_sidecar(data_file):
    digest = cached_file_md5(data_file)

    assert digest == _md5(data_file.read_bytes())
    lines = (data_file.parent / "model.pt.md5").read_text().splitlines()
    assert lines[0] == f"{digest}  model.pt"
    size, mtime = data_file.stat().st_size, data_file.stat().st_mtime_ns
    assert lines[1] == f"# fairchem size={size} mtime_ns={mtime}"
    # No temporary file is left behind.
    assert sorted(p.name for p in data_file.parent.iterdir()) == [
        "model.pt",
        "model.pt.md5",
    ]


def _has_gnu_md5sum() -> bool:
    if shutil.which("md5sum") is None:
        return False
    version = subprocess.run(
        ["md5sum", "--version"], capture_output=True, text=True, check=False
    )
    return "GNU coreutils" in version.stdout


@pytest.mark.skipif(not _has_gnu_md5sum(), reason="needs GNU coreutils md5sum")
def test_md5sum_checks_the_sidecar(data_file):
    cached_file_md5(data_file)
    subprocess.run(
        ["md5sum", "-c", "--strict", "model.pt.md5"],
        cwd=data_file.parent,
        check=True,
        capture_output=True,
    )


def test_reuses_a_valid_sidecar(data_file, monkeypatch):
    digest = cached_file_md5(data_file)
    _forbid_hashing(monkeypatch)
    assert cached_file_md5(data_file) == digest


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_rehashes_a_changed_file(data_file, change):
    cached_file_md5(data_file)
    stat = data_file.stat()
    if change == "size":
        data_file.write_bytes(b"other weights")
    else:
        # Same size, different content, an older modification time.
        data_file.write_bytes(b"W" * stat.st_size)
        os.utime(data_file, ns=(stat.st_atime_ns, stat.st_mtime_ns - 10**9))

    digest = cached_file_md5(data_file)

    assert digest == _md5(data_file.read_bytes())
    sidecar = (data_file.parent / "model.pt.md5").read_text()
    assert sidecar.startswith(f"{digest}  model.pt\n")


def test_keeps_a_published_sidecar(data_file):
    """
    A sidecar without the fairchem line records an expected sum: keep it.
    """
    sidecar = data_file.parent / "model.pt.md5"
    published = f"{'0' * 32}  model.pt\n"
    sidecar.write_text(published)

    assert cached_file_md5(data_file) == _md5(data_file.read_bytes())
    assert sidecar.read_text() == published


def test_replaces_a_damaged_sidecar(data_file):
    digest = cached_file_md5(data_file)
    sidecar = data_file.parent / "model.pt.md5"
    lines = sidecar.read_text().splitlines()
    sidecar.write_text(f"{digest[:20]}\n{lines[1]}\n")

    assert cached_file_md5(data_file) == digest
    assert sidecar.read_text().startswith(f"{digest}  model.pt\n")


def test_skips_a_sidecar_it_cannot_write(data_file, monkeypatch):
    def read_only(source, target):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(checksums.os, "replace", read_only)

    assert cached_file_md5(data_file) == _md5(data_file.read_bytes())
    assert sorted(p.name for p in data_file.parent.iterdir()) == ["model.pt"]


def test_skips_a_file_that_changes_while_hashed(data_file, monkeypatch):
    def hash_then_append(path):
        digest = _md5(data_file.read_bytes())
        with open(path, "ab") as handle:
            handle.write(b"more")
        return digest

    monkeypatch.setattr(checksums, "file_md5", hash_then_append)
    cached_file_md5(data_file)

    assert not (data_file.parent / "model.pt.md5").exists()


def test_symlink_sidecar_sits_next_to_the_link(tmp_path, monkeypatch):
    """
    The Hugging Face cache layout: a snapshot symlink to a content blob.
    """
    blob = tmp_path / "blobs" / "b2673b85"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    link = tmp_path / "snapshots" / "f611b917" / "model.pt"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(blob, link.parent))

    digest = cached_file_md5(link)

    assert digest == _md5(b"weights")
    assert (link.parent / "model.pt.md5").exists()
    assert sorted(p.name for p in blob.parent.iterdir()) == ["b2673b85"]
    _forbid_hashing(monkeypatch)
    assert cached_file_md5(link) == digest
