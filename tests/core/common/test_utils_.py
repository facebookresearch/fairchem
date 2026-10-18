"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import pytest

from fairchem.core.common.utils import (
    get_checkpoint_dirs_sorted_by_save_time,
    get_deep,
    safe_extract_tar,
)


def test_get_deep() -> None:
    d = {"oc20": {"energy": 1.5}}
    assert get_deep(d, "oc20.energy") == 1.5
    assert get_deep(d, "oc20.force", 0.9) == 0.9
    assert get_deep(d, "omol.energy") is None


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


def _make_checkpoint(path: Path, saved_at: float) -> None:
    path.mkdir()
    (path / ".metadata").write_text("")
    os.utime(path / ".metadata", (saved_at, saved_at))
    os.utime(path, (saved_at, saved_at))


class TestGetCheckpointDirsSortedBySaveTime:
    def test_missing_dir(self, tmp_path):
        assert get_checkpoint_dirs_sorted_by_save_time(str(tmp_path / "missing")) == []

    def test_orders_by_metadata_mtime_not_dir_mtime(self, tmp_path):
        _make_checkpoint(tmp_path / "step_10", 1_000.0)
        _make_checkpoint(tmp_path / "step_20", 2_000.0)
        # A file written into step_10 later bumps its directory mtime only.
        (tmp_path / "step_10" / "inference.pt").write_text("artifact")
        os.utime(tmp_path / "step_10", (3_000.0, 3_000.0))

        assert get_checkpoint_dirs_sorted_by_save_time(str(tmp_path)) == [
            (str(tmp_path / "step_10"), 1_000.0),
            (str(tmp_path / "step_20"), 2_000.0),
        ]

    def test_incomplete_dirs_use_their_own_mtime(self, tmp_path):
        _make_checkpoint(tmp_path / "step_10", 1_000.0)
        (tmp_path / "step_20").mkdir()  # no .metadata: save never finished
        os.utime(tmp_path / "step_20", (2_000.0, 2_000.0))
        (tmp_path / "not_a_dir").write_text("")

        assert get_checkpoint_dirs_sorted_by_save_time(str(tmp_path)) == [
            (str(tmp_path / "step_10"), 1_000.0),
            (str(tmp_path / "step_20"), 2_000.0),
        ]
