"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import pytest

from fairchem.core.common.utils import get_deep, safe_extract_tar


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
