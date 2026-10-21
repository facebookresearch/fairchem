"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

File MD5 sums cached in ``md5sum``-compatible sidecar files.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path

# Starts the comment line that marks a sidecar written here; md5sum -c skips
# comment lines.
_FINGERPRINT_PREFIX = "# fairchem "


def file_md5(path: str | os.PathLike) -> str:
    """
    Hash a file's contents with MD5.

    Args:
        path: The file to hash.

    Returns:
        The hex digest.
    """
    md5 = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def cached_file_md5(path: str | os.PathLike) -> str:
    """
    Hash a file with MD5, reusing a ``<path>.md5`` sidecar while it is valid.

    The sidecar is a standard ``md5sum`` file, so ``md5sum -c`` checks it, plus a
    comment line with the file's size and modification time. It is trusted only
    while both still match the file, like rsync's quick check; otherwise the file
    is hashed again and the sidecar rewritten. Sidecars without that line, such
    as published checksums, are never overwritten, and a sidecar that cannot be
    written is skipped.

    Args:
        path: The file to hash. For a symlink, the sidecar sits next to the link
            and the size and modification time are the target's.

    Returns:
        The hex digest.
    """
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.md5")
    fingerprint = _fingerprint(path)
    try:
        lines = sidecar.read_text(encoding="utf-8", errors="surrogateescape")
        lines = lines.splitlines()
    except OSError:
        lines = None
    if lines is not None:
        if len(lines) == 2 and lines[1] == fingerprint:
            digest = lines[0][:32]
            if re.fullmatch("[0-9a-f]{32}", digest) and lines[0] == _line(digest, path):
                return digest
        if not any(line.startswith(_FINGERPRINT_PREFIX) for line in lines):
            return file_md5(path)
    digest = file_md5(path)
    # A file that changed while it was hashed matches neither fingerprint.
    if _fingerprint(path) == fingerprint:
        _write_atomically(sidecar, f"{_line(digest, path)}\n{fingerprint}\n")
    return digest


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{_FINGERPRINT_PREFIX}size={stat.st_size} mtime_ns={stat.st_mtime_ns}"


def _line(digest: str, path: Path) -> str:
    return f"{digest}  {path.name}"


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # O_EXCL never opens an existing file; 0o666 leaves permissions to the
        # umask, so a shared cache stays readable.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(
            descriptor, "w", encoding="utf-8", errors="surrogateescape"
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        logging.debug(f"Not caching the MD5 sum in {path}: {error}")
        with contextlib.suppress(OSError):
            temporary.unlink()
