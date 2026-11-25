"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.artifacts.storage import (
    load_json,
    repository_root,
    staged_directory,
    write_json_atomic,
)
from fairchem.core.regression_suite.execution.environment import (
    prepare_compilation_cache,
)
from fairchem.core.regression_suite.spec.schema import CommandError

if TYPE_CHECKING:
    from collections.abc import Callable


_PROXY_VARIABLES = (
    "X2P_AGENT_PROXY_ADDRESS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


SOURCE_MARKER = "fairchem-regression-source.json"
_SNAPSHOT_PATHS = ("src", "packages/fairchem-core/pyproject.toml")
_PROVENANCE_PATHSPECS = (
    ":(top)src/fairchem/core",
    ":(top)packages/fairchem-core/pyproject.toml",
)


def git_output(repo_dir: str | Path | None, *arguments: str) -> str:
    """
    Run one git command in ``repo_dir`` and return its standard output.
    """
    return subprocess.run(
        ["git", *arguments], cwd=repo_dir, check=True, capture_output=True, text=True
    ).stdout


def git_provenance(repo_dir: str | Path) -> tuple[str, bool]:
    """
    Return the checked-out commit and whether regression-relevant files changed.

    Args:
        repo_dir: Repository checkout to inspect.

    Returns:
        The ``HEAD`` commit and whether FAIRChem core sources are modified.
    """
    commit = git_output(repo_dir, "rev-parse", "HEAD").strip()
    status = git_output(
        repo_dir,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        *_PROVENANCE_PATHSPECS,
    )
    return commit, bool(status.strip())


def create_source_snapshot(
    repo_dir: str | Path,
    commit: str,
    destination: str | Path,
) -> Path:
    """
    Export one committed source tree for SLURM jobs to import.

    Jobs import the snapshot instead of the live checkout, so later edits,
    commits, or branch switches cannot change code that is queued or running.

    Args:
        repo_dir: Repository containing the commit.
        commit: Commit to export.
        destination: New snapshot directory; it must not exist.

    Returns:
        The snapshot directory, which contains ``src`` and a provenance marker.
    """
    output = Path(destination)
    with staged_directory(output) as temporary:
        archive = temporary / "source.tar"
        git_output(
            repo_dir,
            "archive",
            "--format=tar",
            f"--output={archive}",
            commit,
            "--",
            *_SNAPSHOT_PATHS,
        )
        with tarfile.open(archive) as handle:
            handle.extractall(temporary, filter="data")
        archive.unlink()
        write_json_atomic(temporary / SOURCE_MARKER, {"git_commit": commit})
    return output


def source_provenance(root: str | Path | None = None) -> tuple[str | None, bool | None]:
    """
    Return the commit and dirty state of the imported regression-suite code.

    A source snapshot is immutable and records its commit in a marker file.
    Otherwise the checkout containing the imported package is inspected, which
    may differ from the process working directory.

    Args:
        root: Repository or snapshot root; defaults to the imported package's.

    Returns:
        The commit and dirty state, or ``(None, None)`` when unknown.
    """
    source_root = repository_root() if root is None else Path(root)
    marker = source_root / SOURCE_MARKER
    if marker.is_file():
        return str(load_json(marker)["git_commit"]), False
    try:
        return git_provenance(source_root)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def clean_commit(repo_dir: Path, kind: str) -> str:
    commit, dirty = git_provenance(repo_dir)
    if dirty:
        raise CommandError(f"{kind} launch requires a clean source tree")
    return commit


def call_in_source(repo_dir: str, function: Callable[..., Any], /, *args, **kwargs):
    """
    Call a submitted function after preparing its process for a source tree.
    """
    _prepare_environment(repo_dir)
    return function(*args, **kwargs)


def _prepare_environment(repo_dir: str) -> None:
    os.chdir(repo_dir)
    source_directory = str(Path(repo_dir) / "src")
    sys.path[:] = [source_directory] + [
        entry for entry in sys.path if entry != source_directory
    ]
    prepend_pythonpath(source_directory)
    for variable in _PROXY_VARIABLES:
        os.environ.pop(variable, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    prepare_compilation_cache()


def prepend_pythonpath(source_directory: str) -> None:
    entries = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(":")
        if entry and entry != source_directory
    ]
    entries.insert(0, source_directory)
    os.environ["PYTHONPATH"] = ":".join(entries)
