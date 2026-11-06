"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import subprocess

import pytest

from fairchem.core.regression_suite.execution.source import (
    SOURCE_MARKER,
    create_source_snapshot,
    git_provenance,
    source_provenance,
)


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture()
def repository(tmp_path):
    repo = tmp_path / "repo"
    module = repo / "src" / "fairchem" / "core" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 'committed'\n")
    pyproject = repo / "packages" / "fairchem-core" / "pyproject.toml"
    pyproject.parent.mkdir(parents=True)
    pyproject.write_text("[project]\n")
    (repo / "README.md").write_text("outside the snapshot\n")
    _git(repo, "init", "--quiet")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    return repo


def test_snapshot_contains_committed_source_and_ignores_later_edits(
    repository, tmp_path
):
    commit, dirty = git_provenance(repository)
    module = repository / "src" / "fairchem" / "core" / "module.py"
    module.write_text("VALUE = 'edited after launch'\n")

    snapshot = create_source_snapshot(repository, commit, tmp_path / "source")

    assert not dirty
    assert git_provenance(repository) == (commit, True)
    assert (snapshot / "src" / "fairchem" / "core" / "module.py").read_text() == (
        "VALUE = 'committed'\n"
    )
    assert (snapshot / "packages" / "fairchem-core" / "pyproject.toml").is_file()
    assert not (snapshot / "README.md").exists()
    assert source_provenance(snapshot) == (commit, False)
    assert {path.name for path in tmp_path.iterdir()} == {"repo", "source"}


def test_snapshot_refuses_to_replace_an_existing_directory(repository, tmp_path):
    commit, _dirty = git_provenance(repository)
    (tmp_path / "source").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        create_source_snapshot(repository, commit, tmp_path / "source")


def test_failed_snapshot_leaves_no_partial_directory(repository, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        create_source_snapshot(repository, "0" * 40, tmp_path / "snapshots" / "one")

    assert list((tmp_path / "snapshots").iterdir()) == []


def test_source_provenance_inspects_the_given_checkout(repository, tmp_path):
    commit, _dirty = git_provenance(repository)

    assert source_provenance(repository) == (commit, False)
    (repository / "src" / "fairchem" / "core" / "new.py").write_text("")
    assert source_provenance(repository) == (commit, True)
    assert source_provenance(tmp_path / "not-a-checkout") == (None, None)


def test_source_marker_takes_precedence_over_git(repository):
    (repository / SOURCE_MARKER).write_text('{"git_commit": "snapshot-commit"}\n')

    assert source_provenance(repository) == ("snapshot-commit", False)
