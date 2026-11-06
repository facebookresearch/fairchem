"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar

import huggingface_hub
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RevisionNotFoundError,
)

from fairchem.core._config import CACHE_DIR
from fairchem.core.regression_suite.artifacts.baselines import (
    BaselineUnavailableError,
    load_baseline,
)
from fairchem.core.regression_suite.artifacts.storage import (
    BASELINE_SUFFIX,
    open_artifact,
)
from fairchem.core.regression_suite.spec.hashing import file_sha256

if TYPE_CHECKING:
    from collections.abc import Sequence

BASELINE_ARTIFACT_TYPE = "regression-baseline"


@dataclass(frozen=True)
class ResolvedBaseline:
    """
    A baseline available on this host and the pinned reference it came from.

    ``source`` resolves to the same file again: the Hugging Face commit that
    was downloaded, the concrete W&B version, or the absolute local path.
    """

    path: Path
    source: str


@dataclass(frozen=True)
class HuggingFaceReference:
    """
    A file in a Hugging Face model repository, at a revision or ``main``.
    """

    form: ClassVar[str] = "hf:ORG/REPO/PATH[@REVISION]"
    repo_id: str
    filename: str
    revision: str | None = None

    def __str__(self) -> str:
        revision = f"@{self.revision}" if self.revision else ""
        return f"hf:{self.repo_id}/{self.filename}{revision}"

    @classmethod
    def parse(cls, location: str) -> HuggingFaceReference | None:
        path, at, revision = location.partition("@")
        parts = path.split("/")
        if len(parts) < 3 or "" in parts or (at and not revision):
            return None
        return cls("/".join(parts[:2]), "/".join(parts[2:]), revision or None)


@dataclass(frozen=True)
class WandBReference:
    """
    A published W&B artifact version, by alias (default ``latest``).
    """

    form: ClassVar[str] = "wandb:[ENTITY/PROJECT/]NAME[:ALIAS]"
    name: str
    alias: str = "latest"
    project: str | None = None

    @classmethod
    def parse(cls, location: str) -> WandBReference | None:
        path, colon, alias = location.partition(":")
        *project, name = path.split("/")
        if not name or "" in project or len(project) not in (0, 2):
            return None
        if colon and not alias:
            return None
        return cls(name, alias or "latest", "/".join(project) or None)


@dataclass(frozen=True)
class LocalReference:
    """
    A sealed file or run directory, by path or by name under the baselines
    directory; ``current`` is the most recently finalized local baseline.
    """

    location: str


BaselineReference = HuggingFaceReference | WandBReference | LocalReference

_SOURCES = {"hf": HuggingFaceReference, "wandb": WandBReference}


def parse_baseline_reference(reference: str) -> BaselineReference:
    """
    Parse a baseline reference; a prefix from the source table selects a
    remote source and anything else is local.

    Raises:
        BaselineUnavailableError: If a remote reference is malformed.
    """
    scheme, colon, location = reference.partition(":")
    source = _SOURCES.get(scheme) if colon else None
    if source is None:
        return LocalReference(reference)
    parsed = source.parse(location)
    if parsed is None:
        raise BaselineUnavailableError(
            f"malformed baseline reference {reference!r}; expected {source.form}"
        )
    return parsed


def resolve_baseline(
    reference: str,
    baselines: Path,
    *,
    wandb_project: str | None = None,
    wandb_no_proxy_hosts: Sequence[str] = (),
) -> ResolvedBaseline:
    """
    Make the referenced baseline available on this host.

    Remote baselines are downloaded into ``baselines`` once, beside local ones,
    so compute nodes read them from the artifact root and their verification
    reports land in it. Callers still confirm identity with ``check_baseline``.

    Args:
        reference: A baseline reference; see ``parse_baseline_reference``.
        baselines: The artifact root's baselines directory.
        wandb_project: ``ENTITY/PROJECT`` of W&B names given without one.
        wandb_no_proxy_hosts: Hosts that W&B downloads reach without a proxy.

    Raises:
        BaselineUnavailableError: If the reference is malformed or the
            baseline cannot be found or downloaded.
    """
    parsed = parse_baseline_reference(reference)
    try:
        match parsed:
            case HuggingFaceReference():
                return _fetch_huggingface(parsed, baselines)
            case WandBReference():
                add_no_proxy_hosts(wandb_no_proxy_hosts)
                return _fetch_wandb(parsed, baselines, wandb_project)
            case LocalReference():
                return _resolve_local(parsed, baselines)
    except BaselineUnavailableError:
        raise
    except Exception as error:
        raise BaselineUnavailableError(
            f"baseline {reference} is not available: {error}"
        ) from error


def _resolve_local(reference: LocalReference, baselines: Path) -> ResolvedBaseline:
    configured = reference.location
    given = Path(configured).expanduser()
    options = [
        given,
        baselines / f"{configured}{BASELINE_SUFFIX}",
        baselines / configured,
    ]
    path = next((option for option in options if option.exists()), given)
    # Sealed files are the norm; run directories from before sealing still load.
    if not (path.is_file() or (path / "baseline.json").is_file()):
        raise BaselineUnavailableError(
            f"no finalized baseline at {path}; run `fairchem-regression baseline` "
            "or pass --baseline"
        )
    path = path.resolve()
    return ResolvedBaseline(path, str(path))


def _fetch_huggingface(
    reference: HuggingFaceReference, baselines: Path
) -> ResolvedBaseline:
    try:
        cached = Path(
            huggingface_hub.hf_hub_download(
                reference.repo_id,
                reference.filename,
                revision=reference.revision,
                cache_dir=CACHE_DIR,
            )
        )
    except (RemoteEntryNotFoundError, RevisionNotFoundError) as error:
        raise BaselineUnavailableError(
            f"no baseline is published at {reference}; pass --baseline "
            "<local file>, or pin a revision that has one"
        ) from error
    except LocalEntryNotFoundError as error:
        raise BaselineUnavailableError(
            f"{reference} is not in the Hugging Face cache and the Hub is not "
            "reachable (or HF_HUB_OFFLINE is set); download it once with "
            "network access or pass --baseline <local file>"
        ) from error
    # The cache keeps each file under snapshots/<commit>/<path in repo>.
    snapshot = cached.parents[len(PurePosixPath(reference.filename).parts) - 1]
    if snapshot.parent.name != "snapshots":
        raise BaselineUnavailableError(f"{cached} is not in a Hugging Face cache")
    # Each revision gets its own file, so a later download of another revision
    # never replaces one that queued jobs still read, nor a local baseline.
    name = PurePosixPath(cached.name)
    destination = baselines / f"{name.stem}@{snapshot.name[:8]}{name.suffix}"
    if not destination.is_file() or file_sha256(destination) != file_sha256(cached):
        baselines.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=baselines) as temporary:
            copy = Path(temporary) / destination.name
            shutil.copyfile(cached, copy)
            copy.replace(destination)
    return ResolvedBaseline(
        destination, f"hf:{reference.repo_id}/{reference.filename}@{snapshot.name}"
    )


def _fetch_wandb(
    reference: WandBReference, baselines: Path, default_project: str | None
) -> ResolvedBaseline:
    """
    Download a published baseline once and verify it against its metadata.
    """
    project = reference.project or default_project
    name = f"{project}/{reference.name}" if project else reference.name
    api = importlib.import_module("wandb").Api()
    artifact = api.artifact(f"{name}:{reference.alias}", type=BASELINE_ARTIFACT_TYPE)
    metadata = artifact.metadata
    file_name = f"{metadata['baseline_name']}{BASELINE_SUFFIX}"
    # As for Hugging Face, each version gets its own file beside local ones.
    destination = (
        baselines / f"{metadata['baseline_name']}@{artifact.version}{BASELINE_SUFFIX}"
    )
    if not destination.is_file() or file_sha256(destination) != metadata["sha256"]:
        baselines.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=baselines) as temporary:
            downloaded = Path(artifact.download(root=temporary)) / file_name
            downloaded.replace(destination)
    if file_sha256(destination) != metadata["sha256"]:
        raise BaselineUnavailableError(f"downloaded baseline {destination} is corrupt")
    baseline = load_baseline(open_artifact(destination) / "baseline.json")
    if baseline["baseline_id"] != metadata["baseline_id"]:
        raise BaselineUnavailableError(
            f"{destination} is not baseline {metadata['baseline_id']}"
        )
    # Pin the concrete version, since aliases such as ``latest`` move.
    qualified = artifact.qualified_name.rsplit(":", 1)[0]
    return ResolvedBaseline(destination, f"wandb:{qualified}:{artifact.version}")


def add_no_proxy_hosts(hosts: Sequence[str]) -> None:
    """
    Let this process reach the given hosts without the forward proxy.
    """
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(variable, "").split(",") if item]
        entries.extend(host for host in hosts if host not in entries)
        os.environ[variable] = ",".join(entries)
