"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import socket
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.artifacts.storage import (
    baseline_run_directory,
    load_sealed,
    open_artifact,
)
from fairchem.core.regression_suite.spec.schema import (
    BASELINE_SCHEMA_VERSION,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_OPTIONAL_PROVENANCE = ("fairchem_version", "torch_version", "device_name")


def point_current_baseline(baseline: str | Path, link: str | Path) -> None:
    """
    Atomically point the ``current`` baseline link at a finalized baseline.
    """
    target = Path(baseline).resolve()
    link = Path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = target.name if target.parent == link.parent.resolve() else str(target)
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
    temporary.symlink_to(relative)
    os.replace(temporary, link)


class BaselineUnavailableError(RuntimeError):
    """
    Raised when a candidate's baseline cannot be read or is not the one expected.
    """


def check_baseline(
    path: str | Path, baseline_id: str | None = None, checkpoint: str | None = None
) -> dict[str, Any]:
    """
    Load a baseline from this host and confirm its identity before any work.

    Given ``checkpoint``, the baseline must have been built from it. Names and
    paths compare by stem, and the baseline's file name is never trusted.
    """
    try:
        baseline = load_baseline(open_artifact(path) / "baseline.json")
    except Exception as error:
        raise BaselineUnavailableError(
            f"baseline {path} is not available on {socket.gethostname()}: {error}"
        ) from error
    if baseline_id is not None and baseline["baseline_id"] != baseline_id:
        raise BaselineUnavailableError(
            f"baseline {path} on {socket.gethostname()} is "
            f"{baseline['baseline_id']}, not the launched {baseline_id}"
        )
    built_from = str(baseline["checkpoint"])
    if checkpoint is not None and Path(built_from).stem != Path(checkpoint).stem:
        raise BaselineUnavailableError(
            f"baseline {path} was built from checkpoint {built_from}, not the "
            f"requested {checkpoint}; pass --baseline for {checkpoint} or "
            f"--checkpoint {built_from}"
        )
    return baseline


def load_baseline(path: str | Path) -> dict[str, Any]:
    """
    Load a baseline index and verify its content-addressed identity.

    Baselines finalized before they recorded a device and scope ran every
    case on CUDA.
    """
    payload = load_sealed(path, "baseline_id", "baseline identity")
    if payload["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise RegressionSchemaError(
            "baseline schema version is unsupported; re-run finalization "
            "(existing numerical fragments are reusable)"
        )
    payload.setdefault("device", "cuda")
    payload.setdefault("scope", "full")
    return payload


def baseline_provenance(
    baseline: Mapping[str, Any],
    path: str | Path,
    suites: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Describe where a baseline came from, for every place that shows it.

    ``commits`` maps each source commit to the suites built from it, over
    ``suites`` or every suite. Versions and the device name appear only for
    baselines sealed with them.
    """
    commits: dict[str, list[str]] = {}
    for suite in sorted(baseline["suites"] if suites is None else suites):
        commits.setdefault(baseline["suites"][suite]["git_commit"], []).append(suite)
    return {
        "name": baseline_run_directory(path).name,
        "baseline_id": baseline["baseline_id"],
        "created_at": baseline.get("created_at"),
        "checkpoint": baseline["checkpoint"],
        "campaign": baseline["campaign"],
        "scope": baseline["scope"],
        "device": baseline["device"],
        "commits": dict(sorted(commits.items())),
        **{key: baseline[key] for key in _OPTIONAL_PROVENANCE if key in baseline},
    }
