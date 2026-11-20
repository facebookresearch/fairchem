"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fairchem.core.regression_suite.artifacts.storage import (
    pack_baseline,
    sealed_baseline_path,
    write_json_atomic,
)
from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.schema import BASELINE_SCHEMA_VERSION

if TYPE_CHECKING:
    from pathlib import Path


def sealed_baseline(directory: Path, name: str, device: str = "cpu") -> Path:
    """
    Seal a valid baseline without suites and return its file.
    """
    run = directory / name
    run.mkdir(parents=True)
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "checkpoint": "uma-s-1p2p1",
        "campaign": "standard",
        "created_at": "2026-11-05T03:25:37+00:00",
        "device": device,
        "scope": "full",
        "suites": {},
    }
    write_json_atomic(
        run / "baseline.json", {**payload, "baseline_id": stable_hash(payload)}
    )
    return pack_baseline(run, sealed_baseline_path(run))
