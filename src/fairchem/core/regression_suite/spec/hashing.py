"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import zipfile


def canonical_json(value: Any) -> str:
    """
    Serialize a JSON-compatible value deterministically.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_hash(value: Any) -> str:
    """
    Return the SHA-256 digest of a JSON-compatible value.
    """
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def json_value(value: Any) -> Any:
    """
    Convert NumPy values, sets, tuples, and mappings to JSON-compatible data.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(json_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def file_sha256(path: str | Path | zipfile.Path) -> str:
    """
    Return the SHA-256 digest of a file without loading it all into memory.
    """
    with (Path(path) if isinstance(path, str) else path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
