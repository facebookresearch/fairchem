"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

SOLVENT_DESCRIPTOR_ORDER = [
    "n",
    "alpha",
    "beta",
    "gamma",
    "epsilon",
    "aromaticity",
    "en-halogen",
]

SOLVENT_DIM = len(SOLVENT_DESCRIPTOR_ORDER) + 1

_SOLVENT_STATS = {
    "n": {"transform": "shift1", "scale": 0.068400},
    "alpha": {"transform": "linear", "scale": 0.181746},
    "beta": {"transform": "linear", "scale": 0.234892},
    "gamma": {"transform": "linear", "scale": 11.688257},
    "epsilon": {"transform": "born", "scale": 0.165059},
    "aromaticity": {"transform": "linear", "scale": 0.322355},
    "en-halogen": {"transform": "linear", "scale": 0.174984},
}

_EPS_TRANSFORM_ENV = os.environ.get("FAIRCHEM_SOLVENT_EPS_TRANSFORM", "log")
if _EPS_TRANSFORM_ENV == "born":
    _SOLVENT_STATS["epsilon"] = {"transform": "born", "scale": 0.165059}
elif _EPS_TRANSFORM_ENV != "log":
    raise ValueError(
        f"FAIRCHEM_SOLVENT_EPS_TRANSFORM must be 'log' or 'born', "
        f"got {_EPS_TRANSFORM_ENV!r}"
    )


def _transform(name: str, value: float) -> float:
    """Apply a descriptor's vacuum-anchoring transform (0 at the gas phase)."""
    transform = _SOLVENT_STATS[name]["transform"]
    if transform == "shift1":
        return float(value) - 1.0
    if transform == "log":
        return math.log(value)
    if transform == "born":
        return 1.0 - 1.0 / float(value)
    return float(value)

# Names that map to the vacuum / gas-phase null vector instead of a lookup.
_VACUUM_NAMES = {"", "vacuum", "gas", "gas_phase", "gas-phase", "none"}


@functools.lru_cache(maxsize=1)
def _load_raw() -> dict:
    """
    Load the packaged Minnesota Solvent Descriptor Database.

    Returns:
        The parsed ``solvent_descriptors.json`` contents.
    """
    resource = importlib.resources.files("fairchem.core.datasets").joinpath(
        "solvent_descriptors.json"
    )
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _solvents_lower() -> dict:
    """
    Solvent table keyed by lowercased name for case-insensitive lookup.

    JSON keys may carry uppercase (e.g. ``"dimethyl sulfoxide (DMSO)"``), while
    lookup names are lowercased, so they would otherwise miss and fall back to
    the vacuum vector.

    Returns:
        Mapping from lowercased solvent name to its descriptor dict.
    """
    return {k.lower(): v for k, v in _load_raw()["solvents"].items()}


def list_solvents() -> list[str]:
    """
    Return the sorted list of solvent names with solvent descriptors.

    Returns:
        Solvent name keys available for lookup.
    """
    return sorted(_load_raw()["solvents"].keys())


def normalize(raw_vec: Sequence[float]) -> list[float]:
    """
    Normalize a raw solvent descriptor vector (vacuum-anchored).

    Applies each descriptor's vacuum-anchoring transform, then divides by the
    baked scale from ``_SOLVENT_STATS``. There is no mean subtraction: the
    physical gas phase maps to exactly 0 in every channel.

    Args:
        raw_vec: Raw descriptor values in ``SOLVENT_DESCRIPTOR_ORDER`` order.

    Returns:
        The normalized descriptor values.
    """
    if len(raw_vec) != len(SOLVENT_DESCRIPTOR_ORDER):
        raise ValueError(
            f"raw_vec must have {len(SOLVENT_DESCRIPTOR_ORDER)} values, "
            f"got {len(raw_vec)}"
        )
    return [
        _transform(name, value) / _SOLVENT_STATS[name]["scale"]
        for name, value in zip(SOLVENT_DESCRIPTOR_ORDER, raw_vec)
    ]


def get_solvent_vector(solvent_name: str | None, strict: bool = True) -> torch.Tensor:
    """
    Build the solvent conditioning vector for a solvent.

    Args:
        solvent_name: Solvent name to look up. ``None``, an empty string, or a
            vacuum alias (``"vacuum"``, ``"gas"``, ...) returns the null vector.
        strict: If True, raise ``KeyError`` for an unknown solvent; otherwise
            log a warning and return the null vector.

    Returns:
        A ``(1, SOLVENT_DIM)`` float32 tensor: seven normalized descriptors
        followed by a solvent-present mask (1.0 for a real solvent, 0.0 for
        vacuum).
    """
    vec = torch.zeros(1, SOLVENT_DIM, dtype=torch.float32)
    if solvent_name is None:
        return vec

    key = str(solvent_name).strip().lower()
    if key in _VACUUM_NAMES:
        return vec

    solvents = _solvents_lower()
    if key not in solvents:
        if strict:
            raise KeyError(
                f"Unknown solvent '{solvent_name}'. Use strict=False to fall "
                f"back to the vacuum vector, or see list_solvents()."
            )
        logging.warning(
            "Unknown solvent '%s'; using the vacuum solvent vector.", solvent_name
        )
        return vec

    raw = [solvents[key][name] for name in SOLVENT_DESCRIPTOR_ORDER]
    vec[0, : len(SOLVENT_DESCRIPTOR_ORDER)] = torch.tensor(
        normalize(raw), dtype=torch.float32
    )
    vec[0, len(SOLVENT_DESCRIPTOR_ORDER)] = 1.0
    return vec


def _recompute_stats() -> dict:
    """
    Recompute the normalization statistics directly from the packaged JSON.

    Used only by the regression test to verify the baked ``_SOLVENT_STATS``
    stays in sync with ``solvent_descriptors.json``.

    Returns:
        A mapping with the same structure as ``_SOLVENT_STATS``.
    """
    solvents = _load_raw()["solvents"]
    stats = {}
    for name in SOLVENT_DESCRIPTOR_ORDER:
        transform = _SOLVENT_STATS[name]["transform"]
        values = [_transform(name, s[name]) for s in solvents.values()]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats[name] = {"transform": transform, "scale": math.sqrt(var)}
    return stats
