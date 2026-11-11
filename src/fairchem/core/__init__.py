# flake8: noqa: E402
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# TODO: Remove this warning filter when torchtnt fixes pkg_resources deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=(
        "pkg_resources is deprecated as an API. "
        "See https://setuptools.pypa.io/en/latest/pkg_resources.html."
    ),
    category=UserWarning,
)

from fairchem.core._config import clear_cache

if TYPE_CHECKING:
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.calculate.dftd3_calculator import DFTD3Calculator

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    __version__ = ""

# The calculator API is imported on first access, so that importing any
# fairchem.core submodule does not load the full inference stack.
_LAZY_EXPORTS = {
    "FAIRChemCalculator": "fairchem.core.calculate.ase_calculator",
    "DFTD3Calculator": "fairchem.core.calculate.dftd3_calculator",
}

__all__ = [
    "FAIRChemCalculator",
    "DFTD3Calculator",
    "pretrained_mlip",
    "clear_cache",
]


def __getattr__(name: str):
    if name == "pretrained_mlip":
        value = importlib.import_module("fairchem.core.calculate.pretrained_mlip")
    elif name in _LAZY_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
