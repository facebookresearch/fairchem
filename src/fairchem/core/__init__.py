# flake8: noqa: E402
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version

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
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    __version__ = ""

__all__ = [
    "FAIRChemCalculator",
    "pretrained_mlip",
    "clear_cache",
]
