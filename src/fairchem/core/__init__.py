"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fairchem.core._config import clear_cache
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""

__all__ = ["FAIRChemCalculator", "pretrained_mlip", "clear_cache"]
