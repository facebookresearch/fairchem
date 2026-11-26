"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Slow-to-import dependencies that should only be imported by the features that
# use them, not when importing fairchem.core or its inference and training APIs.
LAZY_DEPENDENCIES = {"nvalchemiops", "ray", "wandb", "warp"}


def _imported_modules(statement: str) -> set[str]:
    """
    Return the modules imported by running a statement in a fresh interpreter.
    """
    code = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout.splitlines()[-1]))


def test_import_fairchem_core_is_lightweight():
    modules = _imported_modules("import fairchem.core")
    assert "torch" not in modules
    assert "fairchem.core.calculate" not in modules


@pytest.mark.parametrize(
    "statement",
    [
        "from fairchem.core import FAIRChemCalculator, pretrained_mlip",
        "from fairchem.core.units.mlip_unit.mlip_unit import MLIPTrainEvalUnit",
    ],
)
def test_slow_dependencies_are_imported_lazily(statement):
    modules = _imported_modules(statement)
    imported = {name.split(".")[0] for name in modules} & LAZY_DEPENDENCIES
    assert not imported, f"{statement!r} imported {sorted(imported)}"
