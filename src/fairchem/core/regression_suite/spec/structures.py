"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ase.io
import numpy as np
from ase import Atoms

from fairchem.core.regression_suite.spec.hashing import json_value, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    CaseSpec,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def atoms_hash(atoms: Atoms) -> str:
    """
    Return a stable hash of model-relevant ASE structure data.
    """
    payload = {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64).tolist(),
        "positions": np.asarray(atoms.positions, dtype=np.float64).tolist(),
        "cell": np.asarray(atoms.cell, dtype=np.float64).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "tags": np.asarray(atoms.get_tags(), dtype=np.int64).tolist(),
        "constraints": [
            json_value(constraint.todict()) for constraint in atoms.constraints
        ],
        "charge": json_value(atoms.info.get("charge")),
        "spin": json_value(atoms.info.get("spin")),
    }
    return stable_hash(payload)


def frame_identities(source: str | Path | Iterable[Atoms]) -> list[tuple[str, str]]:
    """
    Return the ``structure_id`` and plan hash of each frame of a structure file
    or sequence of generated structures.
    """
    frames = (
        ase.io.read(source, index=":") if isinstance(source, (str, Path)) else source
    )
    return [(str(atoms.info["structure_id"]), atoms_hash(atoms)) for atoms in frames]


def load_source_atoms(case: CaseSpec, suite_directory: str | Path) -> Atoms:
    """
    Load and hash-check the source structure for one plan case.
    """
    relative_path = Path(case.structure)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RegressionSchemaError("structure path must stay within its suite")
    path = Path(suite_directory) / relative_path
    atoms = ase.io.read(path, index=case.structure_index)
    if not isinstance(atoms, Atoms):
        raise RegressionSchemaError(f"structure did not resolve to Atoms: {path}")
    return check_source_atoms(atoms, case)


def check_source_atoms(atoms: Atoms, case: CaseSpec) -> Atoms:
    """
    Check a loaded or generated source structure against its plan hash.
    """
    if atoms_hash(atoms) != case.structure_hash:
        raise RegressionSchemaError(f"structure hash differs for {case.case_id!r}")
    atoms.calc = None
    return atoms
