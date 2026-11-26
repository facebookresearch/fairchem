"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from ase.vibrations import Vibrations, VibrationsData

from fairchem.core.regression_suite.spec.plan import oriented_case
from fairchem.core.regression_suite.spec.structures import frame_identities

# The molecules are shared with the physical-constraint suite.
_STRUCTURES = frame_identities(
    Path(__file__).parents[1] / "physical_constraints/structures/molecules.extxyz.xz"
)


def vibration_cases():
    """
    Return the shared, frozen molecule and orientation cases.
    """
    return [
        oriented_case(
            index,
            orientation,
            case_id=f"vibrations/{structure_id}/orientation-{orientation:02d}",
            metadata={
                "finite_difference_step": 0.01,
                "kind": "vibration",
                "structure_id": structure_id,
            },
            runtime_seed=42,
            structure="molecules.extxyz.xz",
            structure_hash=structure_hash,
            structure_index=index,
            task_name="omol",
        )
        for index, (structure_id, structure_hash) in enumerate(_STRUCTURES)
        for orientation in range(9)
    ]


def signed_mode_frequencies(frequencies) -> np.ndarray:
    """
    Sort all modes, representing imaginary frequencies as negative magnitudes.
    """
    values = np.asarray(frequencies, dtype=np.complex128).reshape(-1)
    signed = np.where(
        np.abs(values.imag) < 1.0e-6,
        values.real,
        -np.abs(values.imag),
    )
    return np.sort(signed.astype(np.float64, copy=False))


def finite_difference_frequencies(atoms, delta: float) -> np.ndarray:
    """
    Compute mode frequencies from an ASE finite-difference Hessian.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        vibrations = Vibrations(
            atoms,
            name=os.path.join(temporary_directory, "vibrations"),
            delta=delta,
        )
        vibrations.run()
        frequencies = vibrations.get_vibrations().get_frequencies()
    return signed_mode_frequencies(frequencies)


def autograd_frequencies(atoms) -> np.ndarray:
    """
    Compute mode frequencies from the attached calculator's autograd Hessian.
    """
    atom_count = len(atoms)
    hessian = (
        np.asarray(atoms.calc.get_property("hessian", atoms))
        .squeeze()
        .reshape(3 * atom_count, 3 * atom_count)
    )
    frequencies = VibrationsData.from_2d(atoms, hessian).get_frequencies()
    return signed_mode_frequencies(frequencies)
