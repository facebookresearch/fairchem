"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np

from fairchem.core.regression_suite.spec.plan import oriented_case
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.suites.base import (
    ObservableDescription,
    Suite,
)

if TYPE_CHECKING:
    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase


class _Kind(NamedTuple):
    kind: str
    group: str
    task_name: str
    filename: str
    # Structure i in orientation o draws its directions from the seed
    # seed + stride * i + 100 * o.
    seed: int
    stride: int
    # Finite-difference settings, copied into each case's metadata.
    parameters: dict[str, float]


_KINDS = (
    _Kind(
        kind="molecular",
        group="molecule",
        task_name="omol",
        filename="molecules.extxyz.xz",
        seed=42,
        stride=10_000,
        parameters={"direction_count": 4, "finite_difference_step": 0.01},
    ),
    _Kind(
        kind="periodic",
        group="periodic",
        task_name="omat",
        filename="periodic.extxyz.xz",
        seed=20_042,
        stride=1,
        parameters={
            "direction_count": 3,
            "fc2_step": 0.01,
            "fc3_step": 0.03,
            "strain_step": 0.0005,
        },
    ),
)


_STENCIL = (2.0, 1.0, -1.0, -2.0)


def _five_point_derivative(atoms, structures, step: float) -> float:
    """
    Differentiate the energy over the cells and positions displaced by the
    ``_STENCIL`` multiples of ``step``, then restore the structure.
    """
    cell, positions = atoms.get_cell(), atoms.get_positions()
    energies = []
    try:
        for displaced_cell, displaced_positions in structures:
            atoms.set_cell(displaced_cell, scale_atoms=False)
            atoms.set_positions(displaced_positions)
            energies.append(float(atoms.get_potential_energy()))
    finally:
        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(positions)
    energy_p2, energy_p1, energy_m1, energy_m2 = energies
    return (-energy_p2 + 8.0 * energy_p1 - 8.0 * energy_m1 + energy_m2) / (12.0 * step)


def _force_derivative(atoms, positions, direction, step, center=None) -> np.ndarray:
    """
    Return the central difference of the negated forces along a direction, or
    their second difference about the ``center`` forces.
    """
    try:
        atoms.set_positions(positions + step * direction)
        plus = np.asarray(atoms.get_forces(), dtype=np.float64)
        atoms.set_positions(positions - step * direction)
        minus = np.asarray(atoms.get_forces(), dtype=np.float64)
    finally:
        atoms.set_positions(positions)
    if center is None:
        return -(plus - minus) / (2.0 * step)
    return -(plus - 2.0 * center + minus) / step**2


def _step_errors(atoms, positions, directions, step: float, center=None):
    """
    Return the derivatives at half ``step`` and their largest change at ``step``.
    """
    fine, coarse = (
        [
            _force_derivative(atoms, positions, direction, size, center)
            for direction in directions
        ]
        for size in (step / 2.0, step)
    )
    return fine, [
        float(np.max(np.abs(coarse_value - fine_value)))
        for coarse_value, fine_value in zip(coarse, fine)
    ]


def _molecular_measurements(atoms, case: SuiteCase) -> dict[str, np.ndarray]:
    metadata = case.spec.metadata
    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    cell = atoms.get_cell()
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    step = float(metadata["finite_difference_step"])
    rng = np.random.default_rng(int(metadata["direction_seed"]))
    errors = []
    for _ in range(int(metadata["direction_count"])):
        direction = rng.standard_normal(positions.shape)
        direction /= np.linalg.norm(direction)
        finite_difference = _five_point_derivative(
            atoms,
            [(cell, positions + multiple * step * direction) for multiple in _STENCIL],
            step,
        )
        analytic = -float(np.sum(forces * direction))
        errors.append(abs(finite_difference - analytic))
    net_force = np.sqrt(np.mean(np.square(forces.sum(axis=0))))
    return {
        "energy_force_consistency": np.asarray(errors),
        "net_force_conservation": np.asarray(net_force),
    }


def _periodic_measurements(atoms, case: SuiteCase) -> dict[str, np.ndarray]:
    metadata = case.spec.metadata
    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    cell = np.asarray(atoms.cell, dtype=np.float64).copy()
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
    step = float(metadata["strain_step"])
    stress_errors = []
    for row, column in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        deformations = []
        for multiple in _STENCIL:
            strain = np.zeros((3, 3))
            strain[row, column] = strain[column, row] = multiple * step
            deformations.append(np.eye(3) + strain)
        derivative = _five_point_derivative(
            atoms,
            [(cell @ matrix.T, positions @ matrix.T) for matrix in deformations],
            step,
        )
        shear_factor = 2.0 if row != column else 1.0
        stress_errors.append(
            abs(
                derivative / (abs(np.linalg.det(cell)) * shear_factor)
                - stress[row, column]
            )
        )

    rng = np.random.default_rng(int(metadata["direction_seed"]))
    directions = []
    for _ in range(int(metadata["direction_count"])):
        direction = rng.standard_normal(positions.shape)
        direction -= direction.mean(axis=0, keepdims=True)
        direction /= np.linalg.norm(direction)
        directions.append(direction)
    fc2_fine, fc2_step_errors = _step_errors(
        atoms, positions, directions, float(metadata["fc2_step"])
    )
    symmetry_errors = [
        abs(
            float(np.sum(directions[first] * fc2_fine[second]))
            - float(np.sum(directions[second] * fc2_fine[first]))
        )
        for first in range(len(directions))
        for second in range(first + 1, len(directions))
    ]
    center_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    _fc3_fine, fc3_step_errors = _step_errors(
        atoms, positions, directions, float(metadata["fc3_step"]), center_forces
    )
    return {
        "stress_finite_difference_error": np.asarray(stress_errors),
        "fc2_symmetry_error": np.asarray(symmetry_errors),
        "fc2_step_error": np.asarray(fc2_step_errors),
        "fc3_step_error": np.asarray(fc3_step_errors),
    }


class PhysicalConstraintsSuite(Suite):
    """
    Measure force derivatives, force conservation, stress, and force constants.
    """

    name = "physical-constraints"
    explanation = (
        (
            "What it checks",
            "Whether the candidate's outputs still obey the rules that tie them "
            "together. Forces should be the slope of the energy. Stress should "
            "be the slope of the energy under strain. Forces on an isolated "
            "molecule should add up to zero. Force constants, which say how "
            "forces change when atoms move, should be symmetric and settle as "
            "the step shrinks. MD and phonons rely on these rules.",
        ),
        (
            "The cases",
            "Five small molecules of 5 or 6 atoms run with the omol task. They "
            "are dimers from the NCIAtlas set, pairs of molecules held together "
            "by weak forces. Five small crystals of 4 or 5 atoms run with the "
            "omat task. Every structure runs in its original orientation and in "
            "eight random rotations, which gives 90 cases, 45 of each kind.",
        ),
        (
            "Molecule checks",
            "For each molecule, the suite picks 4 random directions that move "
            "all atoms at once. Along each, it estimates the slope of the "
            "energy from four energies, at displacements of 0.01 and 0.02 Å "
            "either way. It compares that slope with the one the forces "
            "predict. It also adds up all the forces and records the size of "
            "the total.",
        ),
        (
            "Crystal checks",
            "For each crystal, the cell is strained in each of its six "
            "independent ways, by up to 0.1%. The slope of the energy is "
            "compared with the predicted stress. The atoms are then moved along "
            "3 random directions. The suite records how much the change in "
            "forces, and its curvature, shift when the step is halved from "
            "0.01 Å and from 0.03 Å. It also checks the response is symmetric "
            "between pairs of directions.",
        ),
        (
            "How it is graded",
            "These errors are not zero even in FP64, because slopes from small "
            "steps are approximate. So the suite grades the absolute difference "
            "between the candidate's error and FP64's error on the same case. "
            "A cluster is one structure in one orientation. Each of the six "
            "checks is averaged per cluster. The median and 95th percentile "
            "must stay within twice the baseline's values.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "energy_force_consistency": ObservableDescription(
            "Gap between the energy's slope along 4 random directions and the slope the forces predict, in eV/Å",
            "one molecule in one orientation (molecules only)",
        ),
        "net_force_conservation": ObservableDescription(
            "Size of the total force on the molecule, which should be zero, as the RMS of its x, y, and z parts, in eV/Å",
            "one molecule in one orientation (molecules only)",
        ),
        "stress_finite_difference_error": ObservableDescription(
            "Gap between the energy's slope under each of 6 strains and the predicted stress, in eV/Å³",
            "one crystal in one orientation (crystals only)",
        ),
        "fc2_symmetry_error": ObservableDescription(
            "How far the force response is from symmetric between pairs of 3 random directions, in eV/Å²",
            "one crystal in one orientation (crystals only)",
        ),
        "fc2_step_error": ObservableDescription(
            "Largest change in the force's first derivative when the step shrinks from 0.01 to 0.005 Å, in eV/Å²",
            "one crystal in one orientation (crystals only)",
        ),
        "fc3_step_error": ObservableDescription(
            "Largest change in the force's second derivative when the step shrinks from 0.03 to 0.015 Å, in eV/Å³",
            "one crystal in one orientation (crystals only)",
        ),
    }
    plan_hash = "6be4bfb0196077000dbea904253cb3260dd759d47b9470988daea6d9db007004"
    smoke_case_ids = (
        "physical/molecule/5.0122/orientation-00",
        "physical/periodic/66415/orientation-00",
    )
    observables: ClassVar[dict[str, ObservableSpec]] = {
        "energy_force_consistency": ObservableSpec(
            units="eV/A", comparison="absolute", histogram=HistogramSpec(1e-12, 10.0)
        ),
        "net_force_conservation": ObservableSpec(
            units="eV/A", comparison="absolute", histogram=HistogramSpec(1e-14, 10.0)
        ),
        "stress_finite_difference_error": ObservableSpec(
            units="eV/A^3", comparison="absolute", histogram=HistogramSpec(1e-12, 10.0)
        ),
        "fc2_symmetry_error": ObservableSpec(
            units="eV/A^2", comparison="absolute", histogram=HistogramSpec(1e-12, 100.0)
        ),
        "fc2_step_error": ObservableSpec(
            units="eV/A^2", comparison="absolute", histogram=HistogramSpec(1e-12, 100.0)
        ),
        "fc3_step_error": ObservableSpec(
            units="eV/A^3", comparison="absolute", histogram=HistogramSpec(1e-12, 1e3)
        ),
    }

    def cases(self) -> list[CaseSpec]:
        structures = Path(__file__).parent / "structures"
        cases = [
            oriented_case(
                index,
                orientation,
                case_id=f"physical/{group}/{structure_id}/orientation-{orientation:02d}",
                metadata={
                    **parameters,
                    "direction_seed": seed + stride * index + 100 * orientation,
                    "kind": kind,
                    "structure_id": structure_id,
                },
                runtime_seed=42,
                structure=filename,
                structure_hash=structure_hash,
                structure_index=index,
                task_name=task_name,
            )
            for kind, group, task_name, filename, seed, stride, parameters in _KINDS
            for index, (structure_id, structure_hash) in enumerate(
                frame_identities(structures / filename)
            )
            for orientation in range(9)
        ]
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        atoms = case.transformed_atoms.copy()
        atoms.calc = calculator
        if case.spec.metadata["kind"] == "molecular":
            return _molecular_measurements(atoms, case)
        return _periodic_measurements(atoms, case)
