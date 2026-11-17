"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import ase.io
import numpy as np

from fairchem.core.components.calculate.recipes.phonons import (
    THz_to_K,
    get_phonopy_object,
    run_mdr_phonon_benchmark,
)
from fairchem.core.regression_suite.spec.hashing import json_value, stable_hash
from fairchem.core.regression_suite.spec.plan import unrotated_case
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import atoms_hash
from fairchem.core.regression_suite.suites.base import (
    ObservableDescription,
    Suite,
)

if TYPE_CHECKING:
    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase


def _metadata(atoms) -> dict:
    """
    Describe one MDR crystal's phonon protocol from its structure metadata.
    """
    supercell = json_value(atoms.info["supercell_matrix"])
    primitive = json_value(atoms.info.get("primitive_matrix"))
    primitive_volume = 1.0 if primitive is None else abs(np.linalg.det(primitive))
    return {
        "displacement_angstrom": 0.01,
        "fix_symmetry_relax": True,
        # Commensurate q-points by the primitive cell's vibrational modes.
        "frequency_shape": [
            round(abs(np.linalg.det(supercell))),
            round(3 * len(atoms) * primitive_volume),
        ],
        "imaginary_frequency_threshold_thz": -0.1,
        "kind": "phonon_crystal",
        "masses_hash": stable_hash(
            np.asarray(atoms.get_masses(), dtype=np.float64).tolist()
        ),
        "primitive_matrix": primitive,
        "run_relax": True,
        "structure_id": str(atoms.info["structure_id"]),
        "supercell_matrix": supercell,
        "symmetrize_force_constants": False,
        "symprec": 1.0e-4,
        "thermal_grid_kelvin": [75.0 * step for step in range(9)],
        "thermal_temperature_kelvin": 300.0,
    }


class PhononCrystalSuite(Suite):
    """
    Compare crystal-phonon spectra and thermal properties against FP64.
    """

    name = "phonon-crystal"
    explanation = (
        (
            "What it checks",
            "Whether the candidate gives the same lattice vibrations, called "
            "phonons, as FP64. Phonons come from small changes in force when "
            "atoms are nudged, so they magnify force noise. They feed heat "
            "capacity, entropy, and checks of whether a crystal is stable.",
        ),
        (
            "The cases",
            "Ten crystals from the MDR phonon set run with the omat task, "
            "without rotation, so there are ten cases. The smallest repeating "
            "unit of each crystal is first relaxed, atoms and cell, with its "
            "symmetry held fixed. It stops when forces are below 0.005 eV/Å, "
            "or after 500 steps. Fixing the symmetry keeps rounding noise from "
            "changing which atoms get nudged.",
        ),
        (
            "What is measured",
            "Phonopy builds a supercell, from 1x1x1 to 3x3x3 copies of the "
            "crystal, and moves one atom at a time by 0.01 Å. The resulting "
            "forces give force constants. From "
            "those, it computes every vibration frequency at the wave vectors "
            "that fit the supercell exactly. That gives 48 to 336 frequencies "
            "per crystal, sorted and stored in THz.",
        ),
        (
            "How it is graded",
            "Only the frequencies decide. Each is compared with the FP64 "
            "frequency at the same wave vector and rank, as an absolute "
            "difference in THz. A "
            "cluster is one crystal, and its differences are averaged into one "
            "number. The median and 95th percentile of the ten numbers must "
            "stay within twice the baseline's values. With ten crystals, the "
            "95th percentile sits close to the worst one.",
        ),
        (
            "Report-only values",
            "Entropy, heat capacity, and free energy at 300 K are report-only, "
            "as is the count of modes below -0.1 THz. Those modes are imaginary "
            "and signal instability. The thermal values use a 20x20x20 grid of "
            "wave vectors filled in by interpolation. In soft crystals, TF32 "
            "force noise creates imaginary modes that change from run to run. "
            "Repeated TF32 runs missed FP64's heat capacity for mp-17048 by "
            "anywhere from 0.03 to 1.3 J/K/mol.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "mode_frequency": ObservableDescription(
            "Every phonon frequency at the supercell's wave vectors, in THz, where negative means imaginary",
            "one crystal",
        ),
        "entropy_300k": ObservableDescription(
            "Vibrational entropy at 300 K from a 20x20x20 wave-vector grid, in J/K/mol",
            "one crystal",
            "TF32 noise makes repeat results vary severalfold",
        ),
        "heat_capacity_300k": ObservableDescription(
            "Vibrational heat capacity at 300 K from a 20x20x20 wave-vector grid, in J/K/mol",
            "one crystal",
            "TF32 noise makes repeat results vary severalfold",
        ),
        "free_energy_300k": ObservableDescription(
            "Vibrational free energy at 300 K, including zero-point energy, from a 20x20x20 wave-vector grid, in kJ/mol",
            "one crystal",
            "TF32 noise makes repeat results vary severalfold",
        ),
        "imaginary_mode_count": ObservableDescription(
            "Number of frequencies below -0.1 THz, a sign of an unstable crystal",
            "one crystal",
            "noise flips whole modes; the frequencies are graded",
        ),
    }
    plan_hash = "a1b47f86d98aedaed27e9222525f58adf34293993e876f6a2c90b7e4de567d7f"
    smoke_case_ids = ("phonon-crystal/mp-17048",)
    observables: ClassVar[dict[str, ObservableSpec]] = {
        "mode_frequency": ObservableSpec(
            units="THz", comparison="absolute", histogram=HistogramSpec(1e-10, 1e4)
        ),
        # TF32 force noise creates run-dependent imaginary interpolation modes in
        # soft crystals, so repeated TF32 thermal errors vary severalfold (mp-17048
        # heat capacity: 0.03 to 1.3 J/K/mol). Thermal properties are therefore
        # reported, and qualification uses the well-conditioned commensurate-point
        # spectrum.
        "entropy_300k": ObservableSpec(
            units="J/K/mol",
            comparison="absolute",
            histogram=HistogramSpec(1e-10, 1e5),
            report_only=True,
        ),
        "heat_capacity_300k": ObservableSpec(
            units="J/K/mol",
            comparison="absolute",
            histogram=HistogramSpec(1e-10, 1e5),
            report_only=True,
        ),
        "free_energy_300k": ObservableSpec(
            units="kJ/mol",
            comparison="absolute",
            histogram=HistogramSpec(1e-10, 1e5),
            report_only=True,
        ),
        "imaginary_mode_count": ObservableSpec(
            units="count",
            comparison="absolute",
            histogram=HistogramSpec(1e-6, 1e3),
            report_only=True,
        ),
    }

    def cases(self) -> list[CaseSpec]:
        frames = ase.io.read(
            Path(__file__).parent / "structures" / "mdr_phonon_10.extxyz.xz", index=":"
        )
        cases = [
            unrotated_case(
                case_id=f"phonon-crystal/{atoms.info['structure_id']}",
                metadata=_metadata(atoms),
                runtime_seed=42,
                sample_seed=42 + index,
                structure="mdr_phonon_10.extxyz.xz",
                structure_hash=atoms_hash(atoms),
                structure_index=index,
                task_name="omat",
            )
            for index, atoms in enumerate(frames)
        ]
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        metadata = case.spec.metadata
        displacement = float(metadata["displacement_angstrom"])
        symprec = float(metadata["symprec"])
        phonon = get_phonopy_object(
            case.transformed_atoms.copy(),
            displacement=displacement,
            supercell_matrix=metadata["supercell_matrix"],
            primitive_matrix=metadata["primitive_matrix"],
            symprec=symprec,
        )
        # Relaxation preserves the space group. Unconstrained relaxation lets
        # rounding noise lower the detected symmetry, which changes the
        # displacement set and makes the interpolated thermal properties jump
        # between identical runs.
        results = run_mdr_phonon_benchmark(
            phonon,
            calculator=calculator,
            displacement=displacement,
            run_relax=bool(metadata["run_relax"]),
            fix_symm_relax=bool(metadata["fix_symmetry_relax"]),
            symprec=symprec,
            symmetrize_fc=bool(metadata["symmetrize_force_constants"]),
        )
        frequencies = np.sort(
            np.asarray(results["frequencies"], dtype=np.float64) / THz_to_K, axis=1
        )
        temperatures = np.asarray(results["temperatures"], dtype=np.float64)
        (thermal_index,) = np.flatnonzero(
            temperatures == float(metadata["thermal_temperature_kelvin"])
        )
        imaginary_count = np.count_nonzero(
            frequencies < float(metadata["imaginary_frequency_threshold_thz"])
        )
        return {
            "mode_frequency": frequencies,
            **{
                f"{name}_300k": np.asarray(
                    [np.asarray(results[name], dtype=np.float64)[thermal_index]]
                )
                for name in ("entropy", "heat_capacity", "free_energy")
            },
            "imaginary_mode_count": np.asarray([imaginary_count]),
        }
