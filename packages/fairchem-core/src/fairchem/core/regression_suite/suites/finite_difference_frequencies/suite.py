"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.suites.base import (
    ObservableDescription,
    Suite,
)
from fairchem.core.regression_suite.suites.vibrational_frequencies.common import (
    finite_difference_frequencies,
    vibration_cases,
)

if TYPE_CHECKING:
    import numpy as np

    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase


class FiniteDifferenceFrequenciesSuite(Suite):
    """
    Compare vibrational spectra computed from finite-difference Hessians.
    """

    name = "finite-difference-frequencies"
    explanation = (
        (
            "What it checks",
            "Whether the candidate gives the same molecular vibration "
            "frequencies as FP64 when they are computed from forces alone. This "
            "is the standard route to vibrational spectra and zero-point "
            "energies. Every profile can run it, because it only needs forces.",
        ),
        (
            "The cases",
            "The five NCIAtlas dimers from physical-constraints run with the "
            "omol task. Each is a pair of small molecules with 5 or 6 atoms in "
            "total. Every dimer runs in its original orientation and in eight "
            "random rotations, which gives 45 cases.",
        ),
        (
            "What is measured",
            "ASE moves each atom by 0.01 Å both ways along x, y, and z and "
            "records the forces. The changes in force give the Hessian, the "
            "matrix of second derivatives of the energy. With the atomic "
            "masses, it gives 15 or 18 frequencies in cm⁻¹, one per degree of "
            "freedom. Imaginary frequencies are stored as negative numbers. "
            "All are sorted from lowest to highest.",
        ),
        (
            "How it is graded",
            "Each frequency is compared with the FP64 frequency of the same "
            "rank, as an absolute difference in cm⁻¹. The six near-zero modes, "
            "from moving or turning the whole molecule, are included. A cluster "
            "is one molecule in one orientation, and its differences are "
            "averaged. The median and 95th percentile of the 45 numbers must "
            "stay within twice the baseline's values.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "finite_difference_mode_frequency": ObservableDescription(
            "Every vibration frequency from forces at 0.01 Å displacements, sorted, in cm⁻¹, where negative means imaginary",
            "one molecule in one orientation",
        ),
    }
    plan_hash = "30982b22127ff47e11c6878ab995cc5951591d798fc611feaab00ce19cc4af9a"
    smoke_case_ids = ("vibrations/5.0122/orientation-00",)
    structures_directory = "../physical_constraints/structures"
    observables: ClassVar[dict[str, ObservableSpec]] = {
        "finite_difference_mode_frequency": ObservableSpec(
            units="cm^-1", comparison="absolute", histogram=HistogramSpec(1e-8, 1e5)
        ),
    }

    def cases(self) -> list[CaseSpec]:
        return vibration_cases()

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        atoms = case.transformed_atoms.copy()
        atoms.calc = calculator
        frequencies = finite_difference_frequencies(
            atoms, float(case.spec.metadata["finite_difference_step"])
        )
        return {"finite_difference_mode_frequency": frequencies}
