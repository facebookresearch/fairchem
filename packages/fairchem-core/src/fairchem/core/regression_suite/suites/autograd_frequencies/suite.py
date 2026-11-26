"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.suites.base import (
    ObservableDescription,
    Suite,
)
from fairchem.core.regression_suite.suites.vibrational_frequencies.common import (
    autograd_frequencies,
    vibration_cases,
)

if TYPE_CHECKING:
    import numpy as np

    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase


class AutogradFrequenciesSuite(Suite):
    """
    Compare vibrational spectra computed from autograd Hessians.
    """

    name = "autograd-frequencies"
    explanation = (
        (
            "What it checks",
            "The same molecular frequencies as finite-difference-frequencies, "
            "but from a Hessian the model computes by automatic "
            "differentiation. Instead of nudging atoms, autograd differentiates "
            "the forces again, once per force component. This tests the code "
            "that computes derivatives at second order.",
        ),
        (
            "The cases",
            "It runs the same 45 cases as finite-difference-frequencies. They "
            "are the five NCIAtlas dimers of 5 or 6 atoms, with the omol task, "
            "each in its original orientation and in eight random rotations. "
            "The suite turns on Hessian output for omol, which the checkpoint "
            "does not provide by default.",
        ),
        (
            "What is measured",
            "The Hessian, with the atomic masses, gives 15 or 18 frequencies "
            "per molecule in cm⁻¹. Imaginary frequencies are stored as negative "
            "numbers. All are sorted from lowest to highest. The six near-zero "
            "modes, from moving or turning the whole molecule, are kept.",
        ),
        (
            "How it is graded",
            "Each frequency is compared with the FP64 autograd frequency of the "
            "same rank, as an absolute difference in cm⁻¹. A cluster is one "
            "molecule in one orientation. The median and 95th percentile of the "
            "45 cluster averages must stay within twice the baseline's values. "
            "The two frequency suites are never compared with each other.",
        ),
        (
            "Unsupported profiles",
            "Some settings cannot take this second derivative, such as custom "
            "fast backward passes and some compiled models. A profile that "
            "cannot run it lists the suite as unsupported in its settings, and "
            "the report marks it UNSUPPORTED instead of grading it. That is "
            "informational and never fails the candidate.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "autograd_mode_frequency": ObservableDescription(
            "Every vibration frequency from the model's autograd Hessian, sorted, in cm⁻¹, where negative means imaginary",
            "one molecule in one orientation",
        ),
    }
    plan_hash = "f5712d33f03e94bb9688e86522f36120028c14bf949b9229dfe2f4c2020b6846"
    smoke_case_ids = ("vibrations/5.0122/orientation-00",)
    structures_directory = "../physical_constraints/structures"
    inference_settings_overrides: ClassVar[dict[str, Any]] = {
        "predict_untrained_hessian": ["omol"],
        "hessian_vmap": False,
    }
    observables: ClassVar[dict[str, ObservableSpec]] = {
        "autograd_mode_frequency": ObservableSpec(
            units="cm^-1", comparison="absolute", histogram=HistogramSpec(1e-8, 1e5)
        ),
    }

    def cases(self) -> list[CaseSpec]:
        return vibration_cases()

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        atoms = case.transformed_atoms.copy()
        atoms.calc = calculator
        return {"autograd_mode_frequency": autograd_frequencies(atoms)}
