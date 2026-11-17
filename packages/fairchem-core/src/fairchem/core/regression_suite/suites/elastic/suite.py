"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import ase.units
import numpy as np
from ase.filters import FrechetCellFilter

from fairchem.core.components.calculate.recipes.elastic import calculate_elasticity
from fairchem.core.regression_suite.spec.plan import unrotated_case
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.suites.base import (
    ObservableDescription,
    Suite,
)

if TYPE_CHECKING:
    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase

_UPPER_TRIANGLE = np.triu_indices(6)
_EV_A3_TO_GPA = 1.0 / ase.units.GPa


class ElasticSuite(Suite):
    """
    Compare complete elastic tensors and Voigt-Reuss-Hill moduli with FP64.
    """

    name = "elastic"
    explanation = (
        (
            "What it checks",
            "Whether the candidate gives the same crystal stiffness as FP64. "
            "Stiffness comes out of a long chain of relaxations and stress "
            "readings. A small force difference can change where a relaxation "
            "stops, so this suite shows how rounding adds up in a real "
            "materials workflow.",
        ),
        (
            "The cases",
            "Ten crystals from the Materials Project elasticity set run with "
            "the omat task. They have 4 to 10 atoms and are not rotated, so "
            "there are ten cases. Each first relaxes its atoms and cell for at "
            "most 500 steps, until the largest force, including the force on "
            "the cell, falls below 0.02 eV/Å.",
        ),
        (
            "What is measured",
            "The relaxed crystal gets 24 small strains. Each axis is stretched "
            "and squeezed by 0.5% and 1%, and each angle is sheared by 3% and "
            "6% both ways. At each strain the atoms relax with the cell held "
            "fixed, and the stress is read. A straight-line fit of stress "
            "against strain gives the elastic tensor.",
        ),
        (
            "The recorded values",
            "The elastic tensor says how hard the crystal resists each kind of "
            "deformation. The suite records its 21 independent entries in GPa. "
            "It also records the bulk modulus, the resistance to squeezing, and "
            "the shear modulus, the resistance to shearing. Both are "
            "Voigt-Reuss-Hill averages, the usual way to turn the tensor into "
            "one number, also in GPa.",
        ),
        (
            "How it is graded",
            "Each value is compared with FP64 as a plain absolute difference in "
            "GPa. A cluster is one crystal, so there are ten. The 21 tensor "
            "differences are averaged into one number per crystal. The tensor "
            "and both moduli are graded on the median and 95th percentile of "
            "the ten numbers, within twice the baseline's values. With ten "
            "crystals, the 95th percentile sits close to the worst one.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "elastic_tensor": ObservableDescription(
            "The 21 independent entries of the elastic tensor, fitted from 24 strained and relaxed copies, in GPa",
            "one crystal",
        ),
        "bulk_modulus_vrh": ObservableDescription(
            "Bulk modulus, the resistance to squeezing (Voigt-Reuss-Hill average), in GPa",
            "one crystal",
        ),
        "shear_modulus_vrh": ObservableDescription(
            "Shear modulus, the resistance to shearing (Voigt-Reuss-Hill average), in GPa",
            "one crystal",
        ),
    }
    plan_hash = "d2d253335121a80986633ced63f9e1c9bc0a9a0fb8470974cbb5a1eb8653f687"
    smoke_case_ids = ("elastic/mp-20560",)
    observables: ClassVar[dict[str, ObservableSpec]] = {
        "elastic_tensor": ObservableSpec(
            units="GPa", comparison="absolute", histogram=HistogramSpec(1e-10, 1e4)
        ),
        "bulk_modulus_vrh": ObservableSpec(
            units="GPa", comparison="absolute", histogram=HistogramSpec(1e-10, 1e4)
        ),
        "shear_modulus_vrh": ObservableSpec(
            units="GPa", comparison="absolute", histogram=HistogramSpec(1e-10, 1e4)
        ),
    }

    def cases(self) -> list[CaseSpec]:
        structures = frame_identities(
            Path(__file__).parent / "structures" / "mp_pbe_elasticity_10.extxyz.xz"
        )
        cases = [
            unrotated_case(
                case_id=f"elastic/{structure_id}",
                metadata={
                    "cell_filter": "FrechetCellFilter",
                    "fix_symmetry": False,
                    "fmax": 0.02,
                    "norm_strains": [-0.01, -0.005, 0.005, 0.01],
                    "relax_initial": True,
                    "relax_strained": True,
                    "relaxation_steps": 500,
                    "shear_strains": [-0.06, -0.03, 0.03, 0.06],
                    "structure_id": structure_id,
                    "use_equilibrium_stress": True,
                },
                runtime_seed=42,
                structure="mp_pbe_elasticity_10.extxyz.xz",
                structure_hash=structure_hash,
                structure_index=index,
                task_name="omat",
            )
            for index, (structure_id, structure_hash) in enumerate(structures)
        ]
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        metadata = case.spec.metadata
        results = calculate_elasticity(
            case.transformed_atoms.copy(),
            calculator=calculator,
            norm_strains=tuple(float(value) for value in metadata["norm_strains"]),
            shear_strains=tuple(float(value) for value in metadata["shear_strains"]),
            relax_initial=bool(metadata["relax_initial"]),
            relax_strained=bool(metadata["relax_strained"]),
            use_equilibrium_stress=bool(metadata["use_equilibrium_stress"]),
            steps=int(metadata["relaxation_steps"]),
            fmax=float(metadata["fmax"]),
            cell_filter_cls=FrechetCellFilter,
            fix_symmetry=bool(metadata["fix_symmetry"]),
        )
        elastic_tensor = np.asarray(results["elastic_tensor"].voigt, dtype=np.float64)
        return {
            "elastic_tensor": elastic_tensor[_UPPER_TRIANGLE] * _EV_A3_TO_GPA,
            **{
                name: np.asarray(results[name] * _EV_A3_TO_GPA)
                for name in ("bulk_modulus_vrh", "shear_modulus_vrh")
            },
        }
