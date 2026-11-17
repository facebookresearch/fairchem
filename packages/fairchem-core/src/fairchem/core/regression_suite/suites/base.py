"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
from ase.constraints import FixAtoms
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.plan import declarative_plan
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import (
    check_source_atoms,
    load_source_atoms,
)
from fairchem.core.regression_suite.spec.transforms import (
    canonicalize_rank_two,
    canonicalize_vectors,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from ase import Atoms

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.regression_suite.spec.schema import (
        CaseSpec,
        MeasurementRecord,
        PerformanceMetricSpec,
        SuiteCase,
        SuitePlan,
    )


class ObservableDescription(NamedTuple):
    """
    What an observable shows and what one of its clusters is, in plain words,
    for the caption of its plot.
    """

    shows: str
    point: str
    # Why the observable is shown but not graded, for report-only observables.
    report_only_reason: str = ""


class GroupValue(NamedTuple):
    """
    One graded value computed across several of a run's cases.
    """

    value: float
    case_count: int


class Suite(ABC):
    """
    Independent suite implementation with shared measurement semantics.

    A suite declares its observables and smoke cases as class attributes,
    builds the frozen cases that its audited ``plan_hash`` pins, and evaluates
    one case at a time, returning its observables in the source frame. Suites
    that generate their structures in code set ``generated_frames``; suites
    with report-only performance observables declare their metrics and
    benchmark protocol as class attributes.
    """

    name: ClassVar[str]
    plan_hash: ClassVar[str]
    observables: ClassVar[Mapping[str, ObservableSpec]]
    smoke_case_ids: ClassVar[tuple[str, ...]]
    structures_directory: ClassVar[str] = "structures"
    inference_settings_overrides: ClassVar[Mapping[str, Any]] = {}
    slurm_case_parallel: ClassVar[bool] = False
    generated_frames: ClassVar[Callable[[], Sequence[Atoms]] | None] = None
    performance_metrics: ClassVar[Mapping[str, PerformanceMetricSpec]] = {}
    performance_protocol: ClassVar[Mapping[str, Any] | None] = None
    # A suite can decide on other summary statistics than the campaign's and
    # scale them by its own safety factor; None keeps the campaign's.
    decision_statistics: ClassVar[tuple[str, ...] | None] = None
    safety_factor: ClassVar[float | None] = None
    # A suite can let a tier share its looser neighbour's limit wherever that
    # neighbour's calibration run stays within the tier's own limit, so a value
    # that cannot tell the two precisions apart only passes or fails there.
    merge_unresolved_tiers: ClassVar[bool] = False
    # What the suite checks and how it is graded, in plain language, as
    # (heading, paragraph) pairs that open the suite's section of a report.
    explanation: ClassVar[tuple[tuple[str, str], ...]] = ()
    # What each observable shows and what one of its clusters is, in plain
    # language, for the captions of the report's plots.
    observable_descriptions: ClassVar[Mapping[str, ObservableDescription]] = {}

    def grading_overrides(self) -> dict[str, Any]:
        """
        Return the grading settings this suite changes from the campaign's.
        """
        overrides = {
            "decision_statistics": None
            if self.decision_statistics is None
            else list(self.decision_statistics),
            "safety_factor": self.safety_factor,
            "merge_unresolved_tiers": self.merge_unresolved_tiers or None,
        }
        return {key: value for key, value in overrides.items() if value is not None}

    def group_values(
        self,
        reference: Sequence[MeasurementRecord],
        records: Sequence[MeasurementRecord],
    ) -> dict[str, dict[str, GroupValue]]:
        """
        Return graded values that depend on several of a run's cases at once.

        Each value is a non-negative distance from ideal behaviour, keyed by
        observable and then by decision cluster, and is graded like an
        absolute difference from zero. ``reference`` holds the FP64 records of
        the same cases. Most suites grade each case alone and have none.
        """
        return {}

    def plan(self) -> SuitePlan:
        """
        Return the suite's cases, checked against its audited plan hash.
        """
        return declarative_plan(self.name, self.cases(), self.plan_hash)

    def source_atoms(self, case: CaseSpec) -> Atoms:
        """
        Return the hash-checked source structure for one plan case.
        """
        if self.generated_frames is None:
            directory = Path(inspect.getfile(type(self))).parent
            return load_source_atoms(case, directory / self.structures_directory)
        return check_source_atoms(
            self.generated_frames()[case.structure_index].copy(), case
        )

    def predictor_group(self, case: CaseSpec) -> Hashable:
        """
        Return the key of cases that share a predictor under a structure scope.

        Composition-specific backends such as merged MOLE models load one
        predictor per structure and task by default.
        """
        return (case.structure, case.structure_index, case.task_name)

    @abstractmethod
    def cases(self) -> list[CaseSpec]:
        """
        Build the frozen cases of this suite's plan.
        """

    @abstractmethod
    def evaluate(
        self,
        calculator: FAIRChemCalculator,
        case: SuiteCase,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate one frozen case and return its values keyed by observable.
        """


# The observables that static_predictions returns, as every suite that calls
# it declares them.
STATIC_PREDICTION_OBSERVABLES = {
    "energy": ObservableSpec(
        units="eV",
        comparison="relative_with_floor",
        histogram=HistogramSpec(1e-16, 1e4),
        floor=1e-8,
    ),
    "forces": ObservableSpec(
        units="eV/A",
        comparison="relative_with_floor",
        histogram=HistogramSpec(1e-16, 1e4),
        floor=1e-6,
    ),
    "stress": ObservableSpec(
        units="eV/A^3",
        comparison="relative_with_floor",
        histogram=HistogramSpec(1e-16, 1e4),
        floor=1e-8,
    ),
}


def static_predictions(
    calculator: FAIRChemCalculator,
    case: SuiteCase,
    *,
    stress: bool = True,
) -> dict[str, np.ndarray]:
    """
    Predict total energy, the forces on atoms that are not held fixed, and
    optional Voigt stress in the source frame.
    """
    atoms = case.transformed_atoms.copy()
    atoms.calc = calculator
    free = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            free[constraint.get_indices()] = False
    forces = atoms.get_forces(apply_constraint=False)[free]
    predictions = {
        "energy": np.asarray(float(atoms.get_potential_energy())),
        "forces": canonicalize_vectors(forces, case),
    }
    if stress:
        transformed = voigt_6_to_full_3x3_stress(
            np.asarray(atoms.get_stress(), dtype=np.float64)
        )
        predictions["stress"] = full_3x3_to_voigt_6_stress(
            canonicalize_rank_two(transformed, case)
        )
    return predictions


def suite_contract_hash(suite: Suite) -> str:
    """
    Hash the observable contracts that every saved run of a suite depends on.

    A suite's own decision statistics and safety factor are part of its
    contract; suites that keep the defaults hash their observables alone.
    """
    contract = {
        name: dataclasses.asdict(spec) for name, spec in suite.observables.items()
    }
    if suite.grading_overrides():
        contract = {"observables": contract, "grading": suite.grading_overrides()}
    return stable_hash(contract)
