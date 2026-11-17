"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from fairchem.core.regression_suite.spec.plan import precision_rotation_sweep
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.suites.base import (
    STATIC_PREDICTION_OBSERVABLES,
    ObservableDescription,
    Suite,
    static_predictions,
)

if TYPE_CHECKING:
    import numpy as np

    from fairchem.core.regression_suite.spec.schema import (
        CaseSpec,
        ObservableSpec,
        SuiteCase,
    )

TASKS = ("omat", "omol", "oc20", "oc22", "oc25", "odac", "omc")


class PredictionFidelitySuite(Suite):
    """
    Record energy, force, and, for bulk materials and molecular crystals,
    stress predictions in the source frame.
    """

    name = "prediction-fidelity"
    explanation = (
        (
            "What it checks",
            "Whether the candidate predicts the same energies, forces, and "
            "stresses as FP64 on ordinary structures. Every other suite builds "
            "on these numbers. It covers all seven UMA tasks, so a problem that "
            "only shows up for molecules, catalyst surfaces, porous frameworks, "
            "or crystals is caught here.",
        ),
        (
            "The cases",
            "Ten structures come from each task, omat, omol, oc20, oc22, oc25, "
            "odac, and omc, for 70 in total. They have 4 to 280 atoms. Each "
            "structure runs in its original orientation and in eight random "
            "rotations, which gives 630 cases. The FP64 reference runs the same "
            "630 cases, so every candidate result has an exact partner.",
        ),
        (
            "What is measured",
            "Each case records the total energy in eV and the force on every "
            "atom in eV/Å. Atoms held fixed in place, as in many catalysis "
            "slabs, are left out. Stress in eV/Å³ is recorded only for the omat "
            "and omc crystals. Forces and stress are rotated back to the "
            "structure's original axes, so every orientation is compared in the "
            "same frame.",
        ),
        (
            "How differences are measured",
            "Every number is compared with its FP64 partner. The difference is "
            "divided by the size of the FP64 value, which gives a normalized "
            "difference. When the FP64 value is tiny, the suite divides by a "
            "fixed floor instead. The floors are 1e-8 eV for energy, 1e-6 eV/Å "
            "for forces, and 1e-8 eV/Å³ for stress. The plain absolute "
            "difference is kept as well.",
        ),
        (
            "How it is graded",
            "A cluster is one structure in one orientation. Stress has 180 "
            "clusters, from the 20 crystals, and energy and forces have all "
            "630. Each cluster's differences are averaged into one number. The "
            "median and 95th percentile of those numbers must stay within twice "
            "what the baseline's run at that precision scored. A quantity "
            "passes if either its normalized or its absolute differences do.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "energy": ObservableDescription(
            "Total energy of the structure, in eV",
            "one structure in one orientation",
        ),
        "forces": ObservableDescription(
            "Force on each atom along x, y, and z, in eV/Å, leaving out fixed atoms",
            "one structure in one orientation",
        ),
        "stress": ObservableDescription(
            "The six stress components, in eV/Å³",
            "one crystal in one orientation (omat and omc only)",
        ),
    }
    plan_hash = "78e035497bad1be6ab45c3bc97ddbae8f96e68bf57943465791bb138dff373d6"
    smoke_case_ids = (
        "accuracy/omat/45153",
        "accuracy/omol/788296",
        "accuracy/oc20/615176",
        "accuracy/oc22/74919",
        "accuracy/oc25/64511",
        "accuracy/odac/5158",
        "accuracy/omc/13003",
    )
    observables: ClassVar[dict[str, ObservableSpec]] = {
        **STATIC_PREDICTION_OBSERVABLES,
    }

    def cases(self) -> list[CaseSpec]:
        structures = Path(__file__).parent / "structures"
        frames = [
            (task, frame, structure_id, structure_hash)
            for task in TASKS
            for frame, (structure_id, structure_hash) in enumerate(
                frame_identities(structures / f"{task}.extxyz.xz")
            )
        ]
        cases = []
        for index, (task, frame, structure_id, structure_hash) in enumerate(frames):
            cases.extend(
                precision_rotation_sweep(
                    case_id=f"accuracy/{task}/{structure_id}",
                    metadata={"structure_id": structure_id},
                    runtime_seed=42 + index,
                    sample_seed=42 + 10_000 * index,
                    structure=f"{task}.extxyz.xz",
                    structure_hash=structure_hash,
                    structure_index=frame,
                    task_name=task,
                )
            )
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        return static_predictions(
            calculator, case, stress=case.spec.task_name in {"omat", "omc"}
        )
