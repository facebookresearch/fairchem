"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

import pytest
import torch
from ase.build import bulk, make_supercell

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from tests.perf.performance_report import PerformanceReport

if TYPE_CHECKING:
    from ase import Atoms


# The scope here ensures that the same report instance is passed to every
# test that is run, letting us build up measurements and defer saving them
# until all tests have finished.
@pytest.fixture(scope="module")
def performance_report() -> Generator[PerformanceReport, None, None]:
    """
    Yields a performance report instance that can be used to aggregate results
    across many test cases. Results are saved when control returned to this
    function.

    Yields:
        PerformanceReport instance used to store performance test results.
    """

    report = PerformanceReport()
    yield report
    print("\n" + json.dumps(report.as_dict(), indent=4))


@dataclass
class InferenceTestCase:
    """
    Stores information used in a single inference test.

    Attributes:
        model: The name of the model to load.
        device: The device to use use for inference requests.
        structures: Each of the ASE atoms objects to use in inference requests.
    """

    model: str
    device: str
    structures: list[Atoms]


def generate_test_cases() -> list[InferenceTestCase]:
    """
    Generates a list of inference test cases to run.

    Returns:
        A list of test cases that should be run when measuring the
        performance of inference requests.
    """

    # Systems with different cell sizes to run inference on
    primitive = bulk("Fe")
    structures = [
        make_supercell(primitive, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        make_supercell(primitive, [[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
    ]

    # Always run tests on cpu. But also run on cuda if available.
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    # Return a test case for each combination of model and device type.
    #
    # Note: We could load the model here. However, all models would need to
    # be saved in memory at the same time, which can OOM on smaller machines.
    # Instead, defer creation of the model instances until they are needed
    # in each test case.
    return [
        InferenceTestCase(
            model=model,
            device=device,
            structures=structures,
        )
        for model in pretrained_mlip.available_models
        for device in devices
    ]


@pytest.mark.parametrize("test_case", generate_test_cases())
def test_pretrained_models(test_case, performance_report) -> None:
    """
    Evaluates the performance of all of the input inference test cases.
    """

    # Setup the predictor
    predictor = pretrained_mlip.get_predict_unit(
        model_name=test_case.model,
        device=test_case.device,
    )

    # Iterate over tasks in the predictor and different structures. We could
    # do this inside of generate_test_cases(). However, get_predict_unit()
    # can be slow so this lets us reuse the same predict unit for as many
    # inference requests as possible.
    #
    # In real tests at the time this was written, this saved around 20
    # minutes when using github runners (1 hour 10 minutes -> 49 minutes).
    for task in predictor.datasets:
        for atoms in test_case.structures:

            # Setup the prediction task
            atomic_data = AtomicData.from_ase(
                input_atoms=atoms,
                task_name=[task],
            )

            def predict(data) -> None:
                predictor.predict(data)
                torch.cuda.synchronize()

            # Run warmup steps without tracking performance
            for _ in range(2):
                predict(atomic_data)

            # Then run inference multiple times to build up useful statistics
            for _ in range(5):
                with performance_report.measure(
                    f"{test_case.model}_{task}_{len(atoms)}-atoms_{test_case.device}"
                ):
                    predict(atomic_data)
