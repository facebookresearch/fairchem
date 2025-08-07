from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

import pytest

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from tests.perf.performance_report import PerformanceReport

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


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
        measurement_name: A unique name identifying each measurement.
        predict_unit: The predict unit that should be used when making the
            inference request.
        task: The name of the task that should be run.
        performance_report: The performance report instance that will track
            all measurements.
    """

    measurement_name: str
    predict_unit: MLIPPredictUnit
    task: str


def generate_test_cases() -> list[InferenceTestCase]:
    """
    Generates a list of inference test cases to run.

    Returns:
        A list of test cases that should be run when measuring the
        performance of inference requests.
    """

    # Iterate over all of the pretrained models that ship with fairchem
    test_cases: list[InferenceTestCase] = []
    for model in pretrained_mlip.available_models:

        # Run tests with and without gpu enabled
        for device in ("cuda", "cpu"):
            predict_unit = pretrained_mlip.get_predict_unit(
                model_name=model,
                device=device,
            )

            # Test each of the different tasks supported by the current model
            test_cases.extend(
                InferenceTestCase(
                    measurement_name=(
                        f"{model}_{task}_{device}" if task
                        else f"{model}_{device}"
                    ),
                    task=task,
                    predict_unit=predict_unit,
                )
                for task in predict_unit.datasets or [None]
            )

    return test_cases


@pytest.mark.parametrize("test_case", generate_test_cases())
def test_pretrained_models(test_case, performance_report) -> None:
    """
    Evaluates the performance of all of the input inference test cases.
    """

    # For each test case, run the inference request multiple times to build
    # up useful statistics
    for _ in range(5):

        calculator = FAIRChemCalculator(
            predict_unit=test_case.predict_unit,
            task_name=test_case.task,
        )

        # TODO Iterate over several structures, each a different test case
        from ase.build import bulk

        atoms = bulk("Fe")
        atoms.calc = calculator

        with performance_report.measure(test_case.measurement_name):
            atoms.get_potential_energy()
