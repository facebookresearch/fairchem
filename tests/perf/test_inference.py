"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Generator

import pytest
import torch
from ase.build import bulk, make_supercell

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from tests.perf.performance_report import MeasurementStats, PerformanceReport

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

    # Number of samples to take
    num_warmup_samples: int = 2
    min_num_samples: int = 5
    max_num_samples: int = 50
    max_time_sec: float = 600

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
    for task in predictor.dataset_to_tasks.keys():
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
            for _ in range(num_warmup_samples):
                predict(atomic_data)

            # Convergence checks
            wall_time_convergence = MeanConvergenceChecker()
            cpu_time_convergence = MeanConvergenceChecker()
            cuda_time_convergence = MeanConvergenceChecker()

            # Then run inference multiple times to build up useful statistics.
            # This runs until convergence is reached or the max number of
            # allowed samples has been taken.
            start = perf_counter()
            for _ in range(max_num_samples):
                with performance_report.measure(
                    f"{test_case.model}_{task}_{len(atoms)}-atoms_{test_case.device}"
                ) as measurements:
                    predict(atomic_data)

                # Check if converged
                wall_time_convergence.add_measurement(measurements.wall_time_sec)
                cpu_time_convergence.add_measurement(measurements.cpu_time_sec)
                cuda_time_convergence.add_measurement(measurements.cuda_time_sec)
                if (
                    measurements.wall_time_sec.num_samples >= min_num_samples
                    and wall_time_convergence.is_converged()
                    and cpu_time_convergence.is_converged()
                    and cuda_time_convergence.is_converged()
                ):
                    break

                # Check if out of time. This protects against running for very
                # long times when large models are being slow to converge.
                if perf_counter() - start > max_time_sec:
                    break


class MeanConvergenceChecker:
    """
    Checks for convergence of the mean value over a number of samples.

    Each time add_measurement() is called, the current mean value over all
    samples is observed. Then the relative change in mean values compared
    to the last add_measurement() call is calculated. This watches for
    this relative change dropping below a configured threshold.

    For example, if the mean value over time is:

        [0.5, 0.8, 0.6, 0.65, 0.63, 0.64]

    Then the relative change at each step would be:

        [None, 0.6, 0.25, 0.083, 0.031, 0.016]

    If the convergence threshold was set to 0.1, and the required steps
    below that threshold was 2, then is_converged() would have returned
    True after 0.63 was observed (since 2 relative changes in a row were
    both below the threshold.)
    """

    def __init__(
        self,
        relative_mean_change_threshold: float = 0.002, # 0.2%
        required_samples_below_threshold: int = 3,
    ) -> None:
        """
        Args:
            relative_mean_change_threshold: After each measurement, the mean
                over all samples is calculated. The relative change in mean
                values is defined as abs((mean_cur - mean_prev) / mean_prev).
                To be considered converged, that change must be below this
                input threshold.
            required_samples_below_threshold: The number of samples in a row
                that must satisfy relative_mean_change_threshold before the
                calculation will be considered converged.
        """
        self._relative_mean_change_threshold = relative_mean_change_threshold
        self._required_samples_below_threshold = required_samples_below_threshold

        self._last_mean: float | None = None
        self._mean_change_history: list[float] = []

    def add_measurement(self, measurement: MeasurementStats) -> None:
        """
        Adds a new measurement to be used in convergence checks.

        Args:
            measurements: A single set of measurements to use when checking
                for convergence.
        """

        # Get the current mean value after the most recent sample was taken
        cur_mean = measurement.mean

        # If a mean value has already been observed, calculate the relative
        # change compared to this new mean
        if (last_mean := self._last_mean) is not None:
            if (difference := cur_mean - last_mean) == 0:
                self._mean_change_history.append(0)
            else:
                try:
                    self._mean_change_history.append(abs(difference / last_mean))
                except ZeroDivisionError:
                    self._mean_change_history.append(float("inf"))

        # Update the reference to the last measured mean value
        self._last_mean = cur_mean

    def is_converged(self) -> bool:
        """
        Returns whether enough measurements have been supplied to satisfy the
        convergence criteria.

        Returns:
            True if convergence has been reached.
        """

        # Grab the most recent relative mean changes
        changes = self._mean_change_history[-self._required_samples_below_threshold:]

        # Converged only if the most recent "required_samples_below_threshold"
        # changes are all below "relative_mean_change_threshold"
        return (
            len(changes) >= self._required_samples_below_threshold
            and max(changes) < self._relative_mean_change_threshold
        )
