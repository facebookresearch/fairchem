from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Generator

import pytest

from fairchem.core._cli import main
from tests.perf.performance_report import PerformanceReport


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
class TrainUMATestCase:
    """
    Stores information for a single train_uma test.
    Attributes:
        config_path: Path to the config file for training.
        measurement_name: Unique name for the performance measurement.
    """

    config_path: str
    measurement_name: str


def generate_train_uma_test_cases() -> list[TrainUMATestCase]:
    """
    Generates a list of train_uma test cases to run.

    Returns:
        A list of test cases that should be run when measuring the
        performance of training requests.
    """
    test_cases: list[TrainUMATestCase] = []
    config_dict = {
        "uma_sm_task_oc20_direct": "configs/uma_oc20_sm_direct.yaml",
        # "uma_sm_task_oc20_conserve": "configs/uma_oc20_sm_conserve.yaml",
        # "uma_md_task_oc20_direct": "configs/uma_oc20_md_direct.yaml",
    }
    for measurement_name, config_path in config_dict.items():
        test_cases.append(
            TrainUMATestCase(config_path=config_path, measurement_name=measurement_name)
        )
    return test_cases


def train_uma(path_to_config):
    """
    Submit a call for uma_s training performance test.
    """
    # command = ["fairchem", "-c", path_to_config]
    # distutils.cleanup()
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", path_to_config]
    sys.argv[1:] = sys_args
    main()


@pytest.mark.parametrize("test_case", generate_train_uma_test_cases())
def test_model_training(test_case, performance_report) -> None:
    """
    Evaluate performance of all of the input training test cases.
    Args:
        test_case: The TrainUMATestCase instance containing the config path
            and measurement name.
        performance_report: The PerformanceReport instance to log results.
    """
    for _ in range(1):
        with performance_report.measure(test_case.measurement_name):
            train_uma(test_case.config_path)
