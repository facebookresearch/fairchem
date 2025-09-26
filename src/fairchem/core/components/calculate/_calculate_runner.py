"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from glob import glob
from typing import TYPE_CHECKING, ClassVar, TypeVar

from fairchem.core.components.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators import Calculator


R = TypeVar("R")


class CalculateRunner(Runner, metaclass=ABCMeta):
    """Runner to run calculations/predictions using an ASE-like calculator and save results to file.

    Note:
        When running with the `fairchemv2` cli, the `job_config` and attribute is set at
        runtime to those given in the config file.

    See the Runner interface class for implementation details.

    Attributes:
        job_config (DictConfig): a managed attribute that gives access to the job config
        result_glob_pattern (str): glob pattern of results written to file
    """

    # pattern used in writing result files
    result_glob_pattern: ClassVar[str] = "*"

    def __init__(
        self,
        calculator: Calculator,
        input_data: Sequence,
    ):
        """Initialize the CalculateRunner with a calculator and input data.

        Args:
            calculator (Calculator): ASE-like calculator to perform calculations
            input_data (Sequence): Input data to be processed by the calculator
        """
        self._calculator = calculator
        self._input_data = input_data
        self._already_calculated = False

    @property
    def calculator(self) -> Calculator:
        """Get the calculator instance.

        Returns:
            Calculator: The ASE-like calculator used for calculations
        """
        return self._calculator

    @property
    def input_data(self) -> Sequence:
        """Get the input data.

        Returns:
            Sequence: The input data to be processed
        """
        return self._input_data

    @abstractmethod
    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> R:
        """Run any calculation using an ASE like Calculator.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            R: Results of the calculation
        """
        raise NotImplementedError

    @abstractmethod
    def write_results(
        self, results: R, results_dir: str, job_num: int = 0, num_jobs: int = 1
    ) -> None:
        """Write results to file in results_dir.

        Args:
            results (R): Results from the calculation
            results_dir (str): Directory to write results to
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.
        """
        raise NotImplementedError

    @abstractmethod
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        """Save the current state of the calculation to a checkpoint.

        Args:
            checkpoint_location (str): Location to save the checkpoint
            is_preemption (bool, optional): Whether this save is due to preemption. Defaults to False.

        Returns:
            bool: True if state was successfully saved, False otherwise
        """
        raise NotImplementedError

    def load_state(self, checkpoint_location: str | None) -> None:
        """Load a previously saved state from a checkpoint.

        Args:
            checkpoint_location (str | None): Location of the checkpoint to load, or None if no checkpoint
        """
        if checkpoint_location is not None:
            os.makedirs(
                self.job_config.metadata.results_dir, exist_ok=True
            )  # TODO Should we have all these dir created in cli main?

            existing_results_files = glob(
                os.path.join(checkpoint_location, self.result_glob_pattern)
            )
            if len(existing_results_files) == 0:
                return

            job_array_num = self.job_config.metadata.array_job_num
            job_array_total = self.job_config.scheduler.num_array_jobs
            # try to find and load the results file for the given job
            for file in existing_results_files:
                matches = re.findall(r"\d+", file)
                if (
                    matches
                    and int(matches[-1]) == job_array_num
                    and int(matches[-2]) == job_array_total
                ):
                    # copy it over to the new run directory
                    shutil.copy(file, self.job_config.metadata.results_dir)

                    job_state_path = os.path.join(
                        os.path.dirname(file),
                        f"{self.result_glob_pattern.split('_*')[0]}_{job_array_total}_{job_array_num}.txt",
                    )
                    new_path = os.path.join(
                        self.job_config.metadata.results_dir,
                        os.path.basename(file),
                    )
                    with open(job_state_path, "w") as f:
                        f.write(f"{file} moved to {new_path}")

                    os.remove(file)
                    # make sure we can load the file?
                    self._already_calculated = True
                    return

    def run(self):
        """Run the actual calculation and save results.

        Creates the results directory if it doesn't exist, runs the calculation,
        and writes the results to the specified directory.

        Note:
            Re-implementing this method in derived classes is discouraged.
        """
        if not self._already_calculated:
            os.makedirs(
                self.job_config.metadata.results_dir, exist_ok=True
            )  # TODO Should we have all these dir created in cli main?

            results = self.calculate(
                job_num=self.job_config.metadata.array_job_num,
                num_jobs=self.job_config.scheduler.num_array_jobs,
            )
            self.write_results(
                results,
                self.job_config.metadata.results_dir,
                job_num=self.job_config.metadata.array_job_num,
                num_jobs=self.job_config.scheduler.num_array_jobs,
            )
