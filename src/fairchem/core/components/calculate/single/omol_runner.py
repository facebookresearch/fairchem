"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fairchem.core.components.calculate import CalculateRunner

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class OMolRunner(CalculateRunner):
    """
    Runner for OMol's evaluation tasks.
    """

    def __init__(
        self,
        calculator: Calculator,
        input_data: dict,
        benchmark_name: str,
        benchmark: Callable,
    ):
        """
        Initialize the OMolRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Input dictionary data
            benchmark_name: Name of the benchmark task
            benchmark: Benchmark function to evaluate the input data
        """
        with open(input_data, "rb") as f:
            input_data = pickle.load(f)
        self.result_glob_pattern = f"{benchmark_name}_*-*.json.gz"
        self.benchmark_name = benchmark_name
        self.benchmark = benchmark
        super().__init__(calculator=calculator, input_data=input_data)

        self.input_keys = list(input_data.keys())

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """
        Perform calculations on a subset of structures.

        Splits the input data into chunks and processes the chunk corresponding to job_num.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            list[dict[str, Any]] - List of dictionaries containing calculation results
        """
        chunk_indices = np.array_split(self.input_keys, num_jobs)[job_num]
        chunk_data = {x: self.input_data[x] for x in chunk_indices}
        all_results = self.benchmark(chunk_data, self.calculator)
        return all_results

    def write_results(
        self,
        results: list[dict[str, Any]],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """
        Write calculation results to a compressed JSON file.

        Args:
            results: List of dictionaries containing elastic properties
            results_dir: Directory path where results will be saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        with gzip.open(
            os.path.join(
                results_dir, f"{self.benchmark_name}_{num_jobs}-{job_num}.json.gz"
            ),
            "wb",
        ) as f:
            f.write(json.dumps(results).encode("utf-8"))

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
