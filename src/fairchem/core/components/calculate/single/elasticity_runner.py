"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING, Any, ClassVar

import ase.units
import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.elastic import (
    calculate_elasticity,
)

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets import AseDBDataset


eVA3_to_GPa = 1 / ase.units.GPa


class ElasticityRunner(CalculateRunner):
    """Calculate elastic tensor for a set of structures."""

    result_glob_pattern: ClassVar[str] = "elasticity_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AseDBDataset,
    ):
        """
        Initialize the ElasticityRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
        """
        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """
        Calculate elastic properties for a batch of structures.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            List of dictionaries containing elastic properties for each structure
        """
        all_results = []

        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(chunk_indices, desc="Running elasticity calculations."):
            atoms = self.input_data.get_atoms(i)
            try:
                results = calculate_elasticity(
                    atoms,
                    calculator=self.calculator,
                    cell_filter_cls=FrechetCellFilter,
                    fix_symmetry=False,
                )
                results["sid"] = atoms.info.get("sid", i)
                # change results to GPa
                results["elastic_tensor"] = (
                    results["elastic_tensor"].voigt * eVA3_to_GPa
                )
                results["shear_modulus_vrh"] = (
                    results["shear_modulus_vrh"] * eVA3_to_GPa
                )
                results["bulk_modulus_vrh"] = results["bulk_modulus_vrh"] * eVA3_to_GPa
                results.update({"errors": "", "traceback": ""})
            except Exception as ex:
                results = {
                    "sid": atoms.info["sid"],
                    "elastic_tensor": np.nan,
                    "shear_modulus_vrh": np.nan,
                    "bulk_modulus_vrh": np.nan,
                    "errors": f"{ex!r}",
                    "traceback": traceback.format_exc(),
                }

            all_results.append(results)

        return all_results

    def write_results(
        self,
        results: list[dict[str, Any]],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """Write calculation results to a compressed JSON file.

        Args:
            results: List of dictionaries containing elastic properties
            results_dir: Directory path where results will be saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        results_df = pd.DataFrame(results)
        results_df.to_json(
            os.path.join(results_dir, f"elasticity_{num_jobs}-{job_num}.json.gz")
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
