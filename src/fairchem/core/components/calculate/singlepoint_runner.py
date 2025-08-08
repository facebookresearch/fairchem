"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.utils import (
    get_property_dict_from_atoms,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets import AseDBDataset


class SinglePointRunner(CalculateRunner):
    """Perform a single point calculation of several structures/molecules.

    This class handles the single point calculation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    result_glob_pattern: ClassVar[str] = "singlepoint_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AseDBDataset,
        calculate_properties: Sequence[str],
        normalize_properties_by: dict[str, str] | None = None,
        save_target_properties: Sequence[str] | None = None,
    ):
        """Initialize the SinglePointRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            calculate_properties: Sequence of properties to calculate
            normalize_properties_by (dict[str, str] | None): Dictionary mapping property names to natoms or a key in
                atoms.info to normalize by
            save_target_properties (Sequence[str] | None): Sequence of target property names to save in the results file
                These properties need to be available using atoms.get_properties or present in the atoms.info dictionary
        """
        self._calculate_properties = calculate_properties
        self._normalize_properties_by = normalize_properties_by or {}
        self._save_target_properties = (
            save_target_properties if save_target_properties is not None else []
        )

        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """Perform singlepoint calculations on a subset of structures.

        Splits the input data into chunks and processes the chunk corresponding to job_num.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            list[dict[str, Any]] - List of dictionaries containing calculation results
        """
        all_results = []
        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(chunk_indices, desc="Running singlepoint calculations"):
            atoms = self.input_data.get_atoms(i)
            results = {
                "sid": atoms.info.get("sid", i),
                "natoms": len(atoms),
            }
            # add target properties if requested
            target_properties = get_property_dict_from_atoms(
                self._save_target_properties, atoms, self._normalize_properties_by
            )
            results.update(
                {f"{key}_target": target_properties[key] for key in target_properties}
            )

            try:
                atoms.calc = self.calculator
                results.update(
                    get_property_dict_from_atoms(
                        self._calculate_properties, atoms, self._normalize_properties_by
                    )
                )
                results.update(
                    {
                        "errors": "",
                        "traceback": "",
                    }
                )
            except Exception as ex:  # TODO too broad-figure out which to catch
                results.update(dict.fromkeys(self._calculate_properties, np.nan))
                results.update(
                    {
                        "errors": f"{ex!r}",
                        "traceback": traceback.format_exc(),
                    }
                )

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
            results: List of dictionaries containing energy and forces results
            results_dir: Directory path where results will be saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        results_df = pd.DataFrame(results)
        results_df.to_json(
            os.path.join(results_dir, f"singlepoint_{num_jobs}-{job_num}.json.gz")
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
