"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import numpy as np
import pandas as pd
from ase.calculators.calculator import PropertyNotPresent
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner

if TYPE_CHECKING:
    from ase.atoms import Atoms
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
        save_target_properties: Sequence[str] | None = None,
        **singlepoint_kwargs,
    ):
        """Initialize the SinglePointRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            save_target_properties (Sequence[str] | None): Sequence of target property names to save in the results file
                These properties need to be available using atoms.get_properties or present in the atoms.info dictionary
            singlepoint_kwargs: Keyword arguments passed to singlepoint. (MAY BE THIS IS NOT REQUIRED, HAVE ADDED THIS USING RELAXATION RUNNER)
        """
        self._save_target_properties = (
            save_target_properties if save_target_properties is not None else []
        )
        self._singlepoint_kwargs = singlepoint_kwargs

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
            results = {}
            atoms = self.input_data.get_atoms(i)
            try:
                for property_name in self._save_target_properties:
                    results[f"{property_name}_target"] = self._get_property_from_atoms(
                        atoms, property_name
                    )

                atoms.calc = self.calculator
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                results.update(
                    {
                        "sid": atoms.info.get("sid", i),
                        "energy": energy,
                        "forces": forces,
                        "errors": "",
                        "traceback": "",
                    }
                )
            except Exception as ex:  # TODO too broad-figure out which to catch
                results.update(
                    {
                        "sid": i if atoms is None else atoms.info.get("sid", i),
                        "energy": np.nan,
                        "forces": np.nan,
                        "errors": f"{ex!r}",
                        "traceback": traceback.format_exc(),
                    }
                )

            all_results.append(results)

        return all_results

    @staticmethod  # TODO make this a utils function?
    def _get_property_from_atoms(atoms: Atoms, property_name: str) -> int | float:
        """Retrieve a property from an Atoms object, either from its properties or info dictionary.

        Args:
            atoms: The ASE Atoms object to extract properties from
            property_name: Name of the property to retrieve

        Returns:
            The property value as an integer or float

        Raises:
            ValueError: If the property is not found in either the properties or info dictionary
        """
        try:
            # get_properties returns a Properties dict-like object, so we index again for the property requested
            prop = atoms.get_properties([property_name])[property_name]
        except PropertyNotPresent:
            try:
                prop = atoms.info[property_name]
            except KeyError as err:
                raise ValueError(
                    f"The listed property {property_name} in `save_target_properties` is not available from"
                    f" the atoms object or its info dictionary"
                ) from err
        return prop

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

    def load_state(self, checkpoint_location: str | None) -> None:
        return
