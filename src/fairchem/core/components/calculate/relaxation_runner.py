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
from monty.dev import requires
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.relax import (
    relax_atoms,
)
from fairchem.core.components.calculate.recipes.utils import (
    get_property_dict_from_atoms,
)

try:
    from pymatgen.io.ase import MSONAtoms

    pmg_installed = True
except ImportError:
    pmg_installed = False


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets import AseDBDataset


@requires(pmg_installed, message="Requires `pymatgen` to be installed")
class RelaxationRunner(CalculateRunner):
    """Relax a sequence of several structures/molecules.

    This class handles the relaxation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    result_glob_pattern: ClassVar[str] = "relaxation_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AseDBDataset,
        calculate_properties: Sequence[str],
        save_relaxed_atoms: bool = True,
        normalize_properties_by: dict[str, str] | None = None,
        save_target_properties: Sequence[str] | None = None,
        **relax_kwargs,
    ):
        """Initialize the RelaxationRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            calculate_properties: Sequence of properties to calculate after relaxation
            save_relaxed_atoms (bool): Whether to save the relaxed structures in the results
            normalize_properties_by (dict[str, str] | None): Dictionary mapping property names to natoms or a key in
                atoms.info to normalize by
            save_target_properties (Sequence[str] | None): Sequence of target property names to save in the results file
                These properties need to be available using atoms.get_properties or present in the atoms.info dictionary
                This is useful if running a benchmark were errors will be computed after running relaxations
            relax_kwargs: Keyword arguments passed to relax. See signature of calculate.recipes.relax_atoms for options
        """
        self._calculate_properties = calculate_properties
        self._save_relaxed_atoms = save_relaxed_atoms
        self._normalize_properties_by = normalize_properties_by or {}
        self._save_target_properties = (
            save_target_properties if save_target_properties is not None else ()
        )
        self._relax_kwargs = relax_kwargs
        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """Perform relaxation calculations on a subset of structures.

        Splits the input data into chunks and processes the chunk corresponding to job_num.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            list[dict[str, Any]] - List of dictionaries containing calculation results
        """
        all_results = []
        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(chunk_indices, desc="Running relaxations"):
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
            if self._save_relaxed_atoms:
                results["atoms_initial"] = MSONAtoms(atoms).as_dict()

            try:
                atoms.calc = self.calculator
                atoms = relax_atoms(atoms, **self._relax_kwargs)
                results.update(
                    get_property_dict_from_atoms(
                        self._calculate_properties, atoms, self._normalize_properties_by
                    )
                )
                results.update(
                    {
                        "opt_nsteps": atoms.info.get("opt_nsteps", np.nan),
                        "opt_converged": atoms.info.get("opt_converged", np.nan),
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
                        "opt_nsteps": np.nan,
                        "opt_converged": np.nan,
                    }
                )

            if self._save_relaxed_atoms:
                results["atoms"] = MSONAtoms(atoms).as_dict()

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
            os.path.join(results_dir, f"relaxation_{num_jobs}-{job_num}.json.gz")
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
