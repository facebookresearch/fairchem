"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from ase.optimize import LBFGS
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.adsorption import (
    adsorb_atoms,
)

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.optimize import Optimizer

    from fairchem.core.datasets import AseDBDataset


class AdsorptionRunner(CalculateRunner):
    """
    Relax an adsorbate+surface configuration to compute the adsorption energy.
    The option to also relax a clean surface is also provided.

    This class handles the relaxation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.

    Input data is an AseDBDataset where each atoms object is organized as
    follows:
        atoms: adsorbate+surface configuration
        atoms.info = {
            gas_ref: float,
            dft_relaxed_adslab_energy: float,
            dft_relaxed_slab_energy: float,
            initial_slab_atoms: ase.Atoms, # Required if relax_surface=True
        }
    """

    result_glob_pattern: ClassVar[str] = "adsorption_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AseDBDataset,
        save_relaxed_atoms: bool = True,
        relax_surface: bool = False,
        optimizer_cls: type[Optimizer] = LBFGS,
        fmax: float = 0.05,
        steps: int = 300,
    ):
        """
        Initialize the AdsorptionRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            save_relaxed_atoms (bool): Whether to save the relaxed structures in the results
            relax_surface (bool): Whether to relax the bare surface
            optimizer_cls (Optimizer): ASE optimizer class to use
            fmax (float): force convergence threshold
            steps (int): max number of relaxation steps
        """
        self._save_relaxed_atoms = save_relaxed_atoms
        self.fmax = fmax
        self.steps = steps
        self.relax_surface = relax_surface
        self.optimizer_cls = optimizer_cls
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
            results = adsorb_atoms(
                atoms,
                self.calculator,
                optimizer_cls=self.optimizer_cls,
                steps=self.steps,
                fmax=self.fmax,
                relax_surface=self.relax_surface,
                save_relaxed_atoms=self._save_relaxed_atoms,
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
            results: List of dictionaries containing elastic properties
            results_dir: Directory path where results will be saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        results_df = pd.DataFrame(results)
        results_df.to_json(
            os.path.join(results_dir, f"adsorption_{num_jobs}-{job_num}.json.gz")
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        return
