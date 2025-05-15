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
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets import AseDBDataset


class AdsorptionSinglePointRunner(CalculateRunner):
    """
    Singlepoint evaluator for OC20 Adsorption systems

    OC20 originally reported adsorption energies. This runner provides the
    ability to compute adsorption energy S2EF numbers by referencing to the
    provided slab atoms object. Total energy S2EF evaluations are also
    possible.
    """

    result_glob_pattern: ClassVar[str] = "adsorption-singlepoint_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AseDBDataset,
        evaluate_total_energy: bool = False,
        adsorption_energy_model: bool = False,
    ):
        """
        Initialize the AdsorptionSinglePointRunner

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            evaluate_total_energy: Whether to evaluate total energies
            adsorption_energy_model: Whether the provided calculator is an adsorption energy model
        """
        self.evaluate_total_energy = evaluate_total_energy
        self.adsorption_energy_model = adsorption_energy_model
        if self.adsorption_energy_model:
            assert (
                not self.evaluate_total_energy
            ), "Total energy evals not available for adsorption energy models"
        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """
        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            list[dict[str, Any]] - List of dictionaries containing calculation results
        """
        all_results = []
        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(chunk_indices, desc="Running singlepoints"):
            atoms = self.input_data.get_atoms(i)
            identifier = atoms.info["identifier"]
            # extract targets
            dft_energy = atoms.get_potential_energy()
            dft_forces = atoms.get_forces()
            dft_slab_atoms = atoms.info["dft_slab_atoms"]
            dft_slab_energy = atoms.info["dft_slab_energy"]
            gas_reference_energy = atoms.info["gas_ref"]
            constraints = atoms.constraints
            if len(constraints) != 0:
                fixed_mask = atoms.constraints[0].index
            else:
                fixed_mask = []
            free_mask = np.array([i for i in range(len(atoms)) if i not in fixed_mask])
            dft_adsorption_energy = dft_energy - dft_slab_energy - gas_reference_energy

            # compute ml predictions
            atoms.calc = self.calculator
            pred_energy = atoms.get_potential_energy()
            pred_forces = atoms.get_forces()

            results = {
                "sid": identifier,
                "forces": pred_forces[free_mask],
                "forces_target": dft_forces[free_mask],
            }
            if self.adsorption_energy_model:
                results["energy_target"] = dft_adsorption_energy
                results["energy"] = pred_energy
            else:
                if self.evaluate_total_energy:
                    results["energy_target"] = dft_energy
                    results["energy"] = pred_energy
                else:
                    dft_slab_atoms.calc = self.calculator
                    results["energy_target"] = dft_adsorption_energy
                    results["energy"] = (
                        pred_energy
                        - dft_slab_atoms.get_potential_energy()
                        - gas_reference_energy
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
            os.path.join(
                results_dir, f"adsorption-singlepoint_{num_jobs}-{job_num}.json.gz"
            )
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        return
