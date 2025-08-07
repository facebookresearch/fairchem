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


def solvation_data(dataset_root, dataset_name):
    """
    Load solvation data from a pickle file.
    """
    with open(f"{dataset_root}/{dataset_name}.pkl", "rb") as f:
        data = pd.read_pickle(f)
    return data


class SolvationRunner(CalculateRunner):
    """
    Singlepoint evaluator for OC25 solvation energy. #to-do: write an explanation
    """

    result_glob_pattern: ClassVar[str] = "solvation_*-*.json.gz"

    def __init__(self, calculator: Calculator, input_data: dict):
        """
        Initialize the SolvationRunner

        Args:
            calculator: ASE calculator to use for energy calculations
            input_data (dict): Dictionary containing atomic structures to process
        """
        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """
        Perform solvation energy calculations on the input data.

        Args:
            job_num: Job number for parallel processing
            num_jobs: Total number of jobs for parallel processing

        Returns:
            list[dict[str, Any]] containing results for each key in the input data
        """
        all_results = []
        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(
            chunk_indices, desc="Calculating ground truth and running single points"
        ):
            key = list(self.input_data.keys())[i]
            # define all the atoms for calculator set up
            randsh0 = self.input_data[key]["randsh0"]
            randsh0_surface = self.input_data[key]["randsh0_surface"]
            randsh0_adsorbate_surface = self.input_data[key][
                "randsh0_adsorbate_surface"
            ]
            randsh0_solvent_surface = self.input_data[key]["randsh0_solvent_surface"]

            # compute dft ground truth
            randsh0_dft_en = randsh0.get_potential_energy()
            randsh0_surface_dft_en = randsh0_surface.get_potential_energy()
            randsh0_adsorbate_surface_dft_en = (
                randsh0_adsorbate_surface.get_potential_energy()
            )
            randsh0_solvent_surface_dft_en = (
                randsh0_solvent_surface.get_potential_energy()
            )

            dft_solvation_en = (randsh0_dft_en - randsh0_solvent_surface_dft_en) - (
                randsh0_adsorbate_surface_dft_en - randsh0_surface_dft_en
            )
            dft_solvent_ads_en = randsh0_dft_en - randsh0_solvent_surface_dft_en
            dft_clean_ads_en = randsh0_adsorbate_surface_dft_en - randsh0_surface_dft_en

            # predict ml predictions
            randsh0.calc = self.calculator
            randsh0_pred_en = randsh0.get_potential_energy()

            randsh0_surface.calc = self.calculator
            randsh0_surface_pred_en = randsh0_surface.get_potential_energy()

            randsh0_adsorbate_surface.calc = self.calculator
            randsh0_adsorbate_surface_pred_en = (
                randsh0_adsorbate_surface.get_potential_energy()
            )

            randsh0_solvent_surface.calc = self.calculator
            randsh0_solvent_surface_pred_en = (
                randsh0_solvent_surface.get_potential_energy()
            )

            ml_solvation_en = (randsh0_pred_en - randsh0_solvent_surface_pred_en) - (
                randsh0_adsorbate_surface_pred_en - randsh0_surface_pred_en
            )
            ml_solvent_ads_en = randsh0_pred_en - randsh0_solvent_surface_pred_en
            ml_clean_ads_en = (
                randsh0_adsorbate_surface_pred_en - randsh0_surface_pred_en
            )

            results = {
                "identifier": key,
                "dft_solvation_energy": dft_solvation_en,
                "dft_solvent_ads_energy": dft_solvent_ads_en,
                "dft_clean_ads_energy": dft_clean_ads_en,
                "ml_solvation_energy": ml_solvation_en,
                "ml_solvent_ads_energy": ml_solvent_ads_en,
                "ml_clean_ads_energy": ml_clean_ads_en,
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
            results: List of dictionaries containing solvation energy results
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
