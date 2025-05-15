"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from ase.optimize import LBFGS
from pymatgen.io.ase import MSONAtoms
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.optimize import Optimizer


class AdsorbMLRunner(CalculateRunner):
    """
    Run the AdsorbML pipeline to identify the global minima adsorption energy.
    The option to also relax a clean surface is also provided.

    This class handles the relaxation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    result_glob_pattern: ClassVar[str] = "adsorbml_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data_path: str,
        place_on_relaxed_slab: bool = False,
        save_relaxed_atoms: bool = True,
        adsorption_energy_model: bool = False,
        num_placements: int = 100,
        optimizer_cls: type[Optimizer] = LBFGS,
        fmax: float = 0.02,
        steps: int = 300,
    ):
        """
        Initialize the AdsorbMLRunner

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data_path: path to dataset containing slab objects and the names of the adsorbates
            and the minimum DFT adsorption energy target
            save_relaxed_atoms (bool): Whether to save the relaxed structures in the results
            place_on_relaxed_slab (bool): Whether to relax the bare slab before placement
            adsorption_energy_model (bool): Whether to use the adsorption energy model
            num_placements (int): Number of placements to consider
            optimizer_cls (Optimizer): Optimizer to use
            fmax (float): force convergence threshold
            steps (int): max number of relaxation steps
        """
        self._save_relaxed_atoms = save_relaxed_atoms
        self.place_on_relaxed_slab = place_on_relaxed_slab
        self.adsorption_energy_model = adsorption_energy_model
        self.num_placements = num_placements
        self.fmax = fmax
        self.steps = steps
        self.optimizer_cls = optimizer_cls
        with open(input_data_path, "rb") as f:
            input_data = pickle.load(f)
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
        sysids = list(self.input_data.keys())
        chunk_indices = np.array_split(sysids, num_jobs)[job_num]
        for sysid in tqdm(chunk_indices, desc="Running AdsorbML"):
            initial_slab = self.input_data[sysid]["slab_initial"]
            relaxed_slab = self.input_data[sysid]["slab_relax"]
            adsorbate = self.input_data[sysid]["adsorbate"]
            target = self.input_data[sysid]["min_dft_ads_energy"]

            outputs = run_adsorbml(
                initial_slab,
                adsorbate,
                self.calculator,
                optimizer_cls=self.optimizer_cls,
                fmax=self.fmax,
                steps=self.steps,
                num_placements=self.num_placements,
                reference_ml_energies=not self.adsorption_energy_model,
                relaxed_slab_atoms=relaxed_slab.atoms  # In the case of adsorption energy model, use the DFT relaxed slab
                if self.adsorption_energy_model
                else None,
                place_on_relaxed_slab=self.place_on_relaxed_slab,
            )
            top_candidates = outputs["adslabs"]
            if len(top_candidates) == 0:
                ml_energy = np.inf
            else:
                ml_energy = top_candidates[0]["results"]["energy"]
                if not self.adsorption_energy_model:
                    ml_energy = top_candidates[0]["results"][
                        "referenced_adsorption_energy"
                    ]["adsorption_energy"]

            results = {
                "sid": sysid,
                "energy_target": target,
                "energy": ml_energy,
                "anomaly_count": sum([len(x) for x in outputs["adslab_anomalies"]]),
            }
            if self._save_relaxed_atoms and len(top_candidates) > 0:
                results["atoms"] = MSONAtoms(top_candidates[0]["atoms"]).as_dict()

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
            os.path.join(results_dir, f"adsorbml_{num_jobs}-{job_num}.json.gz")
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        return
