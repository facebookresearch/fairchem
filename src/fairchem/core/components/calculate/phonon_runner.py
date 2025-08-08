"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import phonopy
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.phonons import (
    run_mdr_phonon_benchmark,
)


def get_mdr_phonon_data_list(index_df_path, phonon_file_path, debug=False):
    ref_df = pd.read_json(index_df_path)
    if debug:
        ref_df = ref_df[ref_df["nsites"] <= 5].head(10)
    all_ids = list(ref_df["mp_id"])
    phonon_yamls = [Path(phonon_file_path) / f"{mpid}.yaml.bz2" for mpid in all_ids]
    return phonon_yamls


class MDRPhononRunner(CalculateRunner):
    """Calculate elastic tensor for a set of structures."""

    result_glob_pattern: ClassVar[str] = "mdr_phonon_dist*_*-*.json.gz"

    def __init__(
        self,
        calculator,
        input_data,
        displacement: float = 0.01,
    ):
        """Initialize the CalculateRunner with a calculator and input data.

        Args:
            calculator (Calculator): ASE-like calculator to perform calculations
            input_data (Sequence): Input data to be processed by the calculator
        """
        super().__init__(calculator, input_data)
        self.displacement = displacement

    # TODO continue if unfinished
    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        all_results = []
        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]
        for i in tqdm(chunk_indices, desc="Running phonon calculations."):
            try:
                phonon_file_name = self.input_data[i]
                phonon = phonopy.load(phonon_file_name)
                phonon_results = run_mdr_phonon_benchmark(
                    phonon,
                    calculator=self.calculator,
                    displacement=self.displacement,
                    run_relax=True,
                    fix_symm_relax=False,
                    symprec=1e-4,
                    symmetrize_fc=False,
                )

                frequencies = phonon_results["frequencies"]
                max_freq = frequencies.max()
                avg_freq = frequencies.mean()
                # temp at 300 K
                entropy = phonon_results["entropy"][4]
                heat_capacity = phonon_results["heat_capacity"][4]
                free_energy = phonon_results["free_energy"][4]

                results = {
                    "sid": phonon_file_name.stem.split(".")[0],
                    "energy_per_atom": phonon_results["energy_per_atom"],
                    "volume_per_atom": phonon_results["volume_per_atom"],
                    "max_freq": max_freq,
                    "avg_freq": avg_freq,
                    "entropy": entropy,
                    "heat_capacity": heat_capacity,
                    "free_energy": free_energy,
                    "errors": "",
                    "traceback": "",
                }

            except Exception as ex:
                results = {
                    "sid": phonon_file_name.stem.split(".")[0],
                    "energy_per_atom": np.nan,
                    "volume_per_atom": np.nan,
                    "max_freq": np.nan,
                    "avg_freq": np.nan,
                    "entropy": np.nan,
                    "heat_capacity": np.nan,
                    "free_energy": np.nan,
                    "errors": (f"{ex!r}"),
                    "traceback": (traceback.format_exc()),
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
        results_df = pd.DataFrame(results)
        results_df.to_json(
            os.path.join(
                results_dir,
                f"mdr_phonon_dist{self.displacement}_{num_jobs}-{job_num}.json.gz",
            )
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
