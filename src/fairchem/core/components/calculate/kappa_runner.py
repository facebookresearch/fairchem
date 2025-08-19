"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import traceback
import warnings
from copy import deepcopy
from typing import Any, ClassVar

import ase.io
import ase.units
import numpy as np
import pandas as pd
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from monty.dev import requires

try:
    # pylint: disable=E0611
    from matbench_discovery.phonons import check_imaginary_freqs
    from matbench_discovery.phonons import thermal_conductivity as ltc
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter
    from pymatviz.enums import Key
    from tqdm import tqdm

    mbd_installed = True
except ImportError:
    mbd_installed = False

from fairchem.core.components.calculate import CalculateRunner


def get_kappa103_data_list(reference_data_path: str, debug=False):
    atoms = ase.io.read(reference_data_path, format="extxyz", index=":")
    if debug:
        atoms = atoms[:1]
    return atoms


@requires(
    mbd_installed,
    message="Requires `matbench_discovery[symmetry,phonons]` to be installed",
)
class KappaRunner(CalculateRunner):
    """Calculate elastic tensor for a set of structures."""

    result_glob_pattern: ClassVar[str] = "kappa103_dist*_*-*.json.gz"

    def __init__(
        self,
        calculator,
        input_data,
        displacement: float = 0.03,
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
        max_steps = 5000
        force_max = 1e-4  # Run until the forces are smaller than this in eV/A
        symprec = 1e-5
        enforce_relax_symm = True
        conductivity_broken_symm = False
        prog_bar = True
        # save_forces = False  # Save force sets to file
        temperatures = [300]

        all_results = []

        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]

        for i in tqdm(chunk_indices, desc="Running kappa calculations"):
            atoms = self.input_data[i]

            mat_id = atoms.info.get(Key.mat_id, f"id-{len(all_results)}")
            init_info = deepcopy(atoms.info)
            formula = atoms.info.get("name", "unknown")
            spg_num = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number

            info_dict = {
                "sid": mat_id,
                Key.mat_id: mat_id,
                Key.formula: formula,
                Key.spg_num: spg_num,
                "errors": [],
                "error_traceback": [],
            }

            relax_dict = {
                "max_stress": None,
                "reached_max_steps": False,
                "broken_symmetry": False,
            }

            try:
                # NOTE: not using the relax_atom recipe because need to use mask for cellfilter.
                atoms.calc = self.calculator
                if max_steps > 0:
                    if enforce_relax_symm:
                        atoms.set_constraint(FixSymmetry(atoms))

                    # Use standard mask for no-tilt constraint
                    filtered_atoms = FrechetCellFilter(
                        atoms, mask=[True] * 3 + [False] * 3
                    )

                    optimizer = FIRE(filtered_atoms, logfile=None)
                    optimizer.run(fmax=force_max, steps=max_steps)

                    reached_max_steps = optimizer.Nsteps >= max_steps
                    if reached_max_steps:
                        logging.info(
                            f"{mat_id=} reached {max_steps=} during relaxation."
                        )

                    max_stress = (
                        atoms.get_stress().reshape((2, 3), order="C").max(axis=1)
                    )
                    atoms.calc = None
                    atoms.constraints = None
                    atoms.info = init_info | atoms.info

                    # Check if symmetry was broken during relaxation
                    relaxed_spg = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
                    broken_symmetry = spg_num != relaxed_spg
                    relax_dict = {
                        "max_stress": max_stress,
                        "reached_max_steps": reached_max_steps,
                        "relaxed_space_group_number": relaxed_spg,
                        "broken_symmetry": broken_symmetry,
                    }

            except Exception as exc:
                warnings.warn(
                    f"Failed to relax {formula=}, {mat_id=}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                info_dict["errors"].append(f"RelaxError: {exc!r}")
                info_dict["error_traceback"].append(traceback.format_exc())
                results = info_dict | relax_dict
                all_results.append(results)
                continue

            # Calculation of force sets
            try:
                # Initialize phono3py with the relaxed structure
                ph3 = ltc.init_phono3py(
                    atoms,
                    fc2_supercell=atoms.info.get("fc2_supercell", [2, 2, 2]),
                    fc3_supercell=atoms.info.get("fc3_supercell", [2, 2, 2]),
                    q_point_mesh=atoms.info.get("q_point_mesh", [10, 10, 10]),
                    displacement_distance=self.displacement,
                    symprec=symprec,
                )

                # Calculate force constants and frequencies
                ph3, fc2_set, freqs = ltc.get_fc2_and_freqs(
                    ph3,
                    calculator=self.calculator,
                    pbar_kwargs={"leave": False, "disable": not prog_bar},
                )

                # Check for imaginary frequencies
                has_imaginary_freqs = check_imaginary_freqs(freqs)
                freqs_dict = {
                    Key.has_imag_ph_modes: has_imaginary_freqs,
                    Key.ph_freqs: freqs,
                }

                # If conductivity condition is met, calculate fc3
                ltc_condition = not has_imaginary_freqs and (
                    not relax_dict["broken_symmetry"] or conductivity_broken_symm
                )

                if ltc_condition:  # Calculate third-order force constants
                    logging.info(f"Calculating FC3 for {mat_id}")
                    ltc.calculate_fc3_set(
                        ph3,
                        calculator=self.calculator,
                        pbar_kwargs={"leave": False, "disable": not prog_bar},
                    )
                    ph3.produce_fc3(symmetrize_fc3r=True)

                if not ltc_condition:
                    results = info_dict | relax_dict | freqs_dict
                    warnings.warn(
                        f"{mat_id=} has imaginary frequencies or broken symmetry",
                        stacklevel=2,
                    )
                    all_results.append(results)
                    continue

            except Exception as exc:
                warnings.warn(
                    f"Failed to calculate force sets {mat_id}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                info_dict["errors"].append(f"ForceConstantError: {exc!r}")
                info_dict["error_traceback"].append(traceback.format_exc())
                results = info_dict | relax_dict
                all_results.append(results)
                continue

            try:  # Calculate thermal conductivity
                logging.info(f"Calculating kappa for {mat_id}")
                ph3, kappa_dict, _cond = ltc.calculate_conductivity(
                    ph3, temperatures=temperatures
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to calculate conductivity {mat_id}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                info_dict["errors"].append(f"ConductivityError: {exc!r}")
                info_dict["error_traceback"].append(traceback.format_exc())
                results = info_dict | relax_dict | freqs_dict
                all_results.append(results)
                continue

            results = info_dict | relax_dict | freqs_dict | kappa_dict
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
                f"kappa103_dist{self.displacement}_{num_jobs}-{job_num}.json.gz",
            )
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
