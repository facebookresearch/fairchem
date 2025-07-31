"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import ase.io
import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize.lbfgs import LBFGS
from tqdm import tqdm

from fairchem.core.components.calculate import CalculateRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator


from ase.md import MDLogger


def get_thermo(filename):
    """
    read thermo logs.
    """
    with open(filename) as f:
        thermo = f.read().splitlines()
        sim_time, Et = [], []
        for i in range(1, len(thermo)):
            t, Etot, _, _, _ = (float(x) for x in thermo[i].split(" ") if x)
            sim_time.append(t)
            Et.append(Etot)
    return np.array(sim_time), np.array(Et)


MD22_TEMP = {
    "AT-AT-CG-CG": 500.0,
    "AT-AT": 500.0,
    "Ac-Ala3-NHMe": 500.0,
    "DHA": 500.0,
    "buckyball-catcher": 400.0,
    "double-walled_nanotube": 400.0,
    "stachyose": 500.0,
}

TM23_TEMP = {
    "Ag": 1235 * 1.25,
    "Au": 1337 * 1.25,
    "Cd": 594 * 1.25,
    "Co": 1768 * 1.25,
    "Cr": 2180 * 1.25,
    "Cu": 1358 * 1.25,
    "Fe": 1811 * 1.25,
    "Hf": 2506 * 1.25,
    "Hg": 234 * 1.25,
    "Ir": 2739 * 1.25,
    "Mn": 1519 * 1.25,
    "Mo": 2896 * 1.25,
    "Nb": 2750 * 1.25,
    "Ni": 1728 * 1.25,
    "Os": 3306 * 1.25,
    "Pd": 1828 * 1.25,
    "Pt": 2041 * 1.25,
    "Re": 3459 * 1.25,
    "Rh": 2237 * 1.25,
    "Ru": 2607 * 1.25,
    "Ta": 3290 * 1.25,
    "Tc": 2430 * 1.25,
    "Ti": 1941 * 1.25,
    "V": 2183 * 1.25,
    "W": 3695 * 1.25,
    "Zn": 693 * 1.25,
    "Zr": 2128 * 1.25,
}


def get_nve_md_data(dataset_root, dataset_name):
    MD22_MOLS = sorted(MD22_TEMP.keys())
    TM23_METALS = sorted(TM23_TEMP.keys())
    if dataset_name == "tm23":
        dataset = [
            (
                ase.io.read(f"{dataset_root}/tm23/{metal}_melt_nequip_test.xyz"),
                TM23_TEMP[metal],
            )
            for metal in TM23_METALS
        ]
    elif dataset_name == "md22":
        dataset = [
            (ase.io.read(f"{dataset_root}/md22/md22_{mol}.xyz"), MD22_TEMP[mol])
            for mol in MD22_MOLS
        ]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


class NVEMDRunner(CalculateRunner):
    """Perform a single point calculation of several structures/molecules.

    This class handles the single point calculation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    result_glob_pattern: ClassVar[str] = "thermo_*-*.log"

    def __init__(
        self,
        calculator: Calculator,
        input_data: Sequence,
        time_step: float,
        steps: float,
        save_frequency: int = 10,
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
        self.time_step = time_step
        self.steps = steps
        self.save_frequency = save_frequency
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
        assert num_jobs == len(
            self.input_data
        ), "num_jobs must be equal to the number of input data"
        atoms, temp = self.input_data[job_num]
        atoms.calc = self.calculator

        # relax
        opt = LBFGS(
            atoms,
            logfile=str(
                Path(self.job_config.metadata.results_dir)
                / f"relax_{num_jobs}-{job_num}.log"
            ),
        )
        opt.run(fmax=0.05, steps=1000)

        # run MD
        MaxwellBoltzmannDistribution(atoms, temp * units.kB)
        integrator = VelocityVerlet(atoms=atoms, timestep=self.time_step * units.fs)
        logger = MDLogger(
            dyn=integrator,
            atoms=atoms,
            logfile=str(
                Path(self.job_config.metadata.results_dir)
                / f"thermo_{num_jobs}-{job_num}.log"
            ),
            peratom=True,
        )
        integrator.attach(logger, interval=self.save_frequency)

        for _step in tqdm(range(self.steps)):
            integrator.run(1)

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
        assert (
            Path(self.job_config.metadata.results_dir)
            / f"thermo_{num_jobs}-{job_num}.log"
        ).exists()
        time, Et = get_thermo(
            str(
                Path(self.job_config.metadata.results_dir)
                / f"thermo_{num_jobs}-{job_num}.log"
            )
        )
        assert len(time) == len(Et)

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
