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
from pymatgen.io.ase import MSONAtoms
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from tqdm import tqdm
import numpy as np

from adjust_adsorbate import get_middle_mol
from adjust_adsorbate import get_large_adsorbate
from adjust_adsorbate import shift_adsorbate

from fairchem.core.components.calculate import CalculateRunner
from fairchem.core.components.calculate.recipes.relax import (
    relax_atoms,
)
from fairchem.core.components.calculate.recipes.utils import (
    get_property_dict_from_atoms,
)

from deform_relax_functions import relax_atoms_w_maxstep

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets import AseDBDataset


class DeformationRunner(CalculateRunner):
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
        relax_empty_cell: bool = True,
        save_relaxed_atoms: bool = True,
        normalize_properties_by: dict[str, str] | None = None,
        save_target_properties: Sequence[str] | None = None,
        **relax_kwargs,
    ):
        """Initialize the DeformationRunner.

        Args:
            calculator: ASE calculator to use for energy and force calculations
            input_data: Dataset containing atomic structures to process
            relax_empty_cell: Whether to relax the unti cell as part of the empty MOF relaxation. 
                Disable for calculators which do not implement stresses
            calculate_properties: Sequence of properties to calculate after relaxation
            save_relaxed_atoms (bool): Whether to save the relaxed structures in the results
            normalize_properties_by (dict[str, str] | None): Dictionary mapping property names to natoms or a key in
                atoms.info to normalize by
            save_target_properties (Sequence[str] | None): Sequence of target property names to save in the results file
                These properties need to be available using atoms.get_properties or present in the atoms.info dictionary
                This is useful if running a benchmark were errors will be computed after running relaxations
            relax_kwargs: Keyword arguments passed to relax. See defaults for deformation benchmark below:
                Args:
                atoms: ASE atoms with a calculator
                steps: max number of relaxation steps --> default: 1000
                fmax: force convergence threshold --> default: 0.05
                optimizer_cls: ASE optimizer. --> default: BFGS
                fix_symmetry: fix structure symmetry in relaxation. --> default: False
                cell_filter_cls: An instance of an ASE filter for relaxing empty MOF unit cell. --> default: FrechetCellFilter
        """

        filter_cls = None
        if relax_empty_cell:
            filter_cls = FrechetCellFilter

        DEFAULT_RELAX_KWARGS = {
            "steps": 1000,             # max number of relaxation steps
            "fmax": 0.05,              # force convergence threshold
            "maxstep": 0.05,           # max atomic displacement per iteration
            "optimizer_cls": BFGS,   # ASE optimizer class
            "fix_symmetry": False,     # keep symmetry
            "cell_filter_cls": filter_cls,  # default filter
        }

        self._calculate_properties = calculate_properties
        self._save_relaxed_atoms = save_relaxed_atoms
        self._normalize_properties_by = normalize_properties_by or {}
        self._save_target_properties = (
            save_target_properties if save_target_properties is not None else ()
        )

        self._relax_kwargs = DEFAULT_RELAX_KWARGS.copy()
        self._relax_kwargs.update(relax_kwargs)
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
                # separate MOF from adsorbate
                frame = atoms.copy()
                adsorbate = atoms.copy()
                tags = atoms.get_tags()
                for i in reversed(range(len(tags))):
                    if tags[i] != 0: # adsorbate atom found
                        del frame[i]
                    else:
                        del adsorbate[i]

                # relax empty MOF
                frame.calc = self.calculator
                frame = relax_atoms_w_maxstep(frame, **self._relax_kwargs)
                E_empty = frame.get_potential_energy()
                self._relax_kwargs['cell_filter_cls'] = None # remove cell filter after empty MOF relaxation

                # update adsorbate positions and combine
                adsorbate = shift_adsorbate(adsorbate, frame.cell)
                combo = frame + adsorbate

                # relax combo
                combo.calc = self.calculator
                combo = relax_atoms_w_maxstep(combo, **self._relax_kwargs)
                E_combo = combo.get_potential_energy()

                # single point frame
                frame = combo[:-3]
                frame.calc = self.calculator
                E_frame = frame.get_potential_energy()

                # frame re-relax
                frame = relax_atoms_w_maxstep(frame, **self._relax_kwargs)
                E_frame_relax = frame.get_potential_energy()

                # single point adsorbate
                adsorbate = combo[-3:]
                adsorbate.calc = self.calculator
                E_adsorbate = adsorbate.get_potential_energy()

                # gas phase adsorabte
                adsorbate_big = get_large_adsorbate(adsorbate)
                adsorbate_big.calc = self.calculator
                adsorbate_big = relax_atoms_w_maxstep(adsorbate_big, **self._relax_kwargs)
                E_adsorbate_gas = adsorbate_big.get_potential_energy()

                # compute energies
                E_ref = E_frame_relax
                if E_empty < E_frame_relax:
                    E_ref = E_empty

                E_ads = E_combo - E_ref - E_adsorbate_gas
                E_int = E_combo - E_frame - E_adsorbate
                E_mof_deform = E_frame - E_ref

                results.update(
                    get_property_dict_from_atoms(
                        self._calculate_properties, atoms, self._normalize_properties_by
                    )
                )
                results.update(
                    {
                        "E_ads": E_ads,
                        "E_int": E_int,
                        "E_mof_deform": E_mof_deform,
                        "E_ads_dft": atoms.info['E_ads_dft'],
                        "E_int_dft": atoms.info['E_int_dft'],
                        "E_mof_deform_dft": atoms.info['E_mof_deform_dft'],
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
                results["atoms"] = MSONAtoms(combo).as_dict()

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