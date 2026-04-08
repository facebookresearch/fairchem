"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from ase.io.jsonio import encode
from tqdm import tqdm

from fairchem.core.components.calculate._calculate_runner import CalculateRunner
from fairchem.core.components.calculate.recipes.relax import (
    relax_atoms,
)
from fairchem.core.components.calculate.recipes.utils import (
    get_property_dict_from_atoms,
)
from fairchem.core.components.runner import (
    PreemptableMixin,
    StopfairDetected,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ase.calculators.calculator import Calculator

    from fairchem.core.datasets.atoms_sequence import AtomsSequence


class RelaxationRunner(PreemptableMixin, CalculateRunner):
    """Relax a sequence of several structures/molecules.

    This class handles the relaxation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    result_glob_pattern: ClassVar[str] = "relaxation_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: AtomsSequence,
        calculate_properties: Sequence[str] = ["energy"],
        save_relaxed_atoms: bool = True,
        normalize_properties_by: dict[str, str] | None = None,
        save_target_properties: Sequence[str] | None = None,
        heartbeat_interval: int | None = None,
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
            heartbeat_interval: Interval in optimizer steps for checking for a
                STOPFAIR file. If a STOPFAIR file is found, the in-progress
                structure is abandoned and partial progress is saved to the
                checkpoint. If None, no STOPFAIR checking is performed.
            relax_kwargs: Keyword arguments passed to relax. See signature of calculate.recipes.relax_atoms for options
        """
        self._calculate_properties = calculate_properties
        self._save_relaxed_atoms = save_relaxed_atoms
        self._normalize_properties_by = normalize_properties_by or {}
        self._save_target_properties = (
            save_target_properties if save_target_properties is not None else ()
        )
        self._relax_kwargs = relax_kwargs
        self.heartbeat_interval = heartbeat_interval

        # State tracking for checkpointing
        self._completed_results: list[dict[str, Any]] = []
        self._completed_sids: set = set()
        self._current_results: list[dict[str, Any]] = []
        self._current_job_num: int = 0
        self._current_num_jobs: int = 1
        self._stopped_by_stopfair: bool = False

        super().__init__(calculator=calculator, input_data=input_data)

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        """Perform relaxation calculations on a subset of structures.

        Splits the input data into chunks and processes the chunk corresponding
        to job_num. Skips structures already completed in a previous run
        (loaded from checkpoint). On STOPFAIR, saves partial progress and
        stops gracefully without writing the final results file.

        Args:
            job_num (int, optional): Current job number in array job. Defaults to 0.
            num_jobs (int, optional): Total number of jobs in array. Defaults to 1.

        Returns:
            list[dict[str, Any]] - List of dictionaries containing calculation results
        """
        self._current_job_num = job_num
        self._current_num_jobs = num_jobs
        self._stopped_by_stopfair = False
        self._current_results = list(self._completed_results)

        chunk_indices = np.array_split(range(len(self.input_data)), num_jobs)[job_num]

        # Build STOPFAIR observer if heartbeat_interval is configured
        stopfair_observers = None
        if self.heartbeat_interval is not None and self.heartbeat_interval > 0:

            def check_stopfair():
                if self.check_stopfair():
                    raise StopfairDetected

            stopfair_observers = [(check_stopfair, self.heartbeat_interval)]

        try:
            for i in tqdm(chunk_indices, desc="Running relaxations"):
                atoms = self.input_data[i]
                sid = atoms.info.get("sid", i)

                if sid in self._completed_sids:
                    continue

                results = {
                    "sid": sid,
                    "natoms": len(atoms),
                }

                # add target properties if requested
                target_properties = get_property_dict_from_atoms(
                    self._save_target_properties, atoms, self._normalize_properties_by
                )
                results.update(
                    {
                        f"{key}_target": target_properties[key]
                        for key in target_properties
                    }
                )
                if self._save_relaxed_atoms:
                    results["atoms_initial"] = encode(
                        atoms
                    )  # Note this does not save atoms.info!

                try:
                    atoms.calc = self.calculator
                    atoms = relax_atoms(
                        atoms, observers=stopfair_observers, **self._relax_kwargs
                    )
                    results.update(
                        get_property_dict_from_atoms(
                            self._calculate_properties,
                            atoms,
                            self._normalize_properties_by,
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

                except StopfairDetected:
                    # Current structure is incomplete — do NOT append it.
                    save_path = self.job_config.metadata.preemption_checkpoint_dir
                    logging.info(
                        f"STOPFAIR detected at structure {i}, saving "
                        f"{len(self._current_results)} completed results to {save_path}"
                    )
                    self.save_state(save_path, is_preemption=True)
                    self.stopfair_path.unlink(missing_ok=True)
                    raise  # propagate to outer except StopfairDetected

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
                    results["atoms"] = encode(atoms)

                self._current_results.append(results)

                # Save checkpoint after every completed structure
                self.save_state(
                    self.job_config.metadata.preemption_checkpoint_dir,
                    is_preemption=False,
                )

        except StopfairDetected:
            self._stopped_by_stopfair = True

        return self._current_results

    def write_results(
        self,
        results: list[dict[str, Any]],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """Write calculation results to a compressed JSON file.

        Skips writing if the run was stopped by a STOPFAIR signal; partial
        results remain in the checkpoint for resumption.

        Args:
            results: List of dictionaries containing elastic properties
            results_dir: Directory path where results will be saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        if self._stopped_by_stopfair:
            return
        results_df = pd.DataFrame(results)
        results_df.to_json(
            os.path.join(results_dir, f"relaxation_{num_jobs}-{job_num}.json.gz")
        )

    def save_simulation_state(self, checkpoint_dir: Path, is_preemption: bool) -> None:
        """Save the current relaxation progress into checkpoint_dir.

        The progress file name uses underscores (``relaxation_checkpoint_N_M.json.gz``)
        so it does not match ``result_glob_pattern`` (``relaxation_*-*.json.gz``).

        Args:
            checkpoint_dir: Directory to write state files into.
            is_preemption: Whether this save is due to preemption.
        """
        progress_file = (
            checkpoint_dir
            / f"relaxation_checkpoint_{self._current_num_jobs}_{self._current_job_num}.json.gz"
        )
        pd.DataFrame(self._current_results).to_json(progress_file)

    def load_state(self, checkpoint_location: str | None) -> None:
        """Load a previously saved relaxation state from a checkpoint.

        Calls the CalculateRunner base first to handle fully-completed runs
        (which copies the result file and sets ``_already_calculated``). Then
        delegates to PreemptableRunner to load partial progress via
        load_simulation_state.

        Args:
            checkpoint_location: Directory containing the checkpoint, or None
        """
        CalculateRunner.load_state(self, checkpoint_location)
        if self._already_calculated or checkpoint_location is None:
            return
        PreemptableMixin.load_state(self, checkpoint_location)

    def load_simulation_state(self, checkpoint_dir: Path) -> None:
        """Load partial relaxation progress from checkpoint_dir.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
        """
        if not checkpoint_dir.exists():
            return

        job_array_num = self.job_config.metadata.array_job_num
        job_array_total = self.job_config.scheduler.num_array_jobs
        progress_file = (
            checkpoint_dir
            / f"relaxation_checkpoint_{job_array_total}_{job_array_num}.json.gz"
        )
        if not progress_file.exists():
            return

        df_progress = pd.read_json(progress_file)
        self._completed_results = df_progress.to_dict(orient="records")
        self._completed_sids = {r["sid"] for r in self._completed_results}
        logging.info(
            f"Loaded {len(self._completed_results)} completed structures from checkpoint at "
            f"{checkpoint_dir}, will resume after skipping those."
        )
