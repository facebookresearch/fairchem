"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import ase.io
import numpy as np
import pandas as pd
from ase.md import MDLogger

from fairchem.core.components.calculate._calculate_runner import CalculateRunner
from fairchem.core.components.calculate.parquet_trajectory import (
    ParquetTrajectoryWriter,
)
from fairchem.core.components.calculate.trajectory import (
    TrajectoryFrame,
    TrajectoryWriter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.md.md import MolecularDynamics


class MDRunner(CalculateRunner):
    """
    General-purpose molecular dynamics runner for single structures.

    This class provides a flexible framework for running MD simulations with any ASE
    dynamics integrator and any trajectory writer.
    """

    result_glob_pattern: ClassVar[str] = "trajectory_*.*"

    def __init__(
        self,
        calculator: Calculator,
        atoms: Atoms,
        dynamics: type[MolecularDynamics] | Callable,
        steps: int = 1000,
        trajectory_interval: int = 1,
        log_interval: int = 10,
        trajectory_writer: type[TrajectoryWriter]
        | Callable[..., TrajectoryWriter]
        | None = None,
        trajectory_writer_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the MDRunner for single-structure MD.

        Args:
            calculator: ASE calculator for energy/force calculations
            atoms: Single Atoms object to run MD on
            dynamics: ASE dynamics class or partial function. When using Hydra
                configs with _partial_: true, this will be a partial function with
                all parameters except 'atoms' already bound.
            steps: Total number of MD steps to run
            trajectory_interval: Interval for writing trajectory frames
            log_interval: Interval for writing thermodynamic data to log
            trajectory_writer: Trajectory writer class or factory function.
                Defaults to ParquetTrajectoryWriter if None.
            trajectory_writer_kwargs: Additional kwargs to pass to the trajectory
                writer constructor (e.g., flush_interval for parquet).
        """
        self._atoms = atoms
        self.dynamics = dynamics
        self.steps = steps
        self.trajectory_interval = trajectory_interval
        self.log_interval = log_interval
        self._trajectory_writer_class = trajectory_writer or ParquetTrajectoryWriter
        self._trajectory_writer_kwargs = trajectory_writer_kwargs or {}

        # State tracking
        self._dyn: MolecularDynamics | None = None
        self._trajectory_writer: TrajectoryWriter | None = None
        self._start_step = 0
        self._thermostat_state_to_restore: dict | None = None

        super().__init__(calculator=calculator, input_data=[atoms])

    # Known thermostat/barostat state attributes across different ASE dynamics classes
    # This list can be extended as new thermostats are added
    _THERMOSTAT_STATE_ATTRS: ClassVar[tuple[str, ...]] = (
        # Nose-Hoover NPT (ASE)
        "zeta",  # Thermostat friction
        "zeta_integrated",  # Integrated thermostat friction
        "eta",  # Barostat strain rate (3x3)
        "eta_past",  # Previous eta
        "zeta_past",  # Previous zeta
        "q",  # Fractional coordinates
        "q_past",  # Previous q
        "q_future",  # Future q
        "h",  # Cell matrix
        "h_past",  # Previous cell matrix
        # Nose-Hoover chain variants
        "xi",  # Alternative thermostat variable name
        "p_zeta",  # Chain momenta
        # Berendsen
        "tau",
        # General barostat
        "strain",
        "vbox",
        # Parrinello-Rahman
        "h_inv",
        "inv_h",
        "h0",
        "v",
    )

    # Nested thermostat state for NoseHooverChainNVT and related classes
    _NESTED_THERMOSTAT_ATTRS: ClassVar[tuple[str, ...]] = (
        "_eta",  # Thermostat positions
        "_p_eta",  # Thermostat momenta
    )

    def _get_trajectory_extension(self) -> str:
        """
        Get the file extension for the trajectory based on writer type.

        Returns:
            File extension string (e.g., ".parquet", ".traj")
        """
        writer_name = getattr(
            self._trajectory_writer_class,
            "__name__",
            getattr(
                self._trajectory_writer_class,
                "func",
                type(self._trajectory_writer_class),
            ).__name__,
        )
        if "Parquet" in writer_name:
            return ".parquet"
        elif "ASE" in writer_name:
            return ".traj"
        else:
            return ".traj"

    def _save_thermostat_state(self, dyn: MolecularDynamics) -> dict:
        """
        Extract all thermostat/barostat state from dynamics object.

        This method introspects the dynamics object and saves:
        1. Known thermostat state attributes (xi, eta, zeta, etc.)
        2. Nested thermostat state (for NoseHooverChainNVT and similar)
        3. NumPy RNG state if the dynamics uses stochastic methods
        4. The dynamics class name for verification on restore

        Args:
            dyn: The ASE dynamics object

        Returns:
            Dictionary containing all saved state (JSON-serializable)
        """
        state = {
            "class_name": type(dyn).__name__,
            "attrs": {},
            "nested_thermostat": {},
        }

        for attr in self._THERMOSTAT_STATE_ATTRS:
            if hasattr(dyn, attr):
                value = getattr(dyn, attr)
                if isinstance(value, (int, float)):
                    state["attrs"][attr] = value
                elif isinstance(value, np.ndarray):
                    state["attrs"][attr] = value.tolist()

        # Handle nested thermostat state (NoseHooverChainNVT, etc.)
        if hasattr(dyn, "_thermostat"):
            thermostat = dyn._thermostat
            for attr in self._NESTED_THERMOSTAT_ATTRS:
                if hasattr(thermostat, attr):
                    value = getattr(thermostat, attr)
                    if isinstance(value, (int, float)):
                        state["nested_thermostat"][attr] = value
                    elif isinstance(value, np.ndarray):
                        state["nested_thermostat"][attr] = value.tolist()

        if hasattr(dyn, "rng") and dyn.rng is np.random:
            rng_state = np.random.get_state()
            state["numpy_random_state"] = {
                "name": rng_state[0],
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            }

        return state

    def _restore_thermostat_state(
        self, dyn: MolecularDynamics, state: dict | None
    ) -> None:
        """
        Restore thermostat/barostat state to dynamics object.

        Args:
            dyn: The ASE dynamics object
            state: Previously saved state dictionary, or None
        """
        if state is None:
            return

        saved_class = state.get("class_name", "")
        current_class = type(dyn).__name__
        assert (
            saved_class == current_class
        ), f"Restoring state from {saved_class} to {current_class}"

        for attr, value in state.get("attrs", {}).items():
            if hasattr(dyn, attr):
                current = getattr(dyn, attr)
                if isinstance(value, list) and isinstance(current, np.ndarray):
                    setattr(dyn, attr, np.array(value))
                elif isinstance(value, (int, float)) and isinstance(
                    current, (int, float)
                ):
                    setattr(dyn, attr, value)

        # Handle nested thermostat state (NoseHooverChainNVT, etc.)
        if hasattr(dyn, "_thermostat") and "nested_thermostat" in state:
            thermostat = dyn._thermostat
            for attr, value in state["nested_thermostat"].items():
                if hasattr(thermostat, attr):
                    current = getattr(thermostat, attr)
                    if isinstance(value, list) and isinstance(current, np.ndarray):
                        setattr(thermostat, attr, np.array(value))
                    elif isinstance(value, (int, float)) and isinstance(
                        current, (int, float)
                    ):
                        setattr(thermostat, attr, value)

        if (
            "numpy_random_state" in state
            and hasattr(dyn, "rng")
            and dyn.rng is np.random
        ):
            rng = state["numpy_random_state"]
            np.random.set_state(
                (
                    rng["name"],
                    np.array(rng["keys"], dtype=np.uint32),
                    rng["pos"],
                    rng["has_gauss"],
                    rng["cached_gaussian"],
                )
            )

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> dict[str, Any]:
        """
        Run MD simulation on a single structure.

        Args:
            job_num: Current job number (used for file naming)
            num_jobs: Total number of jobs (used for file naming)

        Returns:
            Dictionary containing MD results and metadata
        """
        results_dir = Path(self.job_config.metadata.results_dir)
        sid = self._atoms.info.get("sid", job_num)

        extension = self._get_trajectory_extension()
        trajectory_file = results_dir / f"trajectory_{num_jobs}-{job_num}{extension}"
        log_file = results_dir / f"thermo_{num_jobs}-{job_num}.log"

        self._atoms.calc = self.calculator
        self._dyn = self.dynamics(atoms=self._atoms)

        if self._thermostat_state_to_restore is not None:
            self._restore_thermostat_state(self._dyn, self._thermostat_state_to_restore)

        self._trajectory_writer = self._trajectory_writer_class(
            trajectory_file, **self._trajectory_writer_kwargs
        )

        # Attach trajectory collector with global step alignment
        # We use interval=1 and check alignment manually to handle checkpoint resume correctly
        def collect_frame():
            global_step = self._dyn.get_number_of_steps() + self._start_step
            if global_step % self.trajectory_interval == 0:
                frame = TrajectoryFrame.from_atoms(
                    self._atoms,
                    step=global_step,
                    time=self._dyn.get_time(),
                )
                self._trajectory_writer.append(frame)

        self._dyn.attach(collect_frame, interval=1)

        logger = MDLogger(
            dyn=self._dyn,
            atoms=self._atoms,
            logfile=log_file,
            header=self._start_step == 0,
            mode="a" if self._start_step > 0 else "w",
        )

        def log_with_alignment():
            global_step = self._dyn.get_number_of_steps() + self._start_step
            if global_step % self.log_interval == 0:
                logger()

        self._dyn.attach(log_with_alignment, interval=1)

        remaining_steps = self.steps - self._start_step
        self._dyn.run(remaining_steps)

        self._trajectory_writer.close()

        return {
            "trajectory_file": str(trajectory_file),
            "log_file": str(log_file),
            "total_steps": self.steps,
            "start_step": self._start_step,
            "structure_id": sid,
        }

    def write_results(
        self,
        results: dict[str, Any],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """
        Write results metadata after simulation completes.

        Trajectory data is already written during simulation.
        This method writes summary metadata as JSON.

        Args:
            results: Dictionary containing results from calculate()
            results_dir: Directory path where results are saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        trajectory_file = Path(results["trajectory_file"])
        log_file = Path(results["log_file"])

        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        if trajectory_file.suffix == ".parquet":
            df = pd.read_parquet(trajectory_file)
            num_frames = len(df)
        else:
            num_frames = (
                self._trajectory_writer.total_frames if self._trajectory_writer else 0
            )

        dynamics_name = getattr(
            self.dynamics,
            "__name__",
            getattr(self.dynamics, "func", type(self.dynamics)).__name__,
        )

        metadata = {
            "trajectory_file": str(trajectory_file),
            "log_file": str(log_file),
            "total_steps": results["total_steps"],
            "num_frames": num_frames,
            "trajectory_interval": self.trajectory_interval,
            "log_interval": self.log_interval,
            "dynamics_class": dynamics_name,
            "structure_id": results["structure_id"],
        }

        metadata_file = Path(results_dir) / f"metadata_{num_jobs}-{job_num}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        """
        Save current MD state for resumption.

        Saves:
        - Atoms state (positions, velocities) in ExtXYZ format
        - Thermostat/barostat state in JSON format
        - MD metadata (step count, etc.) in JSON format

        Args:
            checkpoint_location: Directory to save checkpoint files
            is_preemption: Whether this save is due to preemption

        Returns:
            bool: True if state was successfully saved
        """
        if self._dyn is None or self._atoms is None:
            return False

        checkpoint_dir = Path(checkpoint_location)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self._trajectory_writer:
                self._trajectory_writer.close()

            atoms_path = checkpoint_dir / "checkpoint.xyz"
            ase.io.write(str(atoms_path), self._atoms, format="extxyz")

            thermostat_state = self._save_thermostat_state(self._dyn)
            thermostat_path = checkpoint_dir / "thermostat_state.json"
            with open(thermostat_path, "w") as f:
                json.dump(thermostat_state, f)

            md_state = {
                "current_step": self._dyn.get_number_of_steps() + self._start_step,
                "total_steps": self.steps,
                "trajectory_frames_written": (
                    self._trajectory_writer.total_frames
                    if self._trajectory_writer
                    else 0
                ),
            }
            state_path = checkpoint_dir / "md_state.json"
            with open(state_path, "w") as f:
                json.dump(md_state, f)

            logging.info(f"Saved MD checkpoint to {checkpoint_dir}")
            return True
        except Exception as e:
            logging.exception(f"Failed to save checkpoint: {e}")
            return False

    def load_state(self, checkpoint_location: str | None) -> None:
        """
        Load MD state from checkpoint.

        Restores:
        - Atoms positions and velocities from ExtXYZ
        - Starting step count from metadata
        - Thermostat/barostat state (applied after dynamics creation)

        Args:
            checkpoint_location: Directory containing checkpoint files, or None
        """
        if checkpoint_location is None:
            return

        checkpoint_dir = Path(checkpoint_location)
        atoms_path = checkpoint_dir / "checkpoint.xyz"
        state_path = checkpoint_dir / "md_state.json"

        if not atoms_path.exists() or not state_path.exists():
            return

        self._atoms = ase.io.read(str(atoms_path), format="extxyz")

        with open(state_path) as f:
            md_state = json.load(f)

        self._start_step = md_state["current_step"]

        thermostat_path = checkpoint_dir / "thermostat_state.json"
        if thermostat_path.exists():
            with open(thermostat_path) as f:
                self._thermostat_state_to_restore = json.load(f)
        else:
            self._thermostat_state_to_restore = None

        logging.info(
            f"Loaded MD checkpoint from {checkpoint_dir}, resuming from step {self._start_step}"
        )
