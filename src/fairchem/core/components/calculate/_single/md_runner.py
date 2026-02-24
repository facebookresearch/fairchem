"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import ase.io
import numpy as np
import pandas as pd
from ase.md import MDLogger
from omegaconf import OmegaConf

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


class _StopfairDetected(Exception):
    """
    Raised by the STOPFAIR callback to break out of dyn.run().
    """


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
        dynamics: type[MolecularDynamics] | Callable,
        atoms: Atoms | None = None,
        steps: int = 1000,
        trajectory_interval: int = 1,
        log_interval: int = 10,
        checkpoint_interval: int | None = None,
        heartbeat_interval: int | None = None,
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
            checkpoint_interval: Interval (in steps) for saving a rolling
                checkpoint of the simulation state. If None, no periodic
                checkpointing is performed.
            heartbeat_interval: Interval (in steps) for checking for a
                STOPFAIR file in run_dir. If a STOPFAIR file is found, the
                simulation saves state and stops gracefully. If None, no
                STOPFAIR checking is performed.
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
        self.checkpoint_interval = checkpoint_interval
        self.heartbeat_interval = heartbeat_interval
        self._trajectory_writer_class = trajectory_writer or ParquetTrajectoryWriter
        self._trajectory_writer_kwargs = trajectory_writer_kwargs or {}

        # State tracking
        self._dyn: MolecularDynamics | None = None
        self._trajectory_writer: TrajectoryWriter | None = None
        self._start_step = 0
        self._thermostat_state_to_restore: dict | None = None

        super().__init__(calculator=calculator, input_data=[atoms])

    # Supported dynamics classes for checkpoint save/restore
    _SUPPORTED_DYNAMICS: ClassVar[tuple[str, ...]] = (
        "VelocityVerlet",
        "NoseHooverChainNVT",
        "Bussi",
        "Langevin",
    )

    def _get_trajectory_extension(self) -> str:
        """
        Get the file extension for the trajectory based on writer type.

        Returns:
            File extension string (e.g., ".parquet")
        """
        return ".parquet"

    def _save_thermostat_state(self, dyn: MolecularDynamics) -> dict:
        """
        Extract thermostat state from dynamics object.

        Supports VelocityVerlet (NVE, no thermostat state),
        NoseHooverChainNVT (saves chain positions and momenta),
        Bussi (saves RNG state and transferred energy),
        and Langevin (saves RNG state).

        Args:
            dyn: The ASE dynamics object

        Returns:
            Dictionary containing saved state (JSON-serializable)
        """
        class_name = type(dyn).__name__
        if class_name not in self._SUPPORTED_DYNAMICS:
            raise ValueError(
                f"Unsupported dynamics class '{class_name}' for checkpointing. "
                f"Supported: {self._SUPPORTED_DYNAMICS}"
            )

        state: dict[str, Any] = {"class_name": class_name}

        if class_name == "NoseHooverChainNVT":
            thermostat = dyn._thermostat
            state["eta"] = thermostat._eta.tolist()
            state["p_eta"] = thermostat._p_eta.tolist()
        elif class_name in ("Bussi", "Langevin"):
            rng_state = dyn.rng.get_state()
            state["rng_state"] = {
                "algorithm": rng_state[0],
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            }
            if hasattr(dyn, "transferred_energy"):
                state["transferred_energy"] = float(dyn.transferred_energy)

        return state

    def _restore_thermostat_state(
        self, dyn: MolecularDynamics, state: dict | None
    ) -> None:
        """
        Restore thermostat state to dynamics object.

        Supports VelocityVerlet (NVE, no-op), NoseHooverChainNVT
        (restores chain positions and momenta), Bussi
        (restores RNG state and transferred energy),
        and Langevin (restores RNG state).

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

        if current_class == "NoseHooverChainNVT":
            thermostat = dyn._thermostat
            thermostat._eta = np.array(state["eta"])
            thermostat._p_eta = np.array(state["p_eta"])
        elif current_class in ("Bussi", "Langevin"):
            rng_state = state["rng_state"]
            dyn.rng.set_state(
                (
                    rng_state["algorithm"],
                    np.array(rng_state["keys"], dtype=np.uint32),
                    rng_state["pos"],
                    rng_state["has_gauss"],
                    rng_state["cached_gaussian"],
                )
            )
            if "transferred_energy" in state:
                dyn.transferred_energy = state["transferred_energy"]

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> dict[str, Any]:
        """
        Run MD simulation on a single structure.

        Args:
            job_num: Current job number (used for file naming)
            num_jobs: Total number of jobs (used for file naming)

        Returns:
            Dictionary containing MD results and metadata
        """
        assert self._atoms is not None, (
            "No atoms provided. Pass atoms= to MDRunner or set runner_state_path "
            "in the config to resume from a checkpoint."
        )

        results_dir = Path(self.job_config.metadata.results_dir)
        sid = self._atoms.info.get("sid", job_num)

        extension = self._get_trajectory_extension()
        trajectory_file = results_dir / f"trajectory_{num_jobs}-{job_num}{extension}"
        log_file = results_dir / f"thermo_{num_jobs}-{job_num}.log"

        self._atoms.calc = self.calculator
        self._dyn = self.dynamics(atoms=self._atoms)

        if self._thermostat_state_to_restore is not None:
            self._restore_thermostat_state(self._dyn, self._thermostat_state_to_restore)

        # Restore the step counter so get_time() returns global time
        self._dyn.nsteps = self._start_step

        self._trajectory_writer = self._trajectory_writer_class(
            trajectory_file, **self._trajectory_writer_kwargs
        )

        # Attach trajectory collector with global step alignment
        # We use interval=1 and check alignment manually to handle checkpoint resume correctly
        def collect_frame():
            global_step = self._dyn.get_number_of_steps()
            self._atoms.info["md_step"] = global_step
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
            header=True,
            mode="a" if self._start_step > 0 else "w",
        )

        def log_with_alignment():
            global_step = self._dyn.get_number_of_steps()
            if global_step % self.log_interval == 0:
                logger()

        self._dyn.attach(log_with_alignment, interval=1)

        # Attach STOPFAIR checker if heartbeat_interval is configured
        if self.heartbeat_interval is not None and self.heartbeat_interval > 0:
            stopfair_path = (
                Path(self.job_config.metadata.checkpoint_dir).parent / "STOPFAIR"
            )

            def check_stopfair():
                if self._dyn.get_number_of_steps() == self._start_step:
                    return
                if stopfair_path.exists():
                    current_step = self._dyn.get_number_of_steps()
                    save_path = self.job_config.metadata.preemption_checkpoint_dir
                    logging.info(
                        f"STOPFAIR detected in {stopfair_path.parent}, "
                        f"saving state to {save_path} at step {current_step}"
                    )
                    if self.save_state(save_path, is_preemption=True):
                        stopfair_path.unlink()
                    raise _StopfairDetected

            self._dyn.attach(check_stopfair, interval=self.heartbeat_interval)

        # Attach periodic checkpoint saving if checkpoint_interval is configured
        if self.checkpoint_interval is not None and self.checkpoint_interval > 0:
            checkpoint_save_path = self.job_config.metadata.preemption_checkpoint_dir

            def save_periodic_checkpoint():
                if self._dyn.get_number_of_steps() == self._start_step:
                    return
                current_step = self._dyn.get_number_of_steps()
                logging.info(
                    f"Saving periodic checkpoint at step {current_step} "
                    f"to {checkpoint_save_path}"
                )
                self.save_state(checkpoint_save_path, is_preemption=False)

            self._dyn.attach(
                save_periodic_checkpoint, interval=self.checkpoint_interval
            )

        remaining_steps = self.steps - self._start_step
        stopped_by_stopfair = False

        # On resume, ASE's irun() skips the initial call_observers() because
        # nsteps != 0. Manually trigger it so the resume-step frame is captured
        # and the logger records the initial state.
        if self._start_step > 0:
            self._dyn.call_observers()

        try:
            self._dyn.run(remaining_steps)
        except _StopfairDetected:
            stopped_by_stopfair = True

        if not stopped_by_stopfair:
            self._trajectory_writer.close()

        return {
            "trajectory_file": str(trajectory_file),
            "log_file": str(log_file),
            "total_steps": self.steps,
            "start_step": self._start_step,
            "structure_id": sid,
            "stopped_by_stopfair": stopped_by_stopfair,
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

        traj_df = pd.read_parquet(trajectory_file)
        num_frames = len(traj_df)

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
        - resume_config.yaml: canonical config with runner_state_path set
        - portable_config.yaml: config stripped of system-specific fields
          (metadata, timestamp_id) so it can be run on a new machine

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
                if is_preemption:
                    self._trajectory_writer.close()
                elif hasattr(self._trajectory_writer, "flush"):
                    self._trajectory_writer.flush()

            atoms_path = checkpoint_dir / "checkpoint.xyz"
            current_step = self._dyn.get_number_of_steps()
            self._atoms.info["md_step"] = current_step
            ase.io.write(str(atoms_path), self._atoms, format="extxyz")

            thermostat_state = self._save_thermostat_state(self._dyn)
            thermostat_path = checkpoint_dir / "thermostat_state.json"
            with open(thermostat_path, "w") as f:
                json.dump(thermostat_state, f)

            md_state = {
                "current_step": current_step,
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

            # Save resume configs from the canonical config
            config_path = self.job_config.metadata.config_path
            if os.path.exists(config_path):
                cfg = OmegaConf.load(config_path)
                cfg.job.runner_state_path = checkpoint_location
                # Remove atoms from runner since state is in checkpoint.xyz
                if "atoms" in cfg.get("runner", {}):
                    del cfg.runner.atoms

                # System-specific resume config (same machine)
                resume_path = checkpoint_dir / "resume_config.yaml"
                OmegaConf.save(cfg, resume_path)

                # Portable config (new machine): strip auto-generated fields
                portable_cfg = cfg.copy()
                portable_cfg.job.runner_state_path = "."
                if "run_dir" in portable_cfg.get("job", {}):
                    run_dir = Path(portable_cfg.job.run_dir)
                    portable_cfg.job.run_dir = run_dir.name
                if "metadata" in portable_cfg.get("job", {}):
                    del portable_cfg.job.metadata
                if "timestamp_id" in portable_cfg.get("job", {}):
                    del portable_cfg.job.timestamp_id
                portable_path = checkpoint_dir / "portable_config.yaml"
                OmegaConf.save(portable_cfg, portable_path)
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

        checkpoint_dir = Path(checkpoint_location).absolute()
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
