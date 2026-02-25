"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from fairchem.core.components.calculate import (
    MDRunner,
    ParquetTrajectoryWriter,
    TrajectoryFrame,
)


@dataclass
class MockMetadata:
    results_dir: str
    checkpoint_dir: str = ""
    preemption_checkpoint_dir: str = ""
    config_path: str = ""
    array_job_num: int = 0


@dataclass
class MockScheduler:
    num_array_jobs: int = 1


@dataclass
class MockJobConfig:
    metadata: MockMetadata
    scheduler: MockScheduler


def _create_mock_job_config(
    results_dir: str,
    checkpoint_dir: str = "",
    preemption_checkpoint_dir: str = "",
) -> MockJobConfig:
    return MockJobConfig(
        metadata=MockMetadata(
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            preemption_checkpoint_dir=preemption_checkpoint_dir or checkpoint_dir,
        ),
        scheduler=MockScheduler(num_array_jobs=1),
    )


@pytest.fixture()
def cu_atoms():
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    np.random.seed(42)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    return atoms


@pytest.fixture()
def results_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMDRunner:
    def test_md_correctness_vs_ase(self, cu_atoms, results_dir):
        """
        Verify MDRunner produces identical trajectories to plain ASE.
        """
        atoms_mdrunner = cu_atoms.copy()
        atoms_ase = cu_atoms.copy()

        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_mdrunner, temperature_K=300)
        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_ase, temperature_K=300)

        steps, interval = 20, 5

        mdrunner_dir = results_dir / "mdrunner"
        mdrunner_dir.mkdir()
        runner = MDRunner(
            calculator=EMT(),
            atoms=atoms_mdrunner,
            dynamics=VelocityVerlet,
            timestep_fs=1.0,
            steps=steps,
            trajectory_interval=interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 100},
        )
        runner._job_config = _create_mock_job_config(str(mdrunner_dir))
        results = runner.calculate(job_num=0, num_jobs=1)

        # Reference ASE run
        ase_traj_file = results_dir / "ase_traj.traj"
        atoms_ase.calc = EMT()
        dyn_ase = VelocityVerlet(atoms_ase, timestep=1.0 * units.fs)
        traj_ase = Trajectory(str(ase_traj_file), "w", atoms_ase)
        dyn_ase.attach(traj_ase.write, interval=interval)
        dyn_ase.run(steps)
        traj_ase.close()

        traj_df = pd.read_parquet(results["trajectory_file"])
        ase_frames = Trajectory(str(ase_traj_file), "r")
        assert len(traj_df) == len(ase_frames)

        for i, ase_atoms in enumerate(ase_frames):
            row = traj_df.iloc[i]
            npt.assert_allclose(
                np.vstack(row["positions"]), ase_atoms.get_positions(), atol=1e-10
            )
            npt.assert_allclose(
                np.vstack(row["velocities"]), ase_atoms.get_velocities(), atol=1e-10
            )
            npt.assert_allclose(
                row["energy"], ase_atoms.get_potential_energy(), atol=1e-10
            )

    def test_checkpoint_resume(self, cu_atoms, results_dir):
        """
        Interrupt at a non-aligned step, checkpoint, resume, and verify
        trajectory alignment across runs.
        """
        results_dir1 = results_dir / "results1"
        results_dir2 = results_dir / "results2"
        checkpoint_dir = results_dir / "checkpoint"
        results_dir1.mkdir()
        results_dir2.mkdir()

        trajectory_interval = 10
        interrupt_at_step = 36
        total_steps = 100

        dynamics = partial(NoseHooverChainNVT, temperature_K=300.0)

        runner1 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            timestep_fs=1.0,
            tdamp_fs=25.0,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner1._job_config = _create_mock_job_config(str(results_dir1))

        class SimulatedInterrupt(Exception):
            pass

        def interrupt_callback():
            if runner1._dyn.get_number_of_steps() >= interrupt_at_step:
                raise SimulatedInterrupt

        # Run 1: manually drive dynamics so we can attach an interrupt
        try:
            runner1._atoms.calc = runner1.calculator
            runner1._dyn = NoseHooverChainNVT(
                atoms=runner1._atoms,
                timestep=1.0 * units.fs,
                temperature_K=300.0,
                tdamp=25.0 * units.fs,
            )
            parquet_file1 = results_dir1 / "trajectory_1-0.parquet"
            runner1._trajectory_writer = ParquetTrajectoryWriter(
                parquet_file1, flush_interval=1000
            )

            def collect_frame():
                step = runner1._dyn.get_number_of_steps()
                if step % trajectory_interval == 0:
                    frame = TrajectoryFrame.from_atoms(
                        runner1._atoms, step=step, time=runner1._dyn.get_time()
                    )
                    runner1._trajectory_writer.append(frame)

            runner1._dyn.attach(collect_frame, interval=1)
            runner1._dyn.attach(interrupt_callback, interval=1)
            runner1._dyn.run(total_steps)
        except SimulatedInterrupt:
            final_positions = runner1._atoms.get_positions().copy()
            final_velocities = runner1._atoms.get_velocities().copy()
            runner1.save_state(str(checkpoint_dir), is_preemption=True)

        df1 = pd.read_parquet(parquet_file1)
        steps1 = list(df1["step"])
        assert steps1 == [0, 10, 20, 30]

        # Verify checkpoint files
        assert (checkpoint_dir / "checkpoint.xyz").exists()
        assert (checkpoint_dir / "thermostat_state.json").exists()
        checkpoint_atoms = ase.io.read(str(checkpoint_dir / "checkpoint.xyz"))
        assert checkpoint_atoms.info["md_step"] == interrupt_at_step

        # Run 2: resume from checkpoint
        runner2 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            timestep_fs=1.0,
            tdamp_fs=25.0,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner2._job_config = _create_mock_job_config(str(results_dir2))
        runner2.load_state(str(checkpoint_dir))

        assert runner2._start_step == interrupt_at_step
        npt.assert_allclose(runner2._atoms.get_positions(), final_positions, atol=1e-8)
        npt.assert_allclose(
            runner2._atoms.get_velocities(), final_velocities, atol=1e-8
        )

        results2 = runner2.calculate(job_num=0, num_jobs=1)
        df2 = pd.read_parquet(results2["trajectory_file"])
        steps2 = list(df2["step"])

        all_steps = sorted(steps1 + steps2)
        expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert all_steps == expected, f"Expected {expected}, got {all_steps}"

    def test_stopfair_graceful_stop(self, cu_atoms, results_dir):
        """
        Verify STOPFAIR triggers graceful stop, saves state, and deletes
        the sentinel file.
        """
        md_results_dir = results_dir / "results"
        checkpoint_dir = results_dir / "checkpoints"
        md_results_dir.mkdir()
        checkpoint_dir.mkdir()

        dynamics = partial(NoseHooverChainNVT, temperature_K=300.0)

        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            timestep_fs=1.0,
            tdamp_fs=25.0,
            steps=100,
            trajectory_interval=10,
            heartbeat_interval=20,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner._job_config = _create_mock_job_config(
            str(md_results_dir), checkpoint_dir=str(checkpoint_dir)
        )

        stopfair_path = checkpoint_dir.parent / "STOPFAIR"
        stopfair_path.write_text("")

        results = runner.calculate(job_num=0, num_jobs=1)

        assert results["stopped_by_stopfair"] is True
        assert (checkpoint_dir / "checkpoint.xyz").exists()
        assert (checkpoint_dir / "md_state.json").exists()
        assert not stopfair_path.exists()

        with open(checkpoint_dir / "md_state.json") as f:
            md_state = json.load(f)
        assert md_state["current_step"] == 20

        traj_df = pd.read_parquet(results["trajectory_file"])
        assert list(traj_df["step"]) == [0, 10, 20]
