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
    ASETrajectoryWriter,
    MDRunner,
    ParquetTrajectoryWriter,
    TrajectoryFrame,
)


@dataclass
class MockMetadata:
    results_dir: str
    checkpoint_dir: str = ""
    array_job_num: int = 0


@dataclass
class MockScheduler:
    num_array_jobs: int = 1


@dataclass
class MockJobConfig:
    metadata: MockMetadata
    scheduler: MockScheduler


def _create_mock_job_config(
    results_dir: str, checkpoint_dir: str = ""
) -> MockJobConfig:
    return MockJobConfig(
        metadata=MockMetadata(results_dir=results_dir, checkpoint_dir=checkpoint_dir),
        scheduler=MockScheduler(num_array_jobs=1),
    )


@pytest.fixture()
def cu_atoms():
    """Create a Cu bulk structure with thermal velocities."""
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    np.random.seed(42)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    return atoms


@pytest.fixture()
def results_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTrajectoryFrame:
    """Tests for TrajectoryFrame dataclass."""

    def test_to_dict_serialization(self):
        """Test TrajectoryFrame serialization to dict with nested lists."""
        frame = TrajectoryFrame(
            step=10,
            time=1.0,
            atomic_numbers=np.array([29, 29], dtype=np.int64),
            positions=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            velocities=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64),
            cell=np.eye(3) * 10,
            pbc=np.array([True, True, True]),
            energy=-5.0,
            forces=np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float64),
            stress=np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        )

        d = frame.to_dict()

        assert d["step"] == 10
        assert d["time"] == 1.0
        assert d["energy"] == -5.0
        assert d["natoms"] == 2
        assert all(
            isinstance(d[k], list)
            for k in ["positions", "velocities", "forces", "cell"]
        )

        # Verify array reconstruction from lists
        positions = np.array(d["positions"])
        npt.assert_allclose(positions, frame.positions)

    def test_from_atoms(self):
        """Test TrajectoryFrame.from_atoms class method."""
        atoms = bulk("Cu")
        atoms.calc = EMT()
        atoms.set_velocities(np.ones((len(atoms), 3)) * 0.01)

        frame = TrajectoryFrame.from_atoms(atoms, step=5, time=2.5)

        assert frame.step == 5
        assert frame.time == 2.5
        npt.assert_array_equal(frame.atomic_numbers, atoms.get_atomic_numbers())
        npt.assert_allclose(frame.positions, atoms.get_positions())
        npt.assert_allclose(frame.velocities, atoms.get_velocities())
        npt.assert_allclose(frame.energy, atoms.get_potential_energy())


class TestParquetTrajectoryWriter:
    """Tests for ParquetTrajectoryWriter."""

    def test_buffering_and_flush(self, results_dir):
        """Test ParquetTrajectoryWriter buffering and flushing behavior."""
        path = results_dir / "test.parquet"
        writer = ParquetTrajectoryWriter(path, flush_interval=3)

        def make_frame(step):
            return TrajectoryFrame(
                step=step,
                time=float(step),
                atomic_numbers=np.array([29, 29], dtype=np.int64),
                positions=np.zeros((2, 3)),
                velocities=np.zeros((2, 3)),
                cell=np.eye(3),
                pbc=np.array([True, True, True]),
                energy=float(step),
                forces=np.zeros((2, 3)),
            )

        # Add 2 frames - should not flush yet
        for i in range(2):
            writer.append(make_frame(i))
        assert not path.exists()
        assert len(writer.buffer) == 2

        # Add 1 more - triggers flush
        writer.append(make_frame(2))
        assert path.exists()
        assert writer.total_frames == 3

        # Add 1 more and close
        writer.append(make_frame(3))
        writer.close()

        traj_df = pd.read_parquet(path)
        assert len(traj_df) == 4
        assert list(traj_df["step"]) == [0, 1, 2, 3]


class TestASETrajectoryWriter:
    """Tests for ASETrajectoryWriter."""

    def test_write_and_read(self, results_dir):
        """Test ASETrajectoryWriter writes valid ASE trajectory files."""
        path = results_dir / "test.traj"

        frames = []
        for step in range(5):
            frame = TrajectoryFrame(
                step=step,
                time=float(step) * 0.5,
                atomic_numbers=np.array([29, 29], dtype=np.int64),
                positions=np.array(
                    [[0, 0, 0], [1 + step * 0.1, 1, 1]], dtype=np.float64
                ),
                velocities=np.array(
                    [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64
                ),
                cell=np.eye(3) * 10,
                pbc=np.array([True, True, True]),
                energy=-5.0 + step,
                forces=np.array(
                    [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float64
                ),
            )
            frames.append(frame)

        writer = ASETrajectoryWriter(path)
        for frame in frames:
            writer.append(frame)
        writer.close()

        assert path.exists()
        assert writer.total_frames == 5

        # Read back and verify
        traj = Trajectory(str(path), "r")
        assert len(traj) == 5

        for i, atoms in enumerate(traj):
            npt.assert_allclose(atoms.get_positions(), frames[i].positions)
            assert atoms.info["step"] == i
            npt.assert_allclose(atoms.info["energy"], frames[i].energy)


class TestMDRunner:
    """Tests for MDRunner class."""

    def test_md_correctness_vs_ase(self, cu_atoms, results_dir):
        """Compare MDRunner output with ASE Trajectory to verify correctness."""
        atoms_mdrunner = cu_atoms.copy()
        atoms_ase = cu_atoms.copy()

        # Reset velocities identically
        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_mdrunner, temperature_K=300)
        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_ase, temperature_K=300)

        steps, interval = 20, 5

        # MDRunner
        mdrunner_dir = results_dir / "mdrunner"
        mdrunner_dir.mkdir()
        runner = MDRunner(
            calculator=EMT(),
            atoms=atoms_mdrunner,
            dynamics=partial(VelocityVerlet, timestep=1.0 * units.fs),
            steps=steps,
            trajectory_interval=interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 100},
        )
        runner._job_config = _create_mock_job_config(str(mdrunner_dir))
        results = runner.calculate(job_num=0, num_jobs=1)

        # Traditional ASE
        ase_traj_file = results_dir / "ase_traj.traj"
        atoms_ase.calc = EMT()
        dyn_ase = VelocityVerlet(atoms_ase, timestep=1.0 * units.fs)
        traj_ase = Trajectory(str(ase_traj_file), "w", atoms_ase)
        dyn_ase.attach(traj_ase.write, interval=interval)
        dyn_ase.run(steps)
        traj_ase.close()

        # Compare
        traj_df = pd.read_parquet(results["trajectory_file"])
        ase_frames = Trajectory(str(ase_traj_file), "r")
        assert len(traj_df) == len(ase_frames), "Frame count mismatch"

        for i, ase_atoms in enumerate(ase_frames):
            row = traj_df.iloc[i]
            parquet_pos = np.vstack(row["positions"])
            parquet_vel = np.vstack(row["velocities"])
            parquet_forces = np.vstack(row["forces"])

            npt.assert_allclose(parquet_pos, ase_atoms.get_positions(), atol=1e-10)
            npt.assert_allclose(parquet_vel, ase_atoms.get_velocities(), atol=1e-10)
            npt.assert_allclose(parquet_forces, ase_atoms.get_forces(), atol=1e-10)
            npt.assert_allclose(
                row["energy"], ase_atoms.get_potential_energy(), atol=1e-10
            )

        # Verify write_results works
        runner.write_results(results, str(mdrunner_dir), job_num=0, num_jobs=1)
        assert (mdrunner_dir / "metadata_1-0.json").exists()

    def test_md_with_ase_trajectory_writer(self, cu_atoms, results_dir):
        """Test MDRunner with ASETrajectoryWriter."""
        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=partial(VelocityVerlet, timestep=1.0 * units.fs),
            steps=10,
            trajectory_interval=2,
            log_interval=5,
            trajectory_writer=ASETrajectoryWriter,
        )
        runner._job_config = _create_mock_job_config(str(results_dir))
        results = runner.calculate(job_num=0, num_jobs=1)

        trajectory_file = Path(results["trajectory_file"])
        assert trajectory_file.exists()
        assert trajectory_file.suffix == ".traj"

        traj = Trajectory(str(trajectory_file), "r")
        assert len(traj) == 6  # steps 0, 2, 4, 6, 8, 10

        steps = [atoms.info["step"] for atoms in traj]
        assert steps == [0, 2, 4, 6, 8, 10]

    def test_checkpoint_resume_and_trajectory_alignment(self, cu_atoms, results_dir):
        """
        Comprehensive test for checkpoint/resume functionality.

        Tests:
        1. Mid-run interrupt at non-aligned step (preemption simulation)
        2. Checkpoint file creation (xyz, thermostat state, metadata)
        3. Thermostat state preservation
        4. Position/velocity restoration accuracy
        5. Trajectory step alignment after resume
        6. Full simulation completion
        """
        results_dir1 = results_dir / "results1"
        results_dir2 = results_dir / "results2"
        checkpoint_dir = results_dir / "checkpoint"
        results_dir1.mkdir()
        results_dir2.mkdir()

        trajectory_interval = 10
        interrupt_at_step = 36
        total_steps = 100

        dynamics = partial(
            NoseHooverChainNVT,
            timestep=1.0 * units.fs,
            temperature_K=300.0,
            tdamp=25 * units.fs,
        )

        # Run 1: Run until interrupted at step 36
        runner1 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
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

        try:
            runner1._atoms.calc = runner1.calculator
            runner1._dyn = runner1.dynamics(atoms=runner1._atoms)

            parquet_file1 = results_dir1 / "trajectory_1-0.parquet"
            runner1._trajectory_writer = ParquetTrajectoryWriter(
                parquet_file1, flush_interval=1000
            )

            def collect_frame():
                global_step = runner1._dyn.get_number_of_steps() + runner1._start_step
                if global_step % trajectory_interval == 0:
                    frame = TrajectoryFrame.from_atoms(
                        runner1._atoms,
                        step=global_step,
                        time=runner1._dyn.get_time(),
                    )
                    runner1._trajectory_writer.append(frame)

            runner1._dyn.attach(collect_frame, interval=1)
            runner1._dyn.attach(interrupt_callback, interval=1)
            runner1._dyn.run(total_steps)
        except SimulatedInterrupt:
            eta_before = runner1._dyn._thermostat._eta.copy()
            p_eta_before = runner1._dyn._thermostat._p_eta.copy()
            final_positions = runner1._atoms.get_positions().copy()
            final_velocities = runner1._atoms.get_velocities().copy()

            saved = runner1.save_state(str(checkpoint_dir), is_preemption=True)
            assert saved, "save_state failed"

        df1 = pd.read_parquet(parquet_file1)
        steps1 = list(df1["step"])
        assert steps1 == [
            0,
            10,
            20,
            30,
        ], f"Run1 steps should be [0,10,20,30], got {steps1}"

        assert (checkpoint_dir / "checkpoint.xyz").exists()
        assert (checkpoint_dir / "thermostat_state.json").exists()
        assert (checkpoint_dir / "md_state.json").exists()

        # Verify md_step is embedded in the checkpoint atoms
        checkpoint_atoms = ase.io.read(str(checkpoint_dir / "checkpoint.xyz"))
        assert checkpoint_atoms.info["md_step"] == interrupt_at_step

        with open(checkpoint_dir / "thermostat_state.json") as f:
            saved_state = json.load(f)
        assert saved_state["class_name"] == "NoseHooverChainNVT"
        npt.assert_allclose(saved_state["eta"], eta_before)
        npt.assert_allclose(saved_state["p_eta"], p_eta_before)

        with open(checkpoint_dir / "md_state.json") as f:
            md_state = json.load(f)
        assert md_state["current_step"] == interrupt_at_step

        # Run 2: Resume from checkpoint
        runner2 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
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

        expected_steps2 = [40, 50, 60, 70, 80, 90, 100]
        assert (
            steps2 == expected_steps2
        ), f"Run2 steps should be {expected_steps2}, got {steps2}"

        all_steps = sorted(steps1 + steps2)
        expected_all = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert (
            all_steps == expected_all
        ), f"Combined trajectory should be {expected_all}, got {all_steps}"

        # Run 3: Full simulation from scratch for comparison
        results_dir3 = results_dir / "results3"
        results_dir3.mkdir()

        runner3 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner3._job_config = _create_mock_job_config(str(results_dir3))
        results3 = runner3.calculate(job_num=0, num_jobs=1)

        df3 = pd.read_parquet(results3["trajectory_file"])
        steps3 = list(df3["step"])
        assert (
            steps3 == expected_all
        ), f"Run3 steps should be {expected_all}, got {steps3}"

        # Compare run1 trajectory with run3 (should match exactly)
        for step in steps1:
            row1 = df1[df1["step"] == step].iloc[0]
            row3 = df3[df3["step"] == step].iloc[0]

            pos1 = np.vstack(row1["positions"])
            pos3 = np.vstack(row3["positions"])
            npt.assert_allclose(
                pos1, pos3, atol=1e-10, err_msg=f"Position mismatch at step {step}"
            )

            vel1 = np.vstack(row1["velocities"])
            vel3 = np.vstack(row3["velocities"])
            npt.assert_allclose(
                vel1, vel3, atol=1e-10, err_msg=f"Velocity mismatch at step {step}"
            )

            npt.assert_allclose(
                row1["energy"],
                row3["energy"],
                atol=1e-10,
                err_msg=f"Energy mismatch at step {step}",
            )

    def test_stopcar_graceful_stop(self, cu_atoms, results_dir):
        """
        Test that STOPCAR file triggers graceful stop with state saved.

        Verifies:
        1. MD stops when STOPCAR is detected at checkpoint_interval
        2. State is saved (checkpoint.xyz, md_state.json, thermostat_state.json)
        3. Trajectory is flushed and readable
        4. Result dict includes stopped_by_stopcar=True
        5. Simulation can be resumed from the saved checkpoint
        """
        run_dir = results_dir / "run_dir"
        md_results_dir = results_dir / "results"
        checkpoint_dir = results_dir / "checkpoints"
        run_dir.mkdir()
        md_results_dir.mkdir()
        checkpoint_dir.mkdir()

        total_steps = 100
        checkpoint_interval = 20
        trajectory_interval = 10

        dynamics = partial(
            NoseHooverChainNVT,
            timestep=1.0 * units.fs,
            temperature_K=300.0,
            tdamp=25 * units.fs,
        )

        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            checkpoint_interval=checkpoint_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner._job_config = _create_mock_job_config(
            str(md_results_dir),
            checkpoint_dir=str(checkpoint_dir),
        )

        # Write STOPCAR in checkpoint_dir - should stop at first checkpoint
        stopcar_path = checkpoint_dir / "STOPCAR"
        stopcar_path.write_text("")

        results = runner.calculate(job_num=0, num_jobs=1)

        # Should have stopped early
        assert results["stopped_by_stopcar"] is True

        # Checkpoint files should exist
        assert (checkpoint_dir / "checkpoint.xyz").exists()
        assert (checkpoint_dir / "md_state.json").exists()
        assert (checkpoint_dir / "thermostat_state.json").exists()

        # Verify checkpoint step matches checkpoint_interval
        with open(checkpoint_dir / "md_state.json") as f:
            md_state = json.load(f)
        assert md_state["current_step"] == checkpoint_interval

        # Trajectory should be readable with frames up to the stop point
        traj_df = pd.read_parquet(results["trajectory_file"])
        steps = list(traj_df["step"])
        expected_steps = [0, 10, 20]
        assert steps == expected_steps, f"Expected {expected_steps}, got {steps}"

        # Resume from checkpoint after removing STOPCAR
        stopcar_path.unlink()
        results_dir2 = results_dir / "results2"
        results_dir2.mkdir()

        runner2 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=dynamics,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            checkpoint_interval=checkpoint_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner2._job_config = _create_mock_job_config(
            str(results_dir2),
            checkpoint_dir=str(checkpoint_dir),
        )
        runner2.load_state(str(checkpoint_dir))
        assert runner2._start_step == checkpoint_interval

        results2 = runner2.calculate(job_num=0, num_jobs=1)
        assert results2["stopped_by_stopcar"] is False

        # Resumed trajectory continues from checkpoint step (which is re-captured)
        traj_df2 = pd.read_parquet(results2["trajectory_file"])
        steps2 = list(traj_df2["step"])
        expected_steps2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert steps2 == expected_steps2, f"Expected {expected_steps2}, got {steps2}"

    def test_no_stopcar_runs_to_completion(self, cu_atoms, results_dir):
        """
        Test that without STOPCAR, checkpoint_interval runs to completion normally.
        """
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir()

        total_steps = 50
        checkpoint_interval = 20

        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            dynamics=partial(VelocityVerlet, timestep=1.0 * units.fs),
            steps=total_steps,
            trajectory_interval=10,
            checkpoint_interval=checkpoint_interval,
            log_interval=10,
            trajectory_writer_kwargs={"flush_interval": 1000},
        )
        runner._job_config = _create_mock_job_config(
            str(results_dir),
            checkpoint_dir=str(checkpoint_dir),
        )
        results = runner.calculate(job_num=0, num_jobs=1)

        assert results["stopped_by_stopcar"] is False

        traj_df = pd.read_parquet(results["trajectory_file"])
        steps = list(traj_df["step"])
        assert steps == [0, 10, 20, 30, 40, 50]
