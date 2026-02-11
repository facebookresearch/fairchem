"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from ase.io import Trajectory

from fairchem.core.components.calculate.trajectory import (
    TrajectoryFrame,
    TrajectoryWriter,
)

if TYPE_CHECKING:
    from pathlib import Path


class ASETrajectoryWriter(TrajectoryWriter):
    """
    Trajectory writer using ASE's native trajectory format.

    Wraps ASE's Trajectory class to conform to the TrajectoryWriter interface.
    The ASE trajectory format stores atoms objects with full information
    including positions, velocities, cell, and calculator results.
    """

    def __init__(self, path: Path | str, mode: str = "w"):
        """
        Initialize the ASE trajectory writer.

        Args:
            path: Path to the output trajectory file (.traj)
            mode: File mode - "w" for write, "a" for append
        """
        super().__init__(path)
        self._trajectory = Trajectory(str(self.path), mode=mode)

    def append(self, frame: TrajectoryFrame) -> None:
        """
        Append a frame to the ASE trajectory.

        Converts the TrajectoryFrame to an ASE Atoms object and writes it.

        Args:
            frame: TrajectoryFrame to append
        """
        atoms = Atoms(
            numbers=frame.atomic_numbers,
            positions=frame.positions,
            cell=frame.cell,
            pbc=frame.pbc,
        )
        atoms.set_velocities(frame.velocities)

        # Store results as SinglePointCalculator-like info
        atoms.info["step"] = frame.step
        atoms.info["time"] = frame.time
        atoms.info["energy"] = frame.energy

        # Store forces and stress in arrays for later retrieval
        atoms.arrays["forces"] = frame.forces
        if frame.stress is not None:
            atoms.info["stress"] = frame.stress

        self._trajectory.write(atoms, energy=frame.energy, forces=frame.forces)
        self.total_frames += 1

    def close(self) -> None:
        """Close the ASE trajectory file."""
        self._trajectory.close()
