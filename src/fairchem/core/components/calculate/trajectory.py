"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class TrajectoryFrame:
    """Single frame of MD trajectory data."""

    step: int
    time: float
    atomic_numbers: np.ndarray  # (N,)
    positions: np.ndarray  # (N, 3)
    velocities: np.ndarray  # (N, 3)
    cell: np.ndarray  # (3, 3)
    pbc: np.ndarray  # (3,) bool
    energy: float
    forces: np.ndarray  # (N, 3)
    stress: np.ndarray | None = None  # (6,) Voigt notation
    sid: str | int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Parquet serialization."""
        return {
            "step": self.step,
            "time": self.time,
            "natoms": len(self.positions),
            "atomic_numbers": self.atomic_numbers.tolist(),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "cell": self.cell.tolist(),
            "pbc": self.pbc.tolist(),
            "energy": self.energy,
            "forces": self.forces.tolist(),
            "stress": self.stress.tolist() if self.stress is not None else None,
            "sid": self.sid,
        }

    @classmethod
    def from_atoms(cls, atoms: Atoms, step: int, time: float) -> TrajectoryFrame:
        """
        Create a TrajectoryFrame from an ASE Atoms object.

        Args:
            atoms: ASE Atoms object with calculator attached
            step: Current MD step number
            time: Current simulation time

        Returns:
            TrajectoryFrame populated with atoms data
        """
        try:
            stress = atoms.get_stress()
        except Exception:
            stress = None

        return cls(
            step=step,
            time=time,
            atomic_numbers=atoms.get_atomic_numbers().copy(),
            positions=atoms.get_positions().copy(),
            velocities=atoms.get_velocities().copy(),
            cell=atoms.get_cell()[:].copy(),
            pbc=np.array(atoms.get_pbc()),
            energy=atoms.get_potential_energy(),
            forces=atoms.get_forces().copy(),
            stress=stress,
            sid=atoms.info.get("sid"),
        )


class TrajectoryWriter(ABC):
    """
    Abstract base class for trajectory writers.

    Trajectory writers are responsible for saving MD simulation frames
    to disk in various formats (parquet, ASE trajectory, etc.).
    """

    def __init__(self, path: Path | str):
        """
        Initialize the trajectory writer.

        Args:
            path: Path to the output trajectory file
        """
        self.path = Path(path)
        self.total_frames = 0

    @abstractmethod
    def append(self, frame: TrajectoryFrame) -> None:
        """
        Append a frame to the trajectory.

        Args:
            frame: TrajectoryFrame to append
        """

    @abstractmethod
    def close(self) -> None:
        """Finalize and close the trajectory file."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
