"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class TrajectoryFrame:
    """
    Single frame of MD trajectory data.
    """

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
        """
        Convert to dictionary for Parquet serialization.
        """
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


class ParquetTrajectoryWriter:
    """
    Buffered writer for MD trajectory data in parquet format.

    Uses PyArrow's ParquetWriter for efficient incremental writes
    without read-modify-write overhead.
    """

    def __init__(self, path: Path | str, flush_interval: int = 1000):
        """
        Initialize the parquet trajectory writer.

        Args:
            path: Path to the output parquet file
            flush_interval: Number of frames to buffer before writing to disk
        """
        self.path = Path(path)
        self.total_frames = 0
        self.flush_interval = flush_interval
        self.buffer: list[dict] = []
        self._writer = None
        self._schema = None

    def append(self, frame: TrajectoryFrame) -> None:
        """
        Add frame to buffer, flush if interval reached.

        Args:
            frame: TrajectoryFrame to append
        """
        self.buffer.append(frame.to_dict())
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """
        Write buffered frames as a new row group.
        """
        if not self.buffer:
            return

        table = pa.Table.from_pydict(
            {k: [row[k] for row in self.buffer] for k in self.buffer[0]}
        )

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")

        self._writer.write_table(table)
        self.total_frames += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        """
        Flush remaining buffer and finalize.
        """
        self.flush()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
