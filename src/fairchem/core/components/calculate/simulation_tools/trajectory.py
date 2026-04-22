"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ase.units
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from ase import Atoms

from ase.calculators.calculator import PropertyNotImplementedError


@dataclass
class TrajectoryFrame:
    """
    Single frame of simulation trajectory data.

    Note:
        Currently only step, atomic_numbers, positions, cell, and pbc are
        required. Additional fields may be made optional in the future as
        new simulation types are supported.
    """

    step: int
    atomic_numbers: np.ndarray  # (N,)
    positions: np.ndarray  # (N, 3)
    cell: np.ndarray  # (3, 3)
    pbc: np.ndarray  # (3,) bool
    time: float | None = None  # femtoseconds
    velocities: np.ndarray | None = None  # (N, 3)
    energy: float | None = None
    forces: np.ndarray | None = None  # (N, 3)
    stress: np.ndarray | None = None  # (6,) Voigt notation
    temperature: float | None = None  # Kelvin
    kinetic_energy: float | None = None  # eV
    pressure: float | None = None  # bar
    sid: str | int | None = None
    tags: np.ndarray | None = None  # (N,) int
    fixed: np.ndarray | None = None  # (N,) bool
    charge: int | None = None
    spin: int | None = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary for Parquet serialization.
        """
        d = {
            "step": self.step,
            "natoms": len(self.positions),
            "atomic_numbers": self.atomic_numbers.tolist(),
            "positions": self.positions.tolist(),
            "cell": self.cell.tolist(),
            "pbc": self.pbc.tolist(),
            "sid": self.sid,
        }
        if self.time is not None:
            d["time"] = self.time
        if self.velocities is not None:
            d["velocities"] = self.velocities.tolist()
        if self.energy is not None:
            d["energy"] = self.energy
        if self.forces is not None:
            d["forces"] = self.forces.tolist()
        if self.stress is not None:
            d["stress"] = self.stress.tolist()
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.kinetic_energy is not None:
            d["kinetic_energy"] = self.kinetic_energy
        if self.pressure is not None:
            d["pressure"] = self.pressure
        if self.tags is not None:
            d["tags"] = self.tags.tolist()
        if self.fixed is not None:
            d["fixed"] = self.fixed.tolist()
        if self.charge is not None:
            d["charge"] = self.charge
        if self.spin is not None:
            d["spin"] = self.spin
        return d

    @classmethod
    def from_atoms(
        cls, atoms: Atoms, step: int, time: float | None = None
    ) -> TrajectoryFrame:
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
        except (PropertyNotImplementedError, RuntimeError):
            stress = None

        try:
            energy = atoms.get_potential_energy()
        except (PropertyNotImplementedError, RuntimeError):
            energy = None

        try:
            forces = atoms.get_forces().copy()
        except (PropertyNotImplementedError, RuntimeError):
            forces = None

        try:
            velocities = atoms.get_velocities().copy()
        except (PropertyNotImplementedError, RuntimeError):
            velocities = None

        try:
            temperature = atoms.get_temperature()
        except Exception:
            temperature = None

        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = None

        # Pressure from stress: P = -trace(stress)/3, converted to bar
        if stress is not None:
            pressure = -stress[:3].mean() / ase.units.bar
        else:
            pressure = None

        # Tags: only store if any are nonzero (all zeros is ASE default)
        raw_tags = atoms.get_tags()
        tags = raw_tags.copy() if np.any(raw_tags) else None

        # Fixed atoms: build bool mask from FixAtoms constraints
        fixed = None
        if atoms.constraints:
            from ase.constraints import FixAtoms

            mask = np.zeros(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    mask[constraint.index] = True
            if np.any(mask):
                fixed = mask

        charge = atoms.info.get("charge", None)
        spin = atoms.info.get("spin", None)

        return cls(
            step=step,
            time=time,
            atomic_numbers=atoms.get_atomic_numbers().copy(),
            positions=atoms.get_positions().copy(),
            cell=atoms.get_cell()[:].copy(),
            pbc=np.array(atoms.get_pbc()),
            velocities=velocities,
            energy=energy,
            forces=forces,
            stress=stress,
            temperature=temperature,
            kinetic_energy=kinetic_energy,
            pressure=pressure,
            sid=atoms.info.get("sid"),
            tags=tags,
            fixed=fixed,
            charge=charge,
            spin=spin,
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
