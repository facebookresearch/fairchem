"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from fairchem.core.components.calculate.trajectory import (
    TrajectoryFrame,
    TrajectoryWriter,
)

if TYPE_CHECKING:
    from pathlib import Path


class ParquetTrajectoryWriter(TrajectoryWriter):
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
        super().__init__(path)
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
        """Write buffered frames as a new row group."""
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
        """Flush remaining buffer and finalize."""
        self.flush()
        if self._writer is not None:
            self._writer.close()
