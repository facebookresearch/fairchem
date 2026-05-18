"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.common.parallelism.graph_partition import (
    PartitionStrategy,
    partition_atoms_index_split,
    partition_atoms_spatial,
)

__all__ = [
    "PartitionStrategy",
    "partition_atoms_index_split",
    "partition_atoms_spatial",
]
