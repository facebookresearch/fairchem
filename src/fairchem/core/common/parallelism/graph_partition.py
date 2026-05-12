"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from enum import Enum

import torch


class PartitionStrategy(Enum):
    """
    Strategy for partitioning atoms across GP ranks.

    INDEX_SPLIT: Simple contiguous index split (existing behavior).
    SPATIAL: Spatial domain decomposition using fast k-means.
    """

    INDEX_SPLIT = "index_split"
    SPATIAL = "spatial"


def _expand_bits_10(v: torch.Tensor) -> torch.Tensor:
    """
    Expand a 10-bit integer so each bit is spaced by 2 zero bits.

    Maps bit at position i to position 3*i, producing a 30-bit
    output suitable for interleaving with two other axes to form
    a Morton Z-order code.

    Args:
        v: Integer tensor with values in [0, 1023].

    Returns:
        Tensor with bits expanded (each input bit at position 3*i).
    """
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def partition_atoms_spatial(
    pos: torch.Tensor,
    num_ranks: int,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    Spatial partitioning via Morton Z-order curve on GPU.

    Computes a 30-bit Morton code per atom by interleaving 10 bits
    from each spatial axis. Sorting by Morton code groups spatially
    nearby atoms together, minimizing boundary edges. Atoms are
    then split into num_ranks equal chunks in sorted order.

    Runs entirely on GPU with zero CPU transfers or sync points
    (unlike recursive coordinate bisection which requires CPU
    round-trips). The Morton curve provides O(N^{2/3}) surface
    fraction per partition, similar to recursive bisection.

    Args:
        pos: Atom positions, shape (N, 3).
        num_ranks: Number of partitions (GP world size).
        num_iters: Unused (kept for API compatibility).

    Returns:
        rank_assignments: Tensor of shape (N,) with rank index
            for each atom, on the same device as pos.
    """
    N = pos.shape[0]
    device = pos.device

    if num_ranks == 1:
        return torch.zeros(N, dtype=torch.long, device=device)

    if num_ranks >= N:
        return torch.arange(N, dtype=torch.long, device=device)

    # Normalize positions to [0, 1023] using a SINGLE global scale
    # factor (the largest bounding-box extent).  Per-dimension
    # normalization would amplify noise in short dimensions, breaking
    # Morton locality (e.g. a 100-unit x-gap becomes indistinguishable
    # from a 2-unit y-gap after independent rescaling).
    min_pos = pos.min(0)[0]
    extent = (pos.max(0)[0] - min_pos).max().clamp(min=1e-8)
    norm = ((pos - min_pos) / extent * 1023).long().clamp(0, 1023)

    # 30-bit Morton Z-order code: interleave x, y, z bits
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]
    morton = _expand_bits_10(x) | (_expand_bits_10(y) << 1) | (_expand_bits_10(z) << 2)

    # Sort by Morton code and assign to ranks in balanced chunks.
    # Use ``i * P // N`` mapping (not ``i // ceil(N/P)``) to ensure
    # EVERY rank receives at least ``floor(N/P)`` atoms.  The ceil-based
    # formula leaves trailing ranks empty when N is not a multiple of P
    # (e.g. 1000 atoms / 64 ranks → rank 63 gets 0 atoms, causing a
    # hang in collective communication).
    _, sorted_indices = morton.sort()
    assignments = torch.empty(N, dtype=torch.long, device=device)
    assignments[sorted_indices] = torch.arange(N, device=device) * num_ranks // N

    return assignments


def partition_atoms_index_split(
    total_atoms: int,
    num_ranks: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Simple contiguous index split (matches existing fairchem behavior).

    Args:
        total_atoms: Total number of atoms.
        num_ranks: Number of GP ranks.
        device: Device for the output tensor.

    Returns:
        rank_assignments: Tensor of shape (total_atoms,) with rank
            for each atom.
    """
    assignments = torch.empty(total_atoms, dtype=torch.long, device=device)
    partitions = torch.tensor_split(torch.arange(total_atoms, device=device), num_ranks)
    for rank, partition in enumerate(partitions):
        assignments[partition] = rank
    return assignments
