"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.common.parallelism.graph_partition import (
    PartitionStrategy,
    partition_atoms_index_split,
    partition_atoms_spatial,
)


class TestPartitionStrategy:
    """
    Tests for the PartitionStrategy enum.
    """

    def test_enum_values(self):
        assert PartitionStrategy.INDEX_SPLIT.value == "index_split"
        assert PartitionStrategy.SPATIAL.value == "spatial"

    def test_from_string(self):
        assert PartitionStrategy("index_split") is PartitionStrategy.INDEX_SPLIT
        assert PartitionStrategy("spatial") is PartitionStrategy.SPATIAL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            PartitionStrategy("random")


class TestPartitionAtomsIndexSplit:
    """
    Tests for partition_atoms_index_split.
    """

    def test_single_rank(self):
        assignments = partition_atoms_index_split(10, 1, torch.device("cpu"))
        assert assignments.shape == (10,)
        assert (assignments == 0).all()

    def test_even_split(self):
        assignments = partition_atoms_index_split(12, 3, torch.device("cpu"))
        assert assignments.shape == (12,)
        # Each rank should get 4 atoms
        for rank in range(3):
            assert (assignments == rank).sum() == 4

    def test_uneven_split(self):
        assignments = partition_atoms_index_split(10, 3, torch.device("cpu"))
        assert assignments.shape == (10,)
        # All ranks should be represented
        for rank in range(3):
            assert (assignments == rank).sum() >= 3

    def test_more_ranks_than_atoms(self):
        assignments = partition_atoms_index_split(3, 8, torch.device("cpu"))
        assert assignments.shape == (3,)
        # Each atom should be assigned to some rank
        assert assignments.max() < 8

    def test_contiguous_assignment(self):
        """
        Index-split should produce contiguous blocks: rank 0 gets
        indices [0..k), rank 1 gets [k..2k), etc.
        """
        assignments = partition_atoms_index_split(16, 4, torch.device("cpu"))
        for rank in range(4):
            indices = (assignments == rank).nonzero(as_tuple=True)[0]
            # Indices should be contiguous
            assert (indices[1:] - indices[:-1] == 1).all()


class TestPartitionAtomsSpatial:
    """
    Tests for partition_atoms_spatial (Morton Z-order curve).
    """

    def test_single_rank(self):
        pos = torch.randn(100, 3)
        assignments = partition_atoms_spatial(pos, 1)
        assert assignments.shape == (100,)
        assert (assignments == 0).all()

    def test_more_ranks_than_atoms(self):
        pos = torch.randn(3, 3)
        assignments = partition_atoms_spatial(pos, 10)
        assert assignments.shape == (3,)
        # Each atom gets a unique rank (0, 1, 2)
        assert assignments.unique().numel() == 3

    def test_balanced_output(self):
        """
        For N atoms and P ranks, each rank should get
        floor(N/P) or ceil(N/P) atoms.
        """
        pos = torch.randn(1000, 3)
        assignments = partition_atoms_spatial(pos, 8)
        counts = torch.bincount(assignments, minlength=8)
        assert counts.min() >= 124  # floor(1000/8) = 125, allow 1 off
        assert counts.max() <= 126

    def test_all_ranks_populated(self):
        """
        No rank should be empty (avoids NCCL deadlock).
        """
        pos = torch.randn(64, 3)
        assignments = partition_atoms_spatial(pos, 8)
        assert assignments.unique().numel() == 8

    def test_spatial_locality(self):
        """
        Atoms in the same spatial cluster should tend to be assigned
        to the same rank.
        """
        # Create 4 well-separated clusters
        cluster_centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [100.0, 100.0, 0.0],
            ]
        )
        pos = torch.cat(
            [center + torch.randn(25, 3) * 0.1 for center in cluster_centers],
            dim=0,
        )

        assignments = partition_atoms_spatial(pos, 4)

        # Each cluster should be dominated by a single rank
        for i in range(4):
            cluster_assignments = assignments[i * 25 : (i + 1) * 25]
            dominant_rank = cluster_assignments.mode()[0]
            dominant_fraction = (cluster_assignments == dominant_rank).float().mean()
            assert dominant_fraction > 0.8, (
                f"Cluster {i}: dominant rank {dominant_rank} "
                f"has only {dominant_fraction:.0%} of atoms"
            )

    def test_device_consistency(self):
        """
        Output device should match input device.
        """
        pos_cpu = torch.randn(50, 3)
        assignments = partition_atoms_spatial(pos_cpu, 4)
        assert assignments.device == pos_cpu.device

    @pytest.mark.gpu()
    def test_gpu(self):
        """
        Spatial partitioning should work on GPU.
        """
        pos = torch.randn(200, 3, device="cuda")
        assignments = partition_atoms_spatial(pos, 8)
        assert assignments.device == pos.device
        assert assignments.shape == (200,)
        assert assignments.unique().numel() == 8

    def test_deterministic(self):
        """
        Same input should produce same output.
        """
        pos = torch.randn(100, 3)
        a1 = partition_atoms_spatial(pos, 4)
        a2 = partition_atoms_spatial(pos, 4)
        assert torch.equal(a1, a2)

    def test_large_num_ranks(self):
        """
        Should handle 64+ ranks without any rank being empty.
        """
        pos = torch.randn(4000, 3)
        assignments = partition_atoms_spatial(pos, 64)
        counts = torch.bincount(assignments, minlength=64)
        assert (
            counts > 0
        ).all(), f"Empty ranks: {(counts == 0).nonzero(as_tuple=True)[0].tolist()}"

    def test_uniform_global_normalization(self):
        """
        Verify that positions are normalized using a single global
        scale (not per-dimension), preserving spatial structure.
        """
        # Positions stretched along x only
        pos = torch.zeros(100, 3)
        pos[:, 0] = torch.linspace(0, 1000, 100)
        pos[:, 1] = torch.randn(100) * 0.01
        pos[:, 2] = torch.randn(100) * 0.01

        assignments = partition_atoms_spatial(pos, 4)
        # Since spread is only in x, spatial partition should
        # split along x-axis predominantly
        # Rank 0 should contain leftmost atoms
        rank0_mean_x = pos[assignments == 0, 0].mean()
        rank3_mean_x = pos[assignments == 3, 0].mean()
        assert rank0_mean_x < rank3_mean_x
