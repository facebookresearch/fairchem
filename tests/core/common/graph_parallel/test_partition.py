"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.common.graph_parallel.partition import (
    PartitionStrategy,
    partition_atoms_by_position,
    partition_atoms_kmeans,
    partition_atoms_to_grid,
    partition_atoms_to_slices,
)


class TestPartitionAtomsToGrid:
    """Tests for partition_atoms_to_grid function."""

    def test_basic_partition(self):
        """Test basic partitioning of atoms into a 2x2x2 grid."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        k = 2
        cell_indices = partition_atoms_to_grid(coords, k)

        assert cell_indices.shape == (8,)
        # All cell indices should be in valid range [0, k^3 - 1]
        assert cell_indices.min() >= 0
        assert cell_indices.max() < k**3

    def test_fewer_atoms_than_cells(self):
        """Test when there are fewer atoms than grid cells (round-robin assignment)."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        k = 2  # 8 cells, but only 2 atoms
        cell_indices = partition_atoms_to_grid(coords, k)

        assert cell_indices.shape == (2,)
        # Should be round-robin: 0, 1
        assert cell_indices[0].item() == 0
        assert cell_indices[1].item() == 1

    def test_more_atoms_than_cells(self):
        """Test when there are more atoms than grid cells."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 1.0],
                [1.0, 1.5, 1.0],
                [1.0, 1.0, 1.5],
                [0.1, 0.1, 0.1],  # Extra atom in cell 0
                [1.9, 1.9, 1.9],  # Extra atom in cell 7
            ]
        )
        k = 2  # 8 cells, 10 atoms
        cell_indices = partition_atoms_to_grid(coords, k)

        assert cell_indices.shape == (10,)
        assert cell_indices.min() >= 0
        assert cell_indices.max() < k**3

    def test_single_cell(self):
        """Test partitioning into a single cell (k=1)."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        k = 1
        cell_indices = partition_atoms_to_grid(coords, k)

        # All atoms should go to cell 0
        assert (cell_indices == 0).all()

    def test_same_coordinates(self):
        """Test when all atoms have the same coordinates."""
        coords = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        k = 2
        cell_indices = partition_atoms_to_grid(coords, k)

        # Should still work without division by zero
        assert cell_indices.shape == (3,)


class TestPartitionAtomsToSlices:
    """Tests for partition_atoms_to_slices function."""

    def test_basic_slices_x_axis(self):
        """Test basic slicing along x-axis."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        K = 2
        slice_indices = partition_atoms_to_slices(coords, K, axis=0)

        assert slice_indices.shape == (4,)
        # First two atoms should be in slice 0, last two in slice 1
        assert slice_indices[0].item() == 0
        assert slice_indices[1].item() == 0
        assert slice_indices[2].item() == 1
        assert slice_indices[3].item() == 1

    def test_slices_y_axis(self):
        """Test slicing along y-axis."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        K = 2
        slice_indices = partition_atoms_to_slices(coords, K, axis=1)

        assert slice_indices.shape == (4,)
        assert slice_indices[0].item() == 0
        assert slice_indices[1].item() == 0
        assert slice_indices[2].item() == 1
        assert slice_indices[3].item() == 1

    def test_slices_z_axis(self):
        """Test slicing along z-axis."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ]
        )
        K = 4
        slice_indices = partition_atoms_to_slices(coords, K, axis=2)

        assert slice_indices.shape == (4,)
        # Each atom in its own slice
        assert slice_indices[0].item() == 0
        assert slice_indices[1].item() == 1
        assert slice_indices[2].item() == 2
        assert slice_indices[3].item() == 3

    def test_same_coordinate_on_axis(self):
        """Test when all atoms have the same coordinate on the slicing axis."""
        coords = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [0.0, 3.0, 4.0],
                [0.0, 5.0, 6.0],
            ]
        )
        K = 3
        slice_indices = partition_atoms_to_slices(coords, K, axis=0)

        # All atoms should go to slice 0 when coordinates are identical
        assert (slice_indices == 0).all()

    def test_single_slice(self):
        """Test partitioning into a single slice."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        K = 1
        slice_indices = partition_atoms_to_slices(coords, K, axis=0)

        # All atoms should go to slice 0
        assert (slice_indices == 0).all()


class TestPartitionAtomsByPosition:
    """Tests for partition_atoms_by_position function."""

    def test_index_split_partition(self):
        """Test INDEX_SPLIT partitioning strategy."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        # Test with rank 0
        node_partition_0, rank_indices = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=0,
            world_size=2,
        )

        # First half of atoms should go to rank 0
        assert len(node_partition_0) == 2
        assert 0 in node_partition_0
        assert 1 in node_partition_0

        # Test with rank 1
        node_partition_1, rank_indices = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=1,
            world_size=2,
        )

        # Second half of atoms should go to rank 1
        assert len(node_partition_1) == 2
        assert 2 in node_partition_1
        assert 3 in node_partition_1

    def test_index_split_uneven_partition(self):
        """Test INDEX_SPLIT with uneven distribution."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        node_partition_0, _ = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=0,
            world_size=2,
        )

        node_partition_1, _ = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=1,
            world_size=2,
        )

        # Should have 2 + 1 = 3 atoms total
        assert len(node_partition_0) + len(node_partition_1) == 3

    def test_slice_partition(self):
        """Test SLICE partitioning strategy."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        node_partition_0, rank_indices = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.SLICE,
            rank=0,
            world_size=2,
        )

        # First half (by x-coordinate) should go to rank 0
        assert len(node_partition_0) == 2

        node_partition_1, _ = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.SLICE,
            rank=1,
            world_size=2,
        )

        # Second half should go to rank 1
        assert len(node_partition_1) == 2

        # Ensure all atoms are accounted for
        all_atoms = set(node_partition_0.tolist()) | set(node_partition_1.tolist())
        assert all_atoms == {0, 1, 2, 3}

    def test_cube_partition(self):
        """Test CUBE partitioning strategy with 8 ranks (2^3)."""
        # Create atoms spread across a 3D space
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        all_assigned_atoms = set()
        for rank in range(8):
            node_partition, rank_indices = partition_atoms_by_position(
                positions,
                method=PartitionStrategy.CUBE,
                rank=rank,
                world_size=8,
            )
            all_assigned_atoms.update(node_partition.tolist())

        # All 8 atoms should be assigned across all ranks
        assert all_assigned_atoms == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_cube_partition_requires_cube_world_size(self):
        """Test that CUBE partitioning requires world_size to be a perfect cube."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        # world_size=7 is not a perfect cube
        with pytest.raises(AssertionError):
            partition_atoms_by_position(
                positions,
                method=PartitionStrategy.CUBE,
                rank=0,
                world_size=7,
            )

    def test_single_rank_world(self):
        """Test partitioning with a single rank (world_size=1)."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        node_partition, rank_indices = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=0,
            world_size=1,
        )

        # All atoms should go to the single rank
        assert len(node_partition) == 3
        assert set(node_partition.tolist()) == {0, 1, 2}

    def test_invalid_method(self):
        """Test that invalid partition method raises an error."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
            ]
        )

        with pytest.raises(ValueError, match="not recognized"):
            partition_atoms_by_position(
                positions,
                method="invalid_method",
                rank=0,
                world_size=1,
            )

    def test_rank_indices_consistency(self):
        """Test that rank_indices are consistent across all ranks."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        _, rank_indices_0 = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=0,
            world_size=2,
        )

        _, rank_indices_1 = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.INDEX_SPLIT,
            rank=1,
            world_size=2,
        )

        # rank_indices should be the same regardless of which rank is querying
        assert torch.equal(rank_indices_0, rank_indices_1)


class TestPartitionPerformance:
    """Performance tests for partition functions."""

    @pytest.mark.gpu()
    def test_kmeans_partition_1m_atoms_128_clusters_gpu(self):
        """Test speed of partitioning 1M atoms to 128 clusters on GPU."""
        import time

        # Create 1M random atoms on GPU
        num_atoms = 1_000_000
        num_clusters = 128
        positions = torch.randn(num_atoms, 3, device="cuda")

        # Warm-up run
        _ = partition_atoms_kmeans(positions, k=num_clusters, seed=42)
        torch.cuda.synchronize()

        # Timed run
        start_time = time.perf_counter()
        cluster_indices = partition_atoms_kmeans(positions, k=num_clusters, seed=42)
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        assert cluster_indices.shape == (num_atoms,)
        assert cluster_indices.min() >= 0
        assert cluster_indices.max() < num_clusters
        assert cluster_indices.device.type == "cuda"

        # Print timing information
        print(
            f"\nKMEANS partition {num_atoms:,} atoms to {num_clusters} clusters on GPU: {elapsed_time:.4f} seconds"
        )


class TestPartitionAtomsKmeans:
    """Tests for partition_atoms_kmeans function."""

    def test_basic_clustering(self):
        """Test basic k-means clustering into 2 clusters."""
        # Create two distinct clusters
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [0.2, 0.0, 0.1],
                [10.0, 10.0, 10.0],
                [10.1, 10.0, 10.1],
                [10.0, 10.2, 10.0],
            ]
        )

        cluster_indices = partition_atoms_kmeans(coords, k=2, seed=42)

        assert cluster_indices.shape == (6,)
        # All indices should be 0 or 1
        assert cluster_indices.min() >= 0
        assert cluster_indices.max() <= 1

        # The first 3 atoms should be in one cluster, last 3 in another
        assert cluster_indices[0] == cluster_indices[1] == cluster_indices[2]
        assert cluster_indices[3] == cluster_indices[4] == cluster_indices[5]
        assert cluster_indices[0] != cluster_indices[3]

    def test_single_cluster(self):
        """Test clustering into a single cluster."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        cluster_indices = partition_atoms_kmeans(coords, k=1, seed=42)

        # All atoms should be in cluster 0
        assert (cluster_indices == 0).all()

    def test_more_clusters_than_atoms(self):
        """Test when k >= N (more clusters than atoms)."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        cluster_indices = partition_atoms_kmeans(coords, k=5, seed=42)

        # Each atom gets its own cluster index
        assert cluster_indices.shape == (2,)
        assert cluster_indices[0].item() == 0
        assert cluster_indices[1].item() == 1

    def test_equal_clusters_and_atoms(self):
        """Test when k == N."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        cluster_indices = partition_atoms_kmeans(coords, k=3, seed=42)

        # Each atom gets its own cluster
        assert cluster_indices.shape == (3,)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same seed."""
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
            ]
        )

        result1 = partition_atoms_kmeans(coords, k=2, seed=123)
        result2 = partition_atoms_kmeans(coords, k=2, seed=123)

        assert torch.equal(result1, result2)

    def test_three_clusters(self):
        """Test clustering into 3 clusters."""
        # Create three distinct clusters
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.1, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
            ]
        )

        cluster_indices = partition_atoms_kmeans(coords, k=3, seed=42)

        assert cluster_indices.shape == (6,)
        # All indices should be 0, 1, or 2
        assert cluster_indices.min() >= 0
        assert cluster_indices.max() <= 2

        # Each pair should be in the same cluster
        assert cluster_indices[0] == cluster_indices[1]
        assert cluster_indices[2] == cluster_indices[3]
        assert cluster_indices[4] == cluster_indices[5]

    def test_all_atoms_assigned(self):
        """Test that all atoms are assigned to some cluster."""
        coords = torch.randn(100, 3)

        cluster_indices = partition_atoms_kmeans(coords, k=5, seed=42)

        assert cluster_indices.shape == (100,)
        # All cluster indices should be valid
        assert cluster_indices.min() >= 0
        assert cluster_indices.max() < 5

    def test_convergence(self):
        """Test that k-means converges within max_iters."""
        coords = torch.randn(50, 3)

        # Should not raise any errors
        cluster_indices = partition_atoms_kmeans(coords, k=3, max_iters=100, seed=42)

        assert cluster_indices.shape == (50,)


class TestPartitionAtomsByPositionKmeans:
    """Tests for partition_atoms_by_position with KMEANS strategy."""

    def test_kmeans_partition_basic(self):
        """Test KMEANS partitioning strategy."""
        # Create two distinct clusters
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [10.0, 10.0, 10.0],
                [10.1, 10.0, 10.1],
            ]
        )

        all_assigned = set()
        for rank in range(2):
            node_partition, rank_indices = partition_atoms_by_position(
                positions,
                method=PartitionStrategy.KMEANS,
                rank=rank,
                world_size=2,
            )
            all_assigned.update(node_partition.tolist())

        # All atoms should be assigned
        assert all_assigned == {0, 1, 2, 3}

    def test_kmeans_partition_any_world_size(self):
        """Test that KMEANS works with any world_size (not just cubes or slices)."""
        positions = torch.randn(20, 3)

        # Test with world_size=7 (not a perfect cube, not easy for slices)
        all_assigned = set()
        for rank in range(7):
            node_partition, rank_indices = partition_atoms_by_position(
                positions,
                method=PartitionStrategy.KMEANS,
                rank=rank,
                world_size=7,
            )
            all_assigned.update(node_partition.tolist())
        # All atoms should be assigned
        assert all_assigned == set(range(20))

    def test_kmeans_partition_single_rank(self):
        """Test KMEANS with single rank."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        node_partition, rank_indices = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.KMEANS,
            rank=0,
            world_size=1,
        )

        # All atoms should go to the single rank
        assert len(node_partition) == 3
        assert set(node_partition.tolist()) == {0, 1, 2}

    def test_kmeans_rank_indices_consistency(self):
        """Test that rank_indices are consistent across all ranks for KMEANS."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
            ]
        )

        _, rank_indices_0 = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.KMEANS,
            rank=0,
            world_size=2,
        )

        _, rank_indices_1 = partition_atoms_by_position(
            positions,
            method=PartitionStrategy.KMEANS,
            rank=1,
            world_size=2,
        )

        # rank_indices should be the same regardless of which rank is querying
        assert torch.equal(rank_indices_0, rank_indices_1)
