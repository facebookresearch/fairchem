"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, Dataset

from fairchem.core.components.common.dataloader_builder import get_dataloader


class SimpleDataset(Dataset):
    """A simple dataset for testing purposes."""

    def __init__(self, size: int = 10):
        self.size = size
        self.data = [{"x": torch.rand(3), "y": torch.rand(1)} for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleBatchSampler:
    """A simple batch sampler for testing purposes."""

    def __init__(self, dataset, num_replicas, rank, batch_size=2):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # Split indices across replicas
        indices = indices[self.rank :: self.num_replicas]
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        n = len(self.dataset) // self.num_replicas
        return (n + self.batch_size - 1) // self.batch_size


def simple_collate_fn(batch):
    """Simple collate function that stacks tensors."""
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
    }


def make_batch_sampler_fn(batch_size=2):
    """Factory function that returns a batch sampler function."""

    def batch_sampler_fn(dataset, num_replicas, rank):
        return SimpleBatchSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            batch_size=batch_size,
        )

    return batch_sampler_fn


class TestGetDataloader:
    """Test cases for the get_dataloader function."""

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_basic_dataloader_creation(self, mock_distutils, mock_gp_utils):
        """Test that get_dataloader creates a DataLoader with correct configuration."""
        # Setup mocks - gp_utils not initialized, so use distutils
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)
        assert dataloader.dataset is dataset
        assert dataloader.num_workers == 0
        assert dataloader.pin_memory is True

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_dataloader_iteration(self, mock_distutils, mock_gp_utils):
        """Test that the dataloader can be iterated correctly."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        batches = list(dataloader)
        assert len(batches) == 5  # 10 samples / 2 batch_size = 5 batches
        for batch in batches:
            assert "x" in batch
            assert "y" in batch
            assert batch["x"].shape[0] == 2  # batch size
            assert batch["x"].shape[1] == 3  # feature dim
            assert batch["y"].shape[0] == 2

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_dataloader_with_gp_utils_initialized(self, mock_distutils, mock_gp_utils):
        """Test that gp_utils is used when initialized."""
        # Setup mocks - gp_utils initialized
        mock_gp_utils.initialized.return_value = True
        mock_gp_utils.get_dp_world_size.return_value = 2
        mock_gp_utils.get_dp_rank.return_value = 1

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        # Verify gp_utils was used
        mock_gp_utils.get_dp_world_size.assert_called_once()
        mock_gp_utils.get_dp_rank.assert_called_once()
        # Verify distutils was not used
        mock_distutils.get_world_size.assert_not_called()
        mock_distutils.get_rank.assert_not_called()

        assert isinstance(dataloader, DataLoader)

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_dataloader_with_distutils(self, mock_distutils, mock_gp_utils):
        """Test that distutils is used when gp_utils is not initialized."""
        # Setup mocks - gp_utils not initialized
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 4
        mock_distutils.get_rank.return_value = 2

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        # Verify gp_utils.initialized was called
        mock_gp_utils.initialized.assert_called_once()
        # Verify distutils was used
        mock_distutils.get_world_size.assert_called_once()
        mock_distutils.get_rank.assert_called_once()

        assert isinstance(dataloader, DataLoader)

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_dataloader_with_multiple_workers(self, mock_distutils, mock_gp_utils):
        """Test dataloader creation with multiple workers."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=2,
            mp_start_method="spawn",
        )

        assert isinstance(dataloader, DataLoader)
        assert dataloader.num_workers == 2

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_multiprocessing_context_set_correctly(self, mock_distutils, mock_gp_utils):
        """Test that multiprocessing_context is only set when num_workers > 0."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=10)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        # With num_workers=0, multiprocessing_context should be None
        dataloader_no_workers = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
            mp_start_method="fork",
        )
        # The dataloader's multiprocessing_context should be None when num_workers=0
        assert dataloader_no_workers.multiprocessing_context is None

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_batch_sampler_receives_correct_args(self, mock_distutils, mock_gp_utils):
        """Test that the batch_sampler_fn receives correct arguments."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 4
        mock_distutils.get_rank.return_value = 2

        dataset = SimpleDataset(size=10)

        # Create a mock batch sampler function to verify arguments
        mock_batch_sampler_fn = MagicMock()
        mock_batch_sampler_fn.return_value = SimpleBatchSampler(
            dataset=dataset,
            num_replicas=4,
            rank=2,
            batch_size=2,
        )

        get_dataloader(
            dataset=dataset,
            batch_sampler_fn=mock_batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        # Verify batch_sampler_fn was called with correct arguments
        mock_batch_sampler_fn.assert_called_once_with(
            dataset=dataset,
            num_replicas=4,
            rank=2,
        )

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_empty_dataset(self, mock_distutils, mock_gp_utils):
        """Test dataloader with an empty dataset."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=0)
        batch_sampler_fn = make_batch_sampler_fn(batch_size=2)

        dataloader = get_dataloader(
            dataset=dataset,
            batch_sampler_fn=batch_sampler_fn,
            collate_fn=simple_collate_fn,
            num_workers=0,
        )

        assert isinstance(dataloader, DataLoader)
        batches = list(dataloader)
        assert len(batches) == 0

    @patch("fairchem.core.components.common.dataloader_builder.gp_utils")
    @patch("fairchem.core.components.common.dataloader_builder.distutils")
    def test_different_batch_sizes(self, mock_distutils, mock_gp_utils):
        """Test dataloader with different batch sizes."""
        mock_gp_utils.initialized.return_value = False
        mock_distutils.get_world_size.return_value = 1
        mock_distutils.get_rank.return_value = 0

        dataset = SimpleDataset(size=10)

        for batch_size in [1, 3, 5, 10]:
            batch_sampler_fn = make_batch_sampler_fn(batch_size=batch_size)

            dataloader = get_dataloader(
                dataset=dataset,
                batch_sampler_fn=batch_sampler_fn,
                collate_fn=simple_collate_fn,
                num_workers=0,
            )

            batches = list(dataloader)
            expected_batches = (10 + batch_size - 1) // batch_size
            assert len(batches) == expected_batches
