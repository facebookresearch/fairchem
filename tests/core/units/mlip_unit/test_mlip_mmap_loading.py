"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
from fairchem.core.units.mlip_unit.utils import load_inference_model, load_tasks


class TestMlipUnitMmapLoading:
    """Test that torch.load calls in mlip_unit use mmap=True for memory efficiency."""

    @pytest.fixture
    def mock_checkpoint(self):
        """Create a mock MLIPInferenceCheckpoint for testing."""
        return MLIPInferenceCheckpoint(
            model_config={
                "_target_": "torch.nn.Identity",
            },
            model_state_dict={},
            ema_state_dict={"n_averaged": torch.tensor(1)},
            tasks_config=[],
        )

    def test_load_inference_model_uses_mmap(self, mock_checkpoint):
        """Test that load_inference_model uses mmap=True when loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "checkpoint.pt"
            torch.save(mock_checkpoint, checkpoint_file)

            with patch("fairchem.core.units.mlip_unit.utils.torch.load") as mock_load:
                mock_load.return_value = mock_checkpoint

                try:
                    load_inference_model(str(checkpoint_file))
                except Exception:
                    pass  # We only care about whether mmap=True was passed

                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args
                # Check that mmap=True is in the kwargs
                assert (
                    call_kwargs.kwargs.get("mmap") is True
                ), "torch.load should be called with mmap=True"

    def test_load_tasks_uses_mmap(self, mock_checkpoint):
        """Test that load_tasks uses mmap=True when loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "checkpoint.pt"
            torch.save(mock_checkpoint, checkpoint_file)

            with patch("fairchem.core.units.mlip_unit.utils.torch.load") as mock_load:
                mock_load.return_value = mock_checkpoint

                try:
                    load_tasks(str(checkpoint_file))
                except Exception:
                    pass  # We only care about whether mmap=True was passed

                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args
                # Check that mmap=True is in the kwargs
                assert (
                    call_kwargs.kwargs.get("mmap") is True
                ), "torch.load should be called with mmap=True"


class TestMlipUnitMmapIntegration:
    """Integration tests that verify mmap loading works correctly in mlip_unit."""

    @pytest.fixture
    def mock_checkpoint(self):
        """Create a mock MLIPInferenceCheckpoint for testing."""
        return MLIPInferenceCheckpoint(
            model_config={
                "_target_": "torch.nn.Identity",
            },
            model_state_dict={},
            ema_state_dict={"n_averaged": torch.tensor(1)},
            tasks_config=[
                {"_target_": "torch.nn.Identity"},
            ],
        )

    def test_load_tasks_works_with_mmap(self, mock_checkpoint):
        """Test that load_tasks works correctly with mmap=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "checkpoint.pt"
            torch.save(mock_checkpoint, checkpoint_file)

            tasks = load_tasks(str(checkpoint_file))
            # Tasks should be loaded successfully (they're Identity modules in our mock)
            assert isinstance(tasks, list)
            assert len(tasks) == 1
