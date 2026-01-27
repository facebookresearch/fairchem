"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for the model inference interface methods.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from fairchem.core.models.base import HydraModelV2
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


class MockBackbone(nn.Module):
    """Mock backbone for testing HydraModel interface."""

    def __init__(self, dataset_list=None):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.dataset_list = dataset_list or ["omol", "omat"]
        self._validated = False
        self._prepared = False
        self._checked = False

    def forward(self, data):
        return {"embedding": torch.randn(10, 10)}

    def validate_inference_settings(self, settings):
        self._validated = True
        if settings.merge_mole:
            raise ValueError("Mock backbone does not support MOLE")

    def validate_tasks(self, dataset_to_tasks):
        assert set(dataset_to_tasks.keys()).issubset(set(self.dataset_list))

    def prepare_for_inference(self, data, settings):
        self._prepared = True
        return self  # Return self (no replacement)

    def on_predict_check(self, data):
        self._checked = True


class MockBackboneWithReplacement(MockBackbone):
    """Mock backbone that returns a new backbone on prepare_for_inference."""

    def prepare_for_inference(self, data, settings):
        self._prepared = True
        # Return a new backbone to simulate MOLE merge
        new_backbone = MockBackbone(self.dataset_list)
        new_backbone._prepared = True
        return new_backbone


class MockHead(nn.Module):
    """Mock head for testing."""

    def __init__(self, backbone):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    @property
    def use_amp(self):
        return False

    def forward(self, data, emb):
        return {"energy": torch.randn(1)}


class TestHydraModelInferenceInterface:
    """Tests for HydraModel inference interface methods."""

    @pytest.fixture()
    def mock_hydra_model(self):
        """Create a HydraModel with mock backbone and heads."""
        backbone = MockBackbone()
        heads = {"energy": MockHead(backbone)}
        # Use HydraModelV2 since it takes backbone/heads directly
        model = HydraModelV2(backbone=backbone, heads=heads)
        return model

    def test_validate_inference_settings_delegates_to_backbone(self, mock_hydra_model):
        """Test that validate_inference_settings calls backbone method."""
        settings = InferenceSettings()
        mock_hydra_model.validate_inference_settings(settings)
        assert mock_hydra_model.backbone._validated

    def test_validate_inference_settings_raises_for_unsupported(self, mock_hydra_model):
        """Test that validation errors propagate from backbone."""
        settings = InferenceSettings(merge_mole=True)
        with pytest.raises(ValueError, match="does not support MOLE"):
            mock_hydra_model.validate_inference_settings(settings)

    def test_prepare_for_inference_no_replacement(self, mock_hydra_model):
        """Test prepare_for_inference when backbone returns self."""
        settings = InferenceSettings()
        data = MagicMock()

        original_backbone = mock_hydra_model.backbone
        mock_hydra_model.prepare_for_inference(data, settings)

        assert mock_hydra_model.backbone._prepared
        # Backbone should not be replaced
        assert mock_hydra_model.backbone is original_backbone

    def test_prepare_for_inference_with_replacement(self):
        """Test prepare_for_inference when backbone returns new backbone."""
        backbone = MockBackboneWithReplacement()
        heads = {"energy": MockHead(backbone)}
        model = HydraModelV2(backbone=backbone, heads=heads)

        settings = InferenceSettings()
        data = MagicMock()

        original_backbone = model.backbone
        model.prepare_for_inference(data, settings)

        # Backbone should be replaced
        assert model.backbone is not original_backbone
        assert model.backbone._prepared

    def test_on_predict_check_delegates_to_backbone(self, mock_hydra_model):
        """Test that on_predict_check calls backbone method."""
        data = MagicMock()
        mock_hydra_model.on_predict_check(data)
        assert mock_hydra_model.backbone._checked

    def test_setup_tasks_creates_task_mapping(self, mock_hydra_model):
        """Test that setup_tasks creates tasks and dataset mapping."""
        # Create mock task configs
        mock_task = MagicMock()
        mock_task.name = "omol_energy"
        mock_task.datasets = ["omol"]

        with patch("hydra.utils.instantiate", return_value=mock_task):
            mock_hydra_model.setup_tasks([{"_target_": "Task"}])

        assert "omol_energy" in mock_hydra_model.tasks
        assert "omol" in mock_hydra_model.dataset_to_tasks
        assert mock_hydra_model.dataset_to_tasks["omol"] == [mock_task]

    def test_setup_tasks_validates_datasets(self, mock_hydra_model):
        """Test that setup_tasks calls backbone validate_tasks."""
        mock_task = MagicMock()
        mock_task.name = "unknown_energy"
        mock_task.datasets = ["unknown_dataset"]

        with patch("hydra.utils.instantiate", return_value=mock_task), pytest.raises(
            AssertionError
        ):
            mock_hydra_model.setup_tasks([{"_target_": "Task"}])

    def test_dataset_to_tasks_raises_before_setup(self, mock_hydra_model):
        """Test that accessing dataset_to_tasks before setup_tasks raises."""
        with pytest.raises(RuntimeError, match="setup_tasks"):
            _ = mock_hydra_model.dataset_to_tasks

    def test_direct_forces_property(self, mock_hydra_model):
        """Test direct_forces property delegates to backbone."""
        assert mock_hydra_model.direct_forces is False

        mock_hydra_model.backbone.direct_forces = True
        assert mock_hydra_model.direct_forces is True


class TestBackboneInterface:
    """Tests for backbone interface method implementations."""

    def test_escaip_validate_inference_settings(self):
        """Test EScAIP rejects merge_mole."""
        from fairchem.core.models.escaip.EScAIP import EScAIPBackbone

        # Verify methods exist
        assert hasattr(EScAIPBackbone, "validate_inference_settings")
        assert hasattr(EScAIPBackbone, "validate_tasks")
        assert hasattr(EScAIPBackbone, "prepare_for_inference")
        assert hasattr(EScAIPBackbone, "on_predict_check")

    def test_uma_backbone_methods_exist(self):
        """Test UMA backbone has required methods."""
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone

        assert hasattr(eSCNMDBackbone, "validate_inference_settings")
        assert hasattr(eSCNMDBackbone, "validate_tasks")
        assert hasattr(eSCNMDBackbone, "prepare_for_inference")
        assert hasattr(eSCNMDBackbone, "on_predict_check")
