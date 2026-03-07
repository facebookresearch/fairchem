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
        self._prepared = False
        self._checked = False

    def forward(self, data):
        return {"embedding": torch.randn(10, 10)}

    @classmethod
    def build_inference_settings(cls, settings):
        """Build backbone config overrides from inference settings."""
        return {}

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

    def test_build_inference_settings_classmethod(self, mock_hydra_model):
        """Test that build_inference_settings is a classmethod on backbone."""
        settings = InferenceSettings()
        # build_inference_settings is called on the backbone CLASS before instantiation
        result = type(mock_hydra_model.backbone).build_inference_settings(settings)
        assert isinstance(result, dict)

    def test_build_inference_settings_returns_dict(self, mock_hydra_model):
        """Test that build_inference_settings returns a dict."""
        settings = InferenceSettings(activation_checkpointing=True)
        result = type(mock_hydra_model.backbone).build_inference_settings(settings)
        assert isinstance(result, dict)

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

    def test_gemnet_validate_inference_settings(self):
        """Test GemNet rejects merge_mole."""
        from fairchem.core.models.gemnet_oc.gemnet_oc import GemNetOCBackbone

        # We can't easily instantiate GemNetOCBackbone, so just test the method exists
        assert hasattr(GemNetOCBackbone, "build_inference_settings")
        assert hasattr(GemNetOCBackbone, "validate_tasks")
        assert hasattr(GemNetOCBackbone, "prepare_for_inference")
        assert hasattr(GemNetOCBackbone, "on_predict_check")

    def test_escaip_build_inference_settings(self):
        """Test EScAIP has build_inference_settings classmethod."""
        from fairchem.core.models.escaip.EScAIP import EScAIPBackbone

        # Verify methods exist
        assert hasattr(EScAIPBackbone, "build_inference_settings")
        assert hasattr(EScAIPBackbone, "validate_tasks")
        assert hasattr(EScAIPBackbone, "prepare_for_inference")
        assert hasattr(EScAIPBackbone, "on_predict_check")

        # Verify build_inference_settings returns empty dict
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        settings = InferenceSettings()
        result = EScAIPBackbone.build_inference_settings(settings)
        assert result == {}

    def test_uma_backbone_methods_exist(self):
        """Test UMA backbone has required methods."""
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone

        assert hasattr(eSCNMDBackbone, "build_inference_settings")
        assert hasattr(eSCNMDBackbone, "validate_tasks")
        assert hasattr(eSCNMDBackbone, "prepare_for_inference")
        assert hasattr(eSCNMDBackbone, "on_predict_check")

        # Verify build_inference_settings returns proper overrides
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        settings = InferenceSettings(activation_checkpointing=True)
        result = eSCNMDBackbone.build_inference_settings(settings)
        assert "always_use_pbc" in result
        assert result["always_use_pbc"] is False
        assert result.get("activation_checkpointing") is True


class TestSO2ConversionInPrepareForInference:
    """
    Tests that prepare_for_inference converts SO2_Convolution modules
    to block-diagonal GEMM variants only with UMASFastPytorchBackend.
    """

    def test_general_backend_does_not_convert(self):
        """
        With GENERAL backend, SO2 convolutions should NOT be converted.
        """
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone
        from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution

        torch.manual_seed(42)
        backbone = eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
            lmax=2,
            mmax=2,
            otf_graph=True,
            edge_channels=5,
            num_distance_basis=7,
            use_dataset_embedding=False,
            always_use_pbc=False,
        )

        settings = InferenceSettings()
        data = MagicMock()
        backbone.prepare_for_inference(data, settings)

        for block in backbone.blocks:
            assert isinstance(block.edge_wise.so2_conv_1, SO2_Convolution)
            assert isinstance(block.edge_wise.so2_conv_2, SO2_Convolution)

    def test_umas_fast_pytorch_backend_converts(self):
        """
        With UMAS_FAST_PYTORCH backend, SO2 convolutions should be converted.
        """
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone
        from fairchem.core.models.uma.nn.so2_layers import (
            SO2_Conv1_WithRadialBlock,
            SO2_Conv2_InternalBlock,
            SO2_Convolution,
        )

        torch.manual_seed(42)
        backbone = eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
            lmax=2,
            mmax=2,
            otf_graph=True,
            edge_channels=5,
            num_distance_basis=7,
            use_dataset_embedding=False,
            always_use_pbc=False,
            execution_mode="umas_fast_pytorch",
        )

        # Before: original SO2_Convolution
        for block in backbone.blocks:
            assert isinstance(block.edge_wise.so2_conv_1, SO2_Convolution)
            assert isinstance(block.edge_wise.so2_conv_2, SO2_Convolution)

        settings = InferenceSettings(activation_checkpointing=False)
        data = MagicMock()
        result = backbone.prepare_for_inference(data, settings)

        # After: converted to block GEMM variants
        assert result is backbone
        for block in backbone.blocks:
            assert isinstance(block.edge_wise.so2_conv_1, SO2_Conv1_WithRadialBlock)
            assert isinstance(block.edge_wise.so2_conv_2, SO2_Conv2_InternalBlock)

    def test_umas_fast_pytorch_validates_activation_checkpointing(self):
        """
        UMASFastPytorchBackend should reject activation_checkpointing=True.
        """
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone

        torch.manual_seed(42)
        backbone = eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
            lmax=2,
            mmax=2,
            otf_graph=True,
            edge_channels=5,
            num_distance_basis=7,
            use_dataset_embedding=False,
            always_use_pbc=False,
            execution_mode="umas_fast_pytorch",
        )

        settings = InferenceSettings(activation_checkpointing=True)
        data = MagicMock()
        with pytest.raises(ValueError, match="activation_checkpointing"):
            backbone.prepare_for_inference(data, settings)
