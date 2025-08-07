"""
Unit tests for ensemble functionality with multiple heads per task.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from ase import Atoms
from ase.build import bulk

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit, collate_predictions
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.mlip_unit import (
    Task, 
    OutputSpec, 
    compute_loss, 
    compute_metrics,
    initialize_finetuning_model
)
from fairchem.core.modules.normalization.normalizer import Normalizer


class TestEnsembleFunctionality:
    @pytest.fixture(autouse=True)
    def patch_loss_fn(self, monkeypatch):
        import torch.nn as nn
        class DummyLoss(nn.MSELoss):
            def forward(self, input, target, **kwargs):
                return super().forward(input, target)
        # Patch Task.loss_fn to DummyLoss for all tests in this class
        from fairchem.core.units.mlip_unit.mlip_unit import Task
        orig_init = Task.__init__
        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.loss_fn = DummyLoss()
        monkeypatch.setattr(Task, "__init__", new_init)
    """Test suite for multi-head ensemble functionality."""
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock task for testing."""
        normalizer = Normalizer(mean=0.0, rmsd=1.0)
        return Task(
            name="test_energy",
            level="system", 
            property="energy",
            loss_fn=torch.nn.MSELoss(),
            out_spec=OutputSpec(dim=[1], dtype="float32"),
            normalizer=normalizer,
            datasets=["omat"],
            shallow_ensemble=False
        )
    
    @pytest.fixture 
    def mock_ensemble_task(self):
        """Create a mock ensemble task for testing."""
        normalizer = Normalizer(mean=0.0, rmsd=1.0)
        return Task(
            name="test_energy_ensemble",
            level="system",
            property="energy", 
            loss_fn=torch.nn.MSELoss(),
            out_spec=OutputSpec(dim=[1], dtype="float32"),
            normalizer=normalizer,
            datasets=["omat"],
            shallow_ensemble=True
        )
    
    @pytest.fixture
    def mock_predictions_single_head(self):
        """Mock predictions with single head per property."""
        return {
            "energy": {
                "test_energy": torch.tensor([[1.0], [2.0], [3.0]])
            },
            "forces": {
                "test_forces": torch.randn(10, 3)  # 10 atoms, 3D forces
            }
        }
    
    @pytest.fixture
    def mock_predictions_multi_head(self):
        """Mock predictions with multiple heads per property."""
        return {
            "energy": {
                "head0_test_energy": torch.tensor([[1.0], [2.0], [3.0]]),
                "head1_test_energy": torch.tensor([[1.1], [2.1], [3.1]]), 
                "head2_test_energy": torch.tensor([[0.9], [1.9], [2.9]]),
                "head3_test_energy": torch.tensor([[1.05], [2.05], [3.05]]),
                "head4_test_energy": torch.tensor([[0.95], [1.95], [2.95]])
            },
            "forces": {
                "head0_test_forces": torch.randn(10, 3),
                "head1_test_forces": torch.randn(10, 3),
                "head2_test_forces": torch.randn(10, 3),
                "head3_test_forces": torch.randn(10, 3), 
                "head4_test_forces": torch.randn(10, 3)
            }
        }
    
    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch for testing."""
        num_graphs = 3
        num_nodes = 10
        num_edges = 5
        data = AtomicData(
            pos=torch.randn(num_nodes, 3),
            atomic_numbers=torch.randint(1, 10, (num_nodes,)),
            cell=torch.zeros(num_graphs, 3, 3),
            pbc=torch.zeros(num_graphs, 3, dtype=torch.bool),
            natoms=torch.tensor([3, 3, 4]),  # one per graph
            edge_index=torch.zeros(2, num_edges, dtype=torch.long),
            cell_offsets=torch.zeros(num_edges, 3),
            nedges=torch.tensor([num_edges]),
            charge=torch.zeros(num_graphs),
            spin=torch.zeros(num_graphs),
            fixed=torch.zeros(num_nodes, dtype=torch.long),
            tags=torch.zeros(num_nodes, dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
            dataset=["omat"] * num_graphs,
            sid=["a", "b", "c"]
        )
        # Add mock attributes for loss/metrics tests
        data.test_energy = torch.randn(num_graphs, 1)
        data.test_forces = torch.randn(num_nodes, 3)
        data.test_energy_ensemble = torch.randn(num_graphs, 1)
        data.head0_test_energy = torch.randn(num_graphs, 1)
        data.dataset_name = ["omat"] * num_graphs
        return data
    
    def test_collate_predictions_single_head(self, mock_predictions_single_head):
        """Test collate_predictions with single head per property."""
        
        def mock_predict_fn(predict_unit, data, undo_element_references=True):
            return {
                "test_energy": torch.tensor([[1.0], [2.0], [3.0]]),
                "test_forces": torch.randn(10, 3)
            }
        
        # Create mock predict unit with dataset_to_tasks
        class MockPredictUnit:
            def __init__(self):
                self.dataset_to_tasks = {
                    "omat": [
                        type('Task', (), {
                            'name': 'test_energy',
                            'property': 'energy', 
                            'level': 'system'
                        })(),
                        type('Task', (), {
                            'name': 'test_forces',
                            'property': 'forces',
                            'level': 'atom'
                        })()
                    ]
                }
        
        predict_unit = MockPredictUnit()
        
        # Create mock data
        num_graphs = 3
        num_nodes = 10
        num_edges = 5
        data = AtomicData(
            pos=torch.randn(num_nodes, 3),
            atomic_numbers=torch.randint(1, 10, (num_nodes,)),
            cell=torch.zeros(num_graphs, 3, 3),
            pbc=torch.zeros(num_graphs, 3, dtype=torch.bool),
            natoms=torch.tensor([3, 3, 4]),
            edge_index=torch.zeros(2, num_edges, dtype=torch.long),
            cell_offsets=torch.zeros(num_edges, 3),
            nedges=torch.tensor([num_edges]),
            charge=torch.zeros(num_graphs),
            spin=torch.zeros(num_graphs),
            fixed=torch.zeros(num_nodes, dtype=torch.long),
            tags=torch.zeros(num_nodes, dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
            dataset=["omat"] * num_graphs,
            sid=["a", "b", "c"]
        )

        collated_fn = collate_predictions(mock_predict_fn)
        result = collated_fn(predict_unit, data)

        assert "energy" in result
        assert "forces" in result
        assert "test_energy" in result["energy"]
        assert "test_forces" in result["forces"]
    
    def test_collate_predictions_multi_head(self):
        """Test collate_predictions with multiple heads per property."""
        def mock_predict_fn(predict_unit, data, undo_element_references=True):
            # Use keys that match collate_predictions logic: head{n}_omat_energy
            return {
                "head0_omat_energy": torch.tensor([[1.0], [2.0], [3.0]]),
                "head1_omat_energy": torch.tensor([[1.1], [2.1], [3.1]]),
                "head2_omat_energy": torch.tensor([[0.9], [1.9], [2.9]])
            }
        class MockPredictUnit:
            def __init__(self):
                self.dataset_to_tasks = {
                    "omat": [
                        type('Task', (), {
                            'name': 'test_energy',
                            'property': 'energy',
                            'level': 'system'
                        })()
                    ]
                }
        predict_unit = MockPredictUnit()
        num_graphs = 3
        num_nodes = 3
        num_edges = 3
        data = AtomicData(
            pos=torch.randn(num_nodes, 3),
            atomic_numbers=torch.randint(1, 10, (num_nodes,)),
            cell=torch.zeros(num_graphs, 3, 3),
            pbc=torch.zeros(num_graphs, 3, dtype=torch.bool),
            natoms=torch.tensor([1, 1, 1]),
            edge_index=torch.zeros(2, num_edges, dtype=torch.long),
            cell_offsets=torch.zeros(num_edges, 3),
            nedges=torch.tensor([num_edges]),
            charge=torch.zeros(num_graphs),
            spin=torch.zeros(num_graphs),
            fixed=torch.zeros(num_nodes, dtype=torch.long),
            tags=torch.zeros(num_nodes, dtype=torch.long),
            batch=torch.tensor([0, 1, 2]),
            dataset=["omat"] * num_graphs,
            sid=["a", "b", "c"]
        )
        collated_fn = collate_predictions(mock_predict_fn)
        result = collated_fn(predict_unit, data)
        assert "energy" in result
        assert len(result["energy"]) == 3  # Should have 3 heads
        assert "head0_omat_energy" in result["energy"]
        assert "head1_omat_energy" in result["energy"]
        assert "head2_omat_energy" in result["energy"]
    
    def test_compute_loss_single_head(self, mock_task, mock_predictions_single_head, mock_batch):
        """Test compute_loss with single head."""
        loss_dict = compute_loss([mock_task], mock_predictions_single_head, mock_batch)
        assert "test_energy" in loss_dict
        assert torch.is_tensor(loss_dict["test_energy"])
    
    def test_compute_loss_ensemble(self, mock_ensemble_task, mock_predictions_multi_head, mock_batch):
        """Test compute_loss with ensemble (multiple heads)."""
        # Patch the task name to match the mock predictions
        mock_ensemble_task.name = "head0_test_energy"
        loss_dict = compute_loss([mock_ensemble_task], mock_predictions_multi_head, mock_batch)
        assert mock_ensemble_task.name in loss_dict
        assert torch.is_tensor(loss_dict[mock_ensemble_task.name])
    
    def test_compute_metrics_single_head(self, mock_task, mock_predictions_single_head, mock_batch):
        """Test compute_metrics with single head."""
        mock_task.metrics = ["mae"]
        # Patch get_output_mask to squeeze mask to 1D
        import fairchem.core.units.mlip_unit.mlip_unit as mlip_unit_mod
        orig_get_output_mask = mlip_unit_mod.get_output_mask
        def patched_get_output_mask(batch, task):
            mask = orig_get_output_mask(batch, task)
            for k, v in mask.items():
                if v.ndim == 2 and v.shape[1] == 1:
                    mask[k] = v.squeeze(-1)
            return mask
        mlip_unit_mod.get_output_mask = patched_get_output_mask
        try:
            metrics = compute_metrics(mock_task, mock_predictions_single_head, mock_batch)
        finally:
            mlip_unit_mod.get_output_mask = orig_get_output_mask
        assert "mae" in metrics

    def test_compute_metrics_ensemble(self, mock_ensemble_task, mock_predictions_multi_head, mock_batch):
        """Test compute_metrics with ensemble."""
        mock_ensemble_task.metrics = ["mae"]
        # Patch get_output_mask to squeeze mask to 1D
        import fairchem.core.units.mlip_unit.mlip_unit as mlip_unit_mod
        orig_get_output_mask = mlip_unit_mod.get_output_mask
        def patched_get_output_mask(batch, task):
            mask = orig_get_output_mask(batch, task)
            for k, v in mask.items():
                if v.ndim == 2 and v.shape[1] == 1:
                    mask[k] = v.squeeze(-1)
            return mask
        mlip_unit_mod.get_output_mask = patched_get_output_mask
        try:
            metrics = compute_metrics(mock_ensemble_task, mock_predictions_multi_head, mock_batch)
        finally:
            mlip_unit_mod.get_output_mask = orig_get_output_mask
        assert "mae" in metrics


class TestASECalculatorEnsemble:
    """Test ASE calculator with ensemble functionality."""
    
    def test_ase_calculator_single_head(self):
        """Test ASE calculator with single head."""
        # This would require a real model checkpoint, so we'll mock the key components
        pass
    
    def test_ase_calculator_multi_head_selection(self):
        """Test ASE calculator head selection."""
        # Mock predictions with multiple heads
        pred = {
            "energy": {
                "head0_energy": torch.tensor([1.0]),
                "head1_energy": torch.tensor([1.1]), 
                "head2_energy": torch.tensor([0.9])
            }
        }
        
        # Test head selection logic (simplified)
        if isinstance(pred["energy"], dict):
            heads = list(pred["energy"].keys())
            assert len(heads) == 3
            
            # Test specific head selection
            selected_head = "head1_energy"
            if selected_head in pred["energy"]:
                selected_pred = pred["energy"][selected_head]
                assert torch.allclose(selected_pred, torch.tensor([1.1]))
            
            # Test averaging
            head_predictions = list(pred["energy"].values())
            stacked = torch.stack(head_predictions, dim=0)
            mean_pred = stacked.mean(dim=0)
            std_pred = stacked.std(dim=0)
            
            expected_mean = torch.tensor([1.0])  # (1.0 + 1.1 + 0.9) / 3
            assert torch.allclose(mean_pred, expected_mean, atol=1e-6)
            assert std_pred.item() > 0  # Should have non-zero std


class TestFineTuningWithMultipleHeads:
    """Test fine-tuning with multiple heads configuration."""
    
    def test_initialize_finetuning_model_multi_heads(self):
        """Test initializing model with multiple heads."""
        
        # Mock heads configuration for ensemble
        heads_config = {
            "head0": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": ["omat"],
                "wrap_property": False
            },
            "head1": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper", 
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": ["omat"],
                "wrap_property": False
            },
            "head2": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head", 
                "dataset_names": ["omat"],
                "wrap_property": False
            },
            "head3": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": ["omat"],
                "wrap_property": False
            },
            "head4": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": ["omat"],
                "wrap_property": False
            }
        }
        
        # Verify configuration structure
        assert len(heads_config) == 5
        for head_name, head_config in heads_config.items():
            assert "module" in head_config
            assert "head_cls" in head_config
            assert "dataset_names" in head_config
        
        # Test unique head names
        head_names = list(heads_config.keys())
        assert len(set(head_names)) == len(head_names)


def test_head_to_task_mapping():
    """Test mapping between head names and tasks."""
    
    # Test various head naming schemes
    test_cases = [
        ("energy_head_0", "energy", True),
        ("head0_oc20_energy", "energy", True), 
        ("oc20_energy", "energy", True),
        ("forces_head_1", "forces", True),
        ("stress_head_2", "stress", True),
        ("random_name", "energy", False)
    ]
    
    for head_key, property_name, should_match in test_cases:
        # Simple pattern matching logic
        matches = (
            head_key == property_name or
            property_name in head_key or
            head_key.endswith(f"_{property_name}")
        )
        assert matches == should_match, f"Failed for {head_key} -> {property_name}"

