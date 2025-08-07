import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))
"""
Pytest-compatible tests for ensemble model with 5 heads.
This module tests the full workflow from checkpoint to ensemble prediction.
"""

import os
import tempfile
import yaml
from pathlib import Path
import pytest

import torch
import numpy as np
from ase import Atoms
from ase.build import bulk

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model
from fairchem.core import pretrained_mlip


@pytest.fixture(scope="module")
def fine_tuning_config():
    """Create a complete fine-tuning configuration with 5 heads."""
    
    config = {
        # Model configuration with 5 heads
        "model": {
            "heads": {
                "energy_head_0": {
                    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "dataset_names": ["omat"],
                    "wrap_property": False
                },
                "energy_head_1": {
                    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "dataset_names": ["omat"],
                    "wrap_property": False
                },
                "energy_head_2": {
                    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "dataset_names": ["omat"],
                    "wrap_property": False
                },
                "energy_head_3": {
                    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "dataset_names": ["omat"],
                    "wrap_property": False
                },
                "energy_head_4": {
                    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "dataset_names": ["omat"],
                    "wrap_property": False
                }
            }
        },
        
        # Task configuration for ensemble
        "tasks": [
            {
                "name": "test_energy",
                "level": "system",
                "property": "energy",
                "loss_fn": "torch.nn.MSELoss",
                "out_spec": {"dim": [1], "dtype": "float32"},
            "datasets": ["omat"],
                "shallow_ensemble": True,  # Enable ensemble mode
                "metrics": ["mae", "rmse"]
            }
        ]
    }
    
    return config


@pytest.fixture(scope="module")
def ensemble_model(fine_tuning_config):
    """Simulate fine-tuning process and create an ensemble model."""
    
    # Get available models
    available_models = pretrained_mlip.available_models
    
    # Use uma-s-1p1 if available, otherwise use first available
    checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
    
    # Get the checkpoint path
    from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS
    from huggingface_hub import hf_hub_download
    
    # Get checkpoint path similar to how get_predict_unit does it
    model_checkpoint = _MODEL_CKPTS.checkpoints[checkpoint_name]
    base_checkpoint = hf_hub_download(
        filename=model_checkpoint.filename,
        repo_id=model_checkpoint.repo_id,
        subfolder=model_checkpoint.subfolder,
        revision=model_checkpoint.revision,
    )
    
    # Create fine-tuning configuration
    heads_config = fine_tuning_config["model"]["heads"]
    
    # Initialize model with ensemble heads
    model = initialize_finetuning_model(
        checkpoint_location=base_checkpoint,
        heads=heads_config
    )
    
    return model


def test_ensemble_model_initialization(ensemble_model):
    """Test that ensemble model is properly initialized with 5 heads."""
    
    # Verify model structure
    assert len(ensemble_model.output_heads) == 5
    assert hasattr(ensemble_model, 'backbone')
    
    # Check head names
    expected_heads = [f"energy_head_{i}" for i in range(5)]
    actual_heads = list(ensemble_model.output_heads.keys())
    
    for expected_head in expected_heads:
        assert expected_head in actual_heads, f"Missing head: {expected_head}"


def test_ensemble_predictions(ensemble_model):
    # Disable stress/force regression to avoid autograd in eval mode
    ensemble_model.regress_stress = False
    ensemble_model.regress_forces = False
    def disable_regress_flags(module):
        if hasattr(module, 'regress_stress'):
            module.regress_stress = False
        if hasattr(module, 'regress_forces'):
            module.regress_forces = False
        for child in getattr(module, 'children', lambda: [])():
            disable_regress_flags(child)

    for head in getattr(ensemble_model, "output_heads", {}).values():
        disable_regress_flags(head)
    """Test that the ensemble model produces predictions from multiple heads."""
    
    try:
        # Create test input
        from fairchem.core.datasets.atomic_data import AtomicData

        # Create simple test system
        test_atoms = bulk('Cu', 'fcc', a=3.6)
        test_atoms.pbc = True

        # Convert to AtomicData format
        # Use the dataset/task name expected by the model ('omat')
        valid_task = "omat"
        print(f"DEBUG: valid_task = {valid_task}")
        data = AtomicData.from_ase(
            test_atoms,
            r_edges=True,
            max_neigh=12,  # typical for fcc
            radius=3.0,    # nearest neighbor distance for fcc Cu
            task_name=valid_task
        )
        # Ensure dataset is a list of one string, as expected by model
        data.dataset = [valid_task]

        # Run model prediction once
        ensemble_model.eval()
        device = next(ensemble_model.parameters()).device
        predictions = ensemble_model(data.to(device))

        # Check that we have a nested dict: {task_name: {head_name: value}}
        assert isinstance(predictions, dict), f"Predictions should be a dict, got {type(predictions)}"
        assert len(predictions) == 1, f"Expected 1 task, got {len(predictions)}. Keys: {list(predictions.keys())}"
        task_key = next(iter(predictions.keys()))
        heads = predictions[task_key]
        assert isinstance(heads, dict), f"Heads should be a dict, got {type(heads)}"
        assert len(heads) == 5, f"Expected 5 heads, got {len(heads)}. Keys: {list(heads.keys())}"
        for key, value in heads.items():
            assert value is not None, f"Null prediction for {key}"

    except Exception as e:
        pytest.fail(f"Ensemble predictions test failed: {e}")


def test_ase_calculator_with_ensemble():
    """Test ASE calculator functionality with ensemble."""
    
    try:
        # For this test, we'll use a standard model to test the calculator logic
        available_models = pretrained_mlip.available_models
        checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
        
        # Test different ways to create calculator
        calc1 = FAIRChemCalculator.from_model_checkpoint(checkpoint_name, task_name="oc20")
        
        calc2 = FAIRChemCalculator.from_model_checkpoint(
            checkpoint_name, 
            task_name="oc20",
            head_name=None  # Will use default behavior
        )
        
        # Create test system
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        
        # Test predictions
        atoms.calc = calc1
        energy1 = atoms.get_potential_energy()
        forces1 = atoms.get_forces()
        
        atoms.calc = calc2  
        energy2 = atoms.get_potential_energy()
        forces2 = atoms.get_forces()
        
        # Verify reasonable outputs
        assert isinstance(energy1, float), "Energy should be a float"
        assert isinstance(energy2, float), "Energy should be a float"
        assert forces1.shape == (len(atoms), 3), "Forces should have correct shape"
        assert forces2.shape == (len(atoms), 3), "Forces should have correct shape"
        
        # Test head discovery methods
        available_heads = calc1.get_available_heads()
        from collections import defaultdict
        assert isinstance(available_heads, (list, defaultdict)), f"Available heads should be a list or defaultdict, got {type(available_heads)}"

        energy_heads = calc1.list_available_heads_for_property("energy")
        assert isinstance(energy_heads, (list, defaultdict)), f"Energy heads should be a list or defaultdict, got {type(energy_heads)}"
    except Exception as e:
        pytest.fail(f"ASE calculator ensemble test failed: {e}")


def test_ensemble_uncertainty():
    """Test uncertainty quantification with mock ensemble."""
    
    try:
        # Create mock ensemble predictions
        mock_energy_predictions = {
            "energy_head_0": torch.tensor([1.0]),
            "energy_head_1": torch.tensor([1.05]), 
            "energy_head_2": torch.tensor([0.95]),
            "energy_head_3": torch.tensor([1.02]),
            "energy_head_4": torch.tensor([0.98])
        }
        
        # Compute ensemble statistics
        predictions = list(mock_energy_predictions.values())
        stacked = torch.stack(predictions, dim=0)
        
        mean_energy = stacked.mean(dim=0)
        std_energy = stacked.std(dim=0)
        
        # Verify uncertainty is reasonable
        assert std_energy.item() > 0, "Standard deviation should be positive"
        assert std_energy.item() < 0.5, "Standard deviation should be reasonable"
        
        # Check ensemble mean is reasonable
        expected_mean = sum(p.item() for p in predictions) / len(predictions)
        assert abs(mean_energy.item() - expected_mean) < 1e-6, "Mean calculation should be correct"
        
    except Exception as e:
        pytest.fail(f"Uncertainty quantification test failed: {e}")


class TestEnsembleWorkflow:
    """Integration tests for the complete ensemble workflow."""
    
    def test_full_workflow_integration(self):
        """Test that all components work together in a realistic workflow."""
        
        # Get available models
        available_models = pretrained_mlip.available_models
        assert len(available_models) > 0, "No pretrained models available"
        
        # Use uma-s-1p1 if available, otherwise use first available
        checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
        
        # Test basic calculator functionality
        calc = FAIRChemCalculator.from_model_checkpoint(checkpoint_name, task_name="oc20")
        
        # Create test system
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.calc = calc
        
        # Test basic predictions
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Verify outputs are reasonable
        assert isinstance(energy, float), "Energy should be a float"
        assert forces.shape == (len(atoms), 3), "Forces should have correct shape"
        assert not np.isnan(energy), "Energy should not be NaN"
        assert not np.any(np.isnan(forces)), "Forces should not contain NaN"



# Remove the main execution block since pytest will discover and run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
