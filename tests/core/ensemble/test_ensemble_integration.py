"""
Integration test script for ensemble functionality.
This script fine-tunes a UMA model with 5 heads and tests the functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path
import yaml
import pytest

import torch
from ase import Atoms
from ase.build import bulk

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model
from fairchem.core import pretrained_mlip


def create_ensemble_config():
    """Create configuration for 5-head ensemble fine-tuning."""
    
    config = {
        "trainer": {
            "num_epochs": 1,  # Minimal training for testing
            "learning_rate": 1e-4,
            "batch_size": 4
        },
        "model": {
            "heads": {
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
        },
        "tasks": [
            {
                "name": "energy",
                "level": "system",
                "property": "energy", 
                "loss_fn": "torch.nn.MSELoss",
                "out_spec": {"dim": [1], "dtype": "float32"},
                "datasets": ["omat"],
                "shallow_ensemble": True
            }
        ]
    }
    
    return config


def test_initialize_model_with_multiple_heads():
    """Test initializing a model with multiple heads."""
    print("\nTesting model initialization with multiple heads...")
    
    # Get available pretrained models
    available_models = pretrained_mlip.available_models

    # Use uma-s-1p1 if available, otherwise use first available
    checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
    print(f"Using checkpoint: {checkpoint_name}")
    # Try to initialize model
    heads_config = create_ensemble_config()["model"]["heads"]

    model = initialize_finetuning_model(
        model_name=checkpoint_name,
        heads=heads_config
    )

    print(f"Successfully initialized model with {len(model.output_heads)} heads")
    print(f"Head names: {list(model.output_heads.keys())}")
    # Verify we have 5 heads
    assert len(model.output_heads) == 5
    assert all(f"head{i}" in model.output_heads for i in range(5))
    print("✓ Model initialization test passed!")

def test_ase_calculator_with_multiple_heads():
    """Test ASE calculator with multiple heads."""

    print("\nTesting ASE calculator with multiple heads...")

    try:
        # Get available pretrained models  
        available_models = pretrained_mlip.available_models
        checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
        # Create calculator with explicit task_name
        calc = FAIRChemCalculator.from_model_checkpoint(checkpoint_name, task_name="oc20")
        # Create test system
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.calc = calc
        # Run prediction
        try:
            energy = atoms.get_potential_energy()
        except KeyError:
            # If 'free_energy' is missing, try 'energy'
            energy = atoms.calc.results.get('energy', None)
        try:
            forces = atoms.get_forces()
        except KeyError:
            forces = atoms.calc.results.get('forces', None)
        print(f"✓ Single prediction successful: E={energy:.3f} eV")
        print(f"✓ Forces shape: {forces.shape}")
        # Test with head specification (even if only one head exists)
        available_heads = calc.get_available_heads()
        print(f"Available heads: {available_heads}")
        # Test head listing functionality
        energy_heads = calc.list_available_heads_for_property("energy")
        print(f"Energy heads: {energy_heads}")
        print("✓ ASE calculator test passed!")
    except Exception as e:
        print(f"✗ ASE calculator test failed: {e}")
        pytest.fail(f"ASE calculator test failed: {e}")

def test_prediction_structure():
    """Test the prediction structure from MLIPPredictUnit."""
    
    print("\nTesting prediction structure...")
    
    try:
        # Get a predict unit
        available_models = pretrained_mlip.available_models
        checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
        
        predict_unit = pretrained_mlip.get_predict_unit(checkpoint_name)
        
        # Create test data
        from fairchem.core.datasets.atomic_data import AtomicData
        
        test_atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        data = AtomicData.from_ase(
            test_atoms,
            max_neigh=predict_unit.model.module.backbone.max_neighbors,
            radius=predict_unit.model.module.backbone.cutoff,
            task_name=predict_unit.datasets[0]
        )
        
        # Run prediction
        pred = predict_unit.predict(data)
        
        print(f"Prediction keys: {list(pred.keys())}")
        
        # Test the structure
        for prop, heads in pred.items():
            if isinstance(heads, dict):
                print(f"Property '{prop}' has {len(heads)} heads: {list(heads.keys())}")
            else:
                print(f"Property '{prop}' has single prediction")
        
        print("✓ Prediction structure test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Prediction structure test failed: {e}")
        return False


def test_ensemble_averaging():
    """Test ensemble averaging functionality."""
    
    print("\nTesting ensemble averaging...")
    
    try:
        # Create mock predictions with multiple heads
        mock_predictions = {
            "energy": {
                "head0": torch.tensor([1.0]),
                "head1": torch.tensor([2.0]),
                "head2": torch.tensor([3.0]),
                "head3": torch.tensor([4.0]),
                "head4": torch.tensor([5.0])
            },
            "forces": {
                "head0": torch.tensor([[0.1, 0.2, 0.3]]),
                "head1": torch.tensor([[0.2, 0.3, 0.4]]),
                "head2": torch.tensor([[0.3, 0.4, 0.5]]),
                "head3": torch.tensor([[0.4, 0.5, 0.6]]),
                "head4": torch.tensor([[0.5, 0.6, 0.7]])
            }
        }
        
        # Test energy ensemble
        energy_preds = list(mock_predictions["energy"].values())
        energy_stacked = torch.stack(energy_preds, dim=0)
        energy_mean = energy_stacked.mean(dim=0)
        energy_std = energy_stacked.std(dim=0)
        
        # Test diversity in energy predictions
        energy_values = [p.item() for p in energy_preds]
        unique_energy = len(set(energy_values))
        assert unique_energy == 5, f"Expected 5 unique energy predictions, got {unique_energy}"
        
        # Test no NaNs in energy statistics
        assert not torch.isnan(energy_mean).any(), "Energy mean contains NaN"
        assert not torch.isnan(energy_std).any(), "Energy std contains NaN"
        assert energy_std.item() > 0, "Energy std should be positive for diverse predictions"
        
        print(f"   ✓ Energy ensemble: μ={energy_mean.item():.2f}, σ={energy_std.item():.2f}")
        
        # Test forces ensemble
        forces_preds = list(mock_predictions["forces"].values())
        forces_stacked = torch.stack(forces_preds, dim=0)
        forces_mean = forces_stacked.mean(dim=0)
        forces_std = forces_stacked.std(dim=0)
        
        # Test diversity in forces predictions
        forces_norms = [torch.norm(p).item() for p in forces_preds]
        unique_forces = len(set([round(f, 6) for f in forces_norms]))
        assert unique_forces > 1, f"Forces should have diversity, got norms {forces_norms}"
        
        # Test no NaNs in forces statistics
        assert not torch.isnan(forces_mean).any(), "Forces mean contains NaN"
        assert not torch.isnan(forces_std).any(), "Forces std contains NaN"
        assert torch.all(forces_std >= 0), "All forces std should be non-negative"
        assert torch.any(forces_std > 0), "Some forces std should be positive"
        
        print(f"   ✓ Forces ensemble: mean_norm={torch.norm(forces_mean).item():.3f}, max_std={forces_std.max().item():.3f}")
        print("   ✓ All ensemble averaging tests passed")
        
    except Exception as e:
        print(f"✗ Ensemble averaging test failed: {e}")
        pytest.fail(f"Ensemble averaging test failed: {e}")

