"""
Minimal test to validate ensemble fine-tuning workflow.
This script tests creating an ensemble model with 5 heads.
"""

import pytest
import torch
from ase.build import bulk

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model
from fairchem.core import pretrained_mlip


@pytest.mark.slow
def test_ensemble_fine_tuning_workflow():
    """Test the complete ensemble fine-tuning workflow."""
    
    available_models = pretrained_mlip.available_models

    # Use uma-s-1p1 if available
    checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
    
    # Create 5-head configuration
    heads_config = {
        f"energy_head_{i}": {
            "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
            "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
            "dataset_names": ["omat"],
            "wrap_property": False
        } for i in range(5)
    }
    
    # Get checkpoint path
    from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS
    from huggingface_hub import hf_hub_download
    
    # Get checkpoint path similar to how get_predict_unit does it
    model_checkpoint = _MODEL_CKPTS.checkpoints[checkpoint_name]
    checkpoint_path = hf_hub_download(
        filename=model_checkpoint.filename,
        repo_id=model_checkpoint.repo_id,
        subfolder=model_checkpoint.subfolder,
        revision=model_checkpoint.revision,
    )
    
    # Initialize model with 5 heads
    model = initialize_finetuning_model(
        checkpoint_location=checkpoint_path,
        heads=heads_config
    )
    
    # Verify model has 5 heads
    assert len(model.output_heads) == 5
    assert hasattr(model, 'backbone')
    
    # Test that all heads are present
    expected_heads = [f"energy_head_{i}" for i in range(5)]
    actual_heads = list(model.output_heads.keys())
    
    for expected_head in expected_heads:
        assert expected_head in actual_heads, f"Missing head: {expected_head}"
    
    print(f"✓ Successfully created ensemble model with {len(model.output_heads)} heads")


def test_ase_calculator_basic_functionality():
    """Test basic ASE calculator functionality."""
    
    available_models = pretrained_mlip.available_models

    checkpoint_name = "uma-s-1p1" if "uma-s-1p1" in available_models else list(available_models.keys())[0]
    
    # Create calculator
    calc = FAIRChemCalculator.from_model_checkpoint(checkpoint_name, task_name="oc20")
    
    # Create test system  
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.calc = calc
    
    # Test basic predictions
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Verify outputs
    assert isinstance(energy, float)
    assert forces.shape == (len(atoms), 3)
    assert not torch.isnan(torch.tensor(energy))
    assert not torch.any(torch.isnan(torch.tensor(forces)))
    
    print(f"✓ ASE calculator works: E={energy:.3f} eV, F_shape={forces.shape}")


def test_uncertainty_quantification():
    """Test uncertainty quantification with mock ensemble predictions."""
    
    # Create mock ensemble predictions
    mock_predictions = {
        f"head_{i}": torch.tensor([1.0 + 0.1*i - 0.05*i**2]) for i in range(5)
    }
    
    # Compute ensemble statistics
    predictions = list(mock_predictions.values())
    stacked = torch.stack(predictions, dim=0)
    
    mean_pred = stacked.mean(dim=0)
    std_pred = stacked.std(dim=0)
    
    # Verify reasonable statistics
    assert std_pred.item() > 0, "Standard deviation should be positive"
    assert mean_pred.item() > 0, "Mean should be reasonable"
    
    # Test confidence intervals
    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred
    
    assert upper_bound > lower_bound, "Confidence interval should be valid"
    
    print(f"✓ Uncertainty quantification: μ={mean_pred.item():.3f}, σ={std_pred.item():.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
