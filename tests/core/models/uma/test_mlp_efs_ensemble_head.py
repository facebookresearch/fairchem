"""
Tests for MLP_EFS_Ensemble_Head class.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from ase.build import bulk
from ase import Atoms

from fairchem.core.models.uma.escn_md import MLP_EFS_Ensemble_Head, eSCNMDBackbone
from fairchem.core.datasets.atomic_data import AtomicData


@pytest.fixture
def mock_backbone():
    """Create a mock backbone for testing."""
    class MockBackbone:
        def __init__(self):
            self.sphere_channels = 128
            self.hidden_channels = 256
            self.regress_stress = False
            self.regress_forces = True
            self.direct_forces = False
            self.energy_block = None
            self.force_block = None
    
    return MockBackbone()


@pytest.fixture
def sample_batch():
    """Create sample batch data for testing."""
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms = atoms * (2, 2, 2)  # 32 atoms
    
    return {
        'pos': torch.randn(32, 3, requires_grad=True),
        'natoms': torch.tensor([32]),
        'batch': torch.zeros(32, dtype=torch.long),
        'pos_original': torch.randn(32, 3, requires_grad=True),
        'cell': torch.eye(3).unsqueeze(0),
    }


@pytest.fixture
def sample_embedding():
    """Create sample node embedding for testing."""
    return {
        'node_embedding': torch.randn(32, 9, 128),  # 32 nodes, 9 l/m features, 128 channels
        'displacement': torch.zeros(1, 3, 3, requires_grad=True),
        'orig_cell': torch.eye(3).unsqueeze(0),
        'batch': torch.zeros(32, dtype=torch.long),
    }


class TestMLPEFSEnsembleHead:
    """Test suite for MLP_EFS_Ensemble_Head."""
    
    def test_initialization(self, mock_backbone):
        """Test that the ensemble head initializes correctly."""
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=5)
        
        # Check basic attributes
        assert head.num_ensemble == 5
        assert head.sphere_channels == 128
        assert head.hidden_channels == 256
        assert head.regress_forces == True
        assert head.regress_stress == False
        assert head.wrap_property == True  # default is True
        
        # Check energy blocks
        assert len(head.energy_blocks) == 5
        for i, block in enumerate(head.energy_blocks):
            assert isinstance(block, nn.Sequential)
            assert len(block) == 5  # 3 Linear + 2 SiLU layers
    
    def test_initialization_custom_params(self, mock_backbone):
        """Test initialization with custom parameters."""
        head = MLP_EFS_Ensemble_Head(
            mock_backbone, 
            num_ensemble=3, 
            prefix="test", 
            wrap_property=False
        )
        
        assert head.num_ensemble == 3
        assert head.prefix == "test"
        assert head.wrap_property == False
        assert len(head.energy_blocks) == 3
    
    def test_forces_only_forward(self, mock_backbone, sample_batch, sample_embedding):
        """Test forward pass for forces-only prediction."""
        mock_backbone.regress_stress = False
        mock_backbone.regress_forces = True
        
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=3, wrap_property=True)
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Check that energy and forces outputs exist
        assert 'energy' in outputs
        assert 'forces' in outputs
        
        # Check ensemble structure
        energy_outputs = outputs['energy']
        forces_outputs = outputs['forces']
        
        assert len(energy_outputs) == 3  # 3 ensemble heads
        assert len(forces_outputs) == 3  # 3 ensemble heads
        
        expected_head_names = ['energyandforcehead1', 'energyandforcehead2', 'energyandforcehead3']
        for head_name in expected_head_names:
            assert head_name in energy_outputs
            assert head_name in forces_outputs
            
            # Check wrapped structure
            assert 'energy' in energy_outputs[head_name]
            assert 'forces' in forces_outputs[head_name]
            
            # Check tensor shapes
            energy = energy_outputs[head_name]['energy']
            forces = forces_outputs[head_name]['forces']
            
            assert energy.shape == (1,)  # 1 system
            assert forces.shape == (32, 3)  # 32 atoms x 3 dimensions
    
    def test_stress_forward(self, mock_backbone, sample_batch, sample_embedding):
        """Test forward pass for stress prediction."""
        mock_backbone.regress_stress = True
        mock_backbone.regress_forces = True
        
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=2, wrap_property=True)
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Check that all outputs exist
        assert 'energy' in outputs
        assert 'forces' in outputs
        assert 'stress' in outputs
        
        # Check ensemble structure
        stress_outputs = outputs['stress']
        assert len(stress_outputs) == 2  # 2 ensemble heads
        
        expected_head_names = ['energyandforcehead1', 'energyandforcehead2']
        for head_name in expected_head_names:
            assert head_name in stress_outputs
            assert 'stress' in stress_outputs[head_name]
            
            # Check tensor shape
            stress = stress_outputs[head_name]['stress']
            assert stress.shape == (1, 9)  # 1 system x 9 stress components
    
    def test_unwrapped_output(self, mock_backbone, sample_batch, sample_embedding):
        """Test forward pass with unwrapped property output."""
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=2, wrap_property=False)
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Check unwrapped structure
        expected_keys = [
            'energy_energyandforcehead1', 'energy_energyandforcehead2',
            'forces_energyandforcehead1', 'forces_energyandforcehead2'
        ]
        
        for key in expected_keys:
            assert key in outputs
            assert isinstance(outputs[key], torch.Tensor)
    
    def test_prefix_output(self, mock_backbone, sample_batch, sample_embedding):
        """Test forward pass with prefix."""
        head = MLP_EFS_Ensemble_Head(
            mock_backbone, 
            num_ensemble=2, 
            prefix="test", 
            wrap_property=True
        )
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Check prefixed keys
        assert 'test_energy' in outputs
        assert 'test_forces' in outputs
        
        # Check ensemble structure within prefixed keys
        energy_outputs = outputs['test_energy']
        forces_outputs = outputs['test_forces']
        
        expected_head_names = ['energyandforcehead1', 'energyandforcehead2']
        for head_name in expected_head_names:
            assert head_name in energy_outputs
            assert head_name in forces_outputs
    
    def test_gradient_computation_efficiency(self, mock_backbone, sample_batch, sample_embedding):
        """Test that gradients are computed for each ensemble member separately."""
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=5)
        
        # Count the number of gradient computations
        original_grad = torch.autograd.grad
        grad_calls = []
        
        def counting_grad(*args, **kwargs):
            grad_calls.append((args, kwargs))
            return original_grad(*args, **kwargs)
        
        torch.autograd.grad = counting_grad
        
        try:
            with torch.enable_grad():
                outputs = head.forward(sample_batch, sample_embedding)
            
            # Should have exactly 5 gradient calls, one for each ensemble member
            # This ensures separate forces/gradients for each ensemble member
            assert len(grad_calls) == 5
            print(f"Number of gradient calls: {len(grad_calls)}")
            
        finally:
            # Restore original function
            torch.autograd.grad = original_grad
    
    def test_ensemble_predictions_differ(self, mock_backbone, sample_batch, sample_embedding):
        """Test that different ensemble heads produce different predictions."""
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=3, wrap_property=True)
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Get energy predictions from different heads
        energy_outputs = outputs['energy']
        energies = [
            energy_outputs['energyandforcehead1']['energy'],
            energy_outputs['energyandforcehead2']['energy'], 
            energy_outputs['energyandforcehead3']['energy']
        ]
        
        # Check that predictions are different (with high probability)
        # Due to random initialization, they should be different
        assert not torch.allclose(energies[0], energies[1], atol=1e-6)
        assert not torch.allclose(energies[1], energies[2], atol=1e-6)
        assert not torch.allclose(energies[0], energies[2], atol=1e-6)
    
    def test_backward_compatibility_with_mlip_unit(self, mock_backbone, sample_batch, sample_embedding):
        """Test that the output format is compatible with mlip_unit expectations."""
        head = MLP_EFS_Ensemble_Head(mock_backbone, num_ensemble=5, wrap_property=True)
        
        with torch.enable_grad():
            outputs = head.forward(sample_batch, sample_embedding)
        
        # Check that outputs match expected structure for mlip_unit
        # Energy structure: outputs['energy'][headname]['energy']
        # Forces structure: outputs['forces'][headname]['forces']
        
        energy_outputs = outputs['energy']
        forces_outputs = outputs['forces']
        
        for i in range(1, 6):  # 1-indexed head names
            head_name = f'energyandforcehead{i}'
            
            # Check nested structure
            assert head_name in energy_outputs
            assert head_name in forces_outputs
            assert isinstance(energy_outputs[head_name], dict)
            assert isinstance(forces_outputs[head_name], dict)
            assert 'energy' in energy_outputs[head_name]
            assert 'forces' in forces_outputs[head_name]
            
            # Check that these are tensors
            assert isinstance(energy_outputs[head_name]['energy'], torch.Tensor)
            assert isinstance(forces_outputs[head_name]['forces'], torch.Tensor)


@pytest.mark.gpu()
def test_ensemble_head_gpu():
    """Test ensemble head on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    from fairchem.core.models.uma.escn_md import MLP_EFS_Ensemble_Head
    
    class MockBackbone:
        def __init__(self):
            self.sphere_channels = 128
            self.hidden_channels = 256
            self.regress_stress = False
            self.regress_forces = True
            self.direct_forces = False
            self.energy_block = None
            self.force_block = None
    
    backbone = MockBackbone()
    head = MLP_EFS_Ensemble_Head(backbone, num_ensemble=3).cuda()
    
    # Create GPU tensors
    sample_batch = {
        'pos': torch.randn(16, 3, requires_grad=True, device='cuda'),
        'natoms': torch.tensor([16], device='cuda'),
        'batch': torch.zeros(16, dtype=torch.long, device='cuda'),
        'pos_original': torch.randn(16, 3, requires_grad=True, device='cuda'),
        'cell': torch.eye(3).unsqueeze(0).cuda(),
    }
    
    sample_embedding = {
        'node_embedding': torch.randn(16, 9, 128, device='cuda'),
        'displacement': torch.zeros(1, 3, 3, requires_grad=True, device='cuda'),
        'orig_cell': torch.eye(3).unsqueeze(0).cuda(),
        'batch': torch.zeros(16, dtype=torch.long, device='cuda'),
    }
    
    with torch.enable_grad():
        outputs = head.forward(sample_batch, sample_embedding)
    
    # Check that outputs are on GPU
    for head_name in ['energyandforcehead1', 'energyandforcehead2', 'energyandforcehead3']:
        energy = outputs['energy'][head_name]['energy']
        forces = outputs['forces'][head_name]['forces']
        assert energy.device.type == 'cuda'
        assert forces.device.type == 'cuda'
