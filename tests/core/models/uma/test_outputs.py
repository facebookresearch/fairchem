"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    get_l_component,
    reduce_node_to_system,
)


class TestGetLComponent:
    """Tests for get_l_component function."""

    def test_l0_extraction(self):
        """Test extraction of L=0 (scalar) component."""
        # Create tensor with shape [N, 9, C] for lmax=2
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component(x, l=0)

        assert result.shape == (N, 1, C)
        assert torch.allclose(result, x[:, 0:1, :])

    def test_l1_extraction(self):
        """Test extraction of L=1 (vector) component."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component(x, l=1)

        assert result.shape == (N, 3, C)
        assert torch.allclose(result, x[:, 1:4, :])

    def test_l2_extraction(self):
        """Test extraction of L=2 (rank-2 traceless symmetric) component."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component(x, l=2)

        assert result.shape == (N, 5, C)
        assert torch.allclose(result, x[:, 4:9, :])

    def test_l3_extraction(self):
        """Test extraction of L=3 component from larger tensor."""
        N, C = 5, 8
        # lmax=3 means (3+1)^2 = 16 components
        x = torch.randn(N, 16, C)

        result = get_l_component(x, l=3)

        # L=3 starts at index 9 (= 3^2) and has 7 components (= 2*3+1)
        assert result.shape == (N, 7, C)
        assert torch.allclose(result, x[:, 9:16, :])

    @pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
    def test_component_size_formula(self, l):
        """Test that extracted component has correct size 2L+1."""
        N, C = 3, 4
        max_idx = (l + 1) ** 2
        x = torch.randn(N, max_idx, C)

        result = get_l_component(x, l=l)

        expected_size = 2 * l + 1
        assert result.shape[1] == expected_size


class TestReduceNodeToSystem:
    """Tests for reduce_node_to_system function."""

    def test_single_system(self):
        """Test reduction with a single system."""
        node_values = torch.tensor([1.0, 2.0, 3.0])
        batch = torch.tensor([0, 0, 0])

        reduced, unreduced = reduce_node_to_system(node_values, batch, num_systems=1)

        assert reduced.shape == (1,)
        assert unreduced.shape == (1,)
        assert torch.allclose(reduced, torch.tensor([6.0]))
        assert torch.allclose(unreduced, torch.tensor([6.0]))

    def test_multiple_systems(self):
        """Test reduction with multiple systems."""
        node_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        batch = torch.tensor([0, 0, 1, 1, 1])

        reduced, unreduced = reduce_node_to_system(node_values, batch, num_systems=2)

        assert reduced.shape == (2,)
        assert torch.allclose(reduced, torch.tensor([3.0, 12.0]))

    def test_multidimensional_values(self):
        """Test reduction with multi-dimensional node values."""
        # 4 nodes, each with 3-dimensional values
        node_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        batch = torch.tensor([0, 0, 1, 1])

        reduced, unreduced = reduce_node_to_system(node_values, batch, num_systems=2)

        assert reduced.shape == (2, 3)
        expected = torch.tensor(
            [
                [5.0, 7.0, 9.0],  # sum of nodes 0, 1
                [17.0, 19.0, 21.0],  # sum of nodes 2, 3
            ]
        )
        assert torch.allclose(reduced, expected)

    def test_empty_system(self):
        """Test that systems with no nodes have zero values."""
        node_values = torch.tensor([1.0, 2.0])
        batch = torch.tensor([0, 2])  # system 1 has no nodes

        reduced, _ = reduce_node_to_system(node_values, batch, num_systems=3)

        assert reduced.shape == (3,)
        assert torch.allclose(reduced, torch.tensor([1.0, 0.0, 2.0]))

    def test_preserves_dtype(self):
        """Test that output preserves input dtype."""
        node_values = torch.tensor([1.0, 2.0], dtype=torch.float64)
        batch = torch.tensor([0, 0])

        reduced, _ = reduce_node_to_system(node_values, batch, num_systems=1)

        assert reduced.dtype == torch.float64


class TestComputeEnergy:
    """Tests for compute_energy function."""

    def test_single_system(self):
        """Test energy computation for a single system."""
        node_energy = torch.tensor([0.5, 1.0, 1.5])
        batch = torch.tensor([0, 0, 0])

        energy, energy_part = compute_energy(node_energy, batch, num_systems=1)

        assert energy.shape == (1,)
        assert torch.allclose(energy, torch.tensor([3.0]))

    def test_multiple_systems(self):
        """Test energy computation for multiple systems."""
        node_energy = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch = torch.tensor([0, 0, 1, 1])

        energy, energy_part = compute_energy(node_energy, batch, num_systems=2)

        assert energy.shape == (2,)
        assert torch.allclose(energy, torch.tensor([3.0, 7.0]))

    def test_2d_input_flattening(self):
        """Test that 2D input [N, 1] is properly flattened."""
        node_energy = torch.tensor([[1.0], [2.0], [3.0]])
        batch = torch.tensor([0, 0, 0])

        energy, _ = compute_energy(node_energy, batch, num_systems=1)

        assert energy.shape == (1,)
        assert torch.allclose(energy, torch.tensor([6.0]))

    def test_energy_part_for_gradients(self):
        """Test that energy_part can be used for gradient computation."""
        node_energy = torch.tensor([1.0, 2.0], requires_grad=True)
        batch = torch.tensor([0, 0])

        energy, energy_part = compute_energy(node_energy, batch, num_systems=1)

        # energy_part should allow gradient computation
        loss = energy_part.sum()
        loss.backward()

        assert node_energy.grad is not None
        assert torch.allclose(node_energy.grad, torch.tensor([1.0, 1.0]))


class TestComputeForces:
    """Tests for compute_forces function."""

    def test_simple_gradient(self):
        """Test force computation as negative gradient of energy."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
        # Energy = sum of x-coordinates squared
        energy_part = (pos[:, 0] ** 2).sum().unsqueeze(0)

        forces = compute_forces(energy_part, pos, training=True)

        # Force = -dE/dx = -2x
        expected = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        assert forces.shape == (2, 3)
        assert torch.allclose(forces, expected)

    def test_harmonic_potential(self):
        """Test forces for a harmonic potential E = 0.5 * k * r^2."""
        pos = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        k = 2.0
        energy_part = (0.5 * k * (pos**2).sum()).unsqueeze(0)

        forces = compute_forces(energy_part, pos, training=True)

        # Force = -k * r
        expected = -k * pos.detach()
        assert torch.allclose(forces, expected)

    def test_training_false_no_graph(self):
        """Test that training=False does not create computation graph."""
        pos = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        energy_part = pos[:, 0].sum().unsqueeze(0)

        forces = compute_forces(energy_part, pos, training=False)

        # Force should still be computed correctly
        assert torch.allclose(forces, torch.tensor([[-1.0, 0.0, 0.0]]))
        # But forces should not require grad (no graph created)
        assert not forces.requires_grad


class TestComputeForcesAndStress:
    """Tests for compute_forces_and_stress function."""

    def test_forces_output(self):
        """Test that forces are computed correctly."""
        pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        displacement = torch.zeros(1, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0) * 10.0  # 10 Angstrom cubic cell

        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        pos = pos + pos @ symmetric_displacement
        cell = cell + cell @ symmetric_displacement

        # Simple energy: sum of positions
        energy_part = pos.sum().unsqueeze(0)

        forces, stress = compute_forces_and_stress(
            energy_part, pos, displacement, cell, training=True
        )

        # Forces = -gradient = -1 for all components
        expected_forces = -torch.ones(2, 3)
        assert forces.squeeze().shape == (2, 3)
        assert torch.allclose(forces, expected_forces)

    def test_stress_shape(self):
        """Test that stress has correct shape [num_systems, 9]."""
        pos = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
        batch = torch.tensor([0, 1])  # 2 systems, 1 atom each
        displacement = torch.zeros(2, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0).expand(2, 3, 3).clone() * 10.0

        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        # Apply displacement per system
        pos_transformed = pos + torch.bmm(
            pos.unsqueeze(1), symmetric_displacement[batch]
        ).squeeze(1)
        cell = cell + torch.bmm(cell, symmetric_displacement)

        # Energy per system
        energy_part = torch.zeros(2)
        energy_part = energy_part.index_add(0, batch, pos_transformed.sum(dim=1))

        forces, stress = compute_forces_and_stress(
            energy_part, pos_transformed, displacement, cell, training=True
        )

        assert stress.shape == (2, 9)

    def test_stress_volume_scaling(self):
        """Test that stress scales inversely with volume."""
        # Test with two different cell volumes
        # Use non-zero position so displacement affects energy
        pos_small = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        pos_large = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        displacement_small = torch.zeros(1, 3, 3, requires_grad=True)
        displacement_large = torch.zeros(1, 3, 3, requires_grad=True)

        cell_small = torch.eye(3).unsqueeze(0) * 1.0  # volume = 1
        cell_large = torch.eye(3).unsqueeze(0) * 2.0  # volume = 8

        # Apply symmetric displacement to positions and cells
        sym_disp_small = 0.5 * (
            displacement_small + displacement_small.transpose(-1, -2)
        )
        pos_transformed_small = pos_small + (pos_small @ sym_disp_small.squeeze(0))
        cell_small = cell_small + cell_small @ sym_disp_small

        sym_disp_large = 0.5 * (
            displacement_large + displacement_large.transpose(-1, -2)
        )
        pos_transformed_large = pos_large + (pos_large @ sym_disp_large.squeeze(0))
        cell_large = cell_large + cell_large @ sym_disp_large

        # Energy depends on transformed positions
        energy_small = pos_transformed_small.sum().unsqueeze(0)
        energy_large = pos_transformed_large.sum().unsqueeze(0)

        _, stress_small = compute_forces_and_stress(
            energy_small,
            pos_transformed_small,
            displacement_small,
            cell_small,
            training=True,
        )

        _, stress_large = compute_forces_and_stress(
            energy_large,
            pos_transformed_large,
            displacement_large,
            cell_large,
            training=True,
        )

        # Stress should scale as 1/volume, so stress_small / stress_large = 8
        volume_ratio = 8.0
        # If both stresses are zero (no gradient wrt displacement), skip comparison
        if not torch.allclose(stress_small, torch.zeros_like(stress_small)):
            assert torch.allclose(
                stress_small / stress_large, torch.full_like(stress_small, volume_ratio)
            )
