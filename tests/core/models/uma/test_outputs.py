"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    get_l_component_range,
    reduce_node_to_system,
)


class TestGetLComponentRange:
    """Tests for get_l_component_range function."""

    def test_l0_extraction(self):
        """Test extraction of L=0 (scalar) component."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=0, l_max=0)

        assert result.shape == (N, 1, C)
        assert torch.allclose(result, x[:, 0:1, :])

    def test_l1_extraction(self):
        """Test extraction of L=1 (vector) component."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=1, l_max=1)

        assert result.shape == (N, 3, C)
        assert torch.allclose(result, x[:, 1:4, :])

    def test_l2_extraction(self):
        """Test extraction of L=2 (rank-2 traceless symmetric) component."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=2, l_max=2)

        assert result.shape == (N, 5, C)
        assert torch.allclose(result, x[:, 4:9, :])

    def test_l3_extraction(self):
        """Test extraction of L=3 component from larger tensor."""
        N, C = 5, 8
        x = torch.randn(N, 16, C)

        result = get_l_component_range(x, l_min=3, l_max=3)

        # L=3 starts at index 9 (= 3^2) and has 7 components (= 2*3+1)
        assert result.shape == (N, 7, C)
        assert torch.allclose(result, x[:, 9:16, :])

    def test_range_l0_to_l1(self):
        """Test extraction of L=0 through L=1 components."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=0, l_max=1)

        # L=0 (1 component) + L=1 (3 components) = 4 components
        assert result.shape == (N, 4, C)
        assert torch.allclose(result, x[:, 0:4, :])

    def test_range_l1_to_l2(self):
        """Test extraction of L=1 through L=2 components."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=1, l_max=2)

        # L=1 (3 components) + L=2 (5 components) = 8 components
        assert result.shape == (N, 8, C)
        assert torch.allclose(result, x[:, 1:9, :])

    def test_range_l0_to_l2(self):
        """Test extraction of all components up to L=2."""
        N, C = 5, 8
        x = torch.randn(N, 9, C)

        result = get_l_component_range(x, l_min=0, l_max=2)

        assert result.shape == (N, 9, C)
        assert torch.allclose(result, x)

    @pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
    def test_single_l_size_formula(self, l):
        """Test that a single-L extraction has size 2L+1."""
        N, C = 3, 4
        x = torch.randn(N, (l + 1) ** 2, C)

        result = get_l_component_range(x, l_min=l, l_max=l)

        assert result.shape[1] == 2 * l + 1

    @pytest.mark.parametrize("l_min,l_max", [(0, 1), (0, 2), (1, 3), (2, 4)])
    def test_range_size_formula(self, l_min, l_max):
        """Test that a range extraction has size (l_max+1)^2 - l_min^2."""
        N, C = 3, 4
        x = torch.randn(N, (l_max + 1) ** 2, C)

        result = get_l_component_range(x, l_min=l_min, l_max=l_max)

        expected_size = (l_max + 1) ** 2 - l_min**2
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


def _make_emb_and_block(node_energies: torch.Tensor) -> tuple:
    """Helper to build a minimal emb dict and identity energy_block from 1D node energies.

    node_energies: 1D tensor of shape [N] representing per-node scalar energy values.
    Returns (emb, energy_block) where emb["node_embedding"] has shape [N, 1, 1].
    """
    node_embedding = node_energies.view(-1, 1, 1)
    return {"node_embedding": node_embedding}, nn.Identity()


class TestComputeEnergy:
    """Tests for compute_energy function."""

    def test_single_system(self):
        """Test energy computation for a single system."""
        emb, energy_block = _make_emb_and_block(torch.tensor([0.5, 1.0, 1.5]))
        batch = torch.tensor([0, 0, 0])

        energy, energy_part = compute_energy(emb, energy_block, batch, num_systems=1)

        assert energy.shape == (1,)
        assert torch.allclose(energy, torch.tensor([3.0]))

    def test_multiple_systems(self):
        """Test energy computation for multiple systems."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        batch = torch.tensor([0, 0, 1, 1])

        energy, energy_part = compute_energy(emb, energy_block, batch, num_systems=2)

        assert energy.shape == (2,)
        assert torch.allclose(energy, torch.tensor([3.0, 7.0]))

    def test_node_energy_flattening(self):
        """Test that energy_block output [N, 1] is properly flattened to [N]."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 2.0, 3.0]))
        batch = torch.tensor([0, 0, 0])

        energy, _ = compute_energy(emb, energy_block, batch, num_systems=1)

        assert energy.shape == (1,)
        assert torch.allclose(energy, torch.tensor([6.0]))

    def test_energy_part_for_gradients(self):
        """Test that energy_part can be used for gradient computation."""
        node_embedding = torch.tensor([[[1.0]], [[2.0]]], requires_grad=True)
        emb = {"node_embedding": node_embedding}
        energy_block = nn.Identity()
        batch = torch.tensor([0, 0])

        energy, energy_part = compute_energy(emb, energy_block, batch, num_systems=1)

        # energy_part should allow gradient computation
        loss = energy_part.sum()
        loss.backward()

        assert node_embedding.grad is not None
        assert torch.allclose(node_embedding.grad, torch.ones_like(node_embedding))

    def test_reduce_mean(self):
        """Test that reduce='mean' divides energy by natoms per system."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 3.0, 2.0, 6.0]))
        batch = torch.tensor([0, 0, 1, 1])
        natoms = torch.tensor([2, 2])

        energy, _ = compute_energy(
            emb, energy_block, batch, num_systems=2, natoms=natoms, reduce="mean"
        )

        assert energy.shape == (2,)
        assert torch.allclose(energy, torch.tensor([2.0, 4.0]))

    def test_reduce_sum_is_default(self):
        """Test that reduce defaults to 'sum'."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 2.0, 3.0]))
        batch = torch.tensor([0, 0, 0])

        energy_default, _ = compute_energy(emb, energy_block, batch, num_systems=1)
        energy_sum, _ = compute_energy(
            emb, energy_block, batch, num_systems=1, reduce="sum"
        )

        assert torch.allclose(energy_default, energy_sum)

    def test_reduce_mean_requires_natoms(self):
        """Test that reduce='mean' raises when natoms is not provided."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 2.0]))
        batch = torch.tensor([0, 0])

        with pytest.raises(ValueError, match="natoms must be provided"):
            compute_energy(emb, energy_block, batch, num_systems=1, reduce="mean")

    def test_reduce_invalid(self):
        """Test that an invalid reduce value raises ValueError."""
        emb, energy_block = _make_emb_and_block(torch.tensor([1.0, 2.0]))
        batch = torch.tensor([0, 0])
        natoms = torch.tensor([2])

        with pytest.raises(ValueError, match="reduce can only be sum or mean"):
            compute_energy(
                emb, energy_block, batch, num_systems=1, natoms=natoms, reduce="max"
            )


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
        """Test that forces are computed correctly from energy gradient."""
        pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        cell = torch.eye(3).unsqueeze(0) * 4.0
        cell.requires_grad = True
        batch = torch.tensor([0, 0])

        # Simple energy: sum of positions
        energy_part = pos.sum().unsqueeze(0) + cell.sum().unsqueeze(0)

        forces, stress = compute_forces_and_stress(
            energy_part, pos, cell, batch, training=True
        )

        # Forces = -gradient = -1 for all components
        expected_forces = -torch.ones(2, 3)
        assert forces.shape == (2, 3)
        assert torch.allclose(forces, expected_forces, atol=1e-6)

    def test_stress_shape(self):
        """Test that stress has correct shape [num_systems, 9]."""
        pos = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
        batch = torch.tensor([0, 1])  # 2 systems, 1 atom each
        cell = torch.eye(3).unsqueeze(0).expand(2, 3, 3).contiguous() * 4.0
        cell.requires_grad = True

        # Energy per system
        energy_part = torch.zeros(2)
        energy_part = energy_part.index_add(
            0, batch, pos.sum(dim=1) + cell.sum(dim=(1, 2))
        )

        forces, stress = compute_forces_and_stress(
            energy_part, pos, cell, batch, training=True
        )

        assert stress.shape == (2, 9)

    def test_multiple_systems(self):
        """Test stress computation with multiple systems."""
        # 4 atoms: 2 in system 0, 2 in system 1
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            requires_grad=True,
        )
        batch = torch.tensor([0, 0, 1, 1])
        cell = torch.eye(3).unsqueeze(0).expand(2, 3, 3).contiguous() * 5.0
        cell.requires_grad = True

        # Energy: sum of positions per system + sum of cell per system
        energy_part = torch.zeros(2)
        energy_part = energy_part.index_add(
            0, batch, pos.sum(dim=1) + cell.sum(dim=(1, 2))[batch]
        )

        forces, stress = compute_forces_and_stress(
            energy_part, pos, cell, batch, training=True
        )

        assert forces.shape == (4, 3)
        assert stress.shape == (2, 9)

    def test_stress_symmetry(self):
        """Test that stress tensor is symmetric."""
        pos = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], requires_grad=True)
        batch = torch.tensor([0, 0])
        cell = torch.tensor([[[3.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 3.0]]])
        cell.requires_grad = True

        # Energy depending on positions and cell
        energy_part = (pos.pow(2).sum() + cell.pow(2).sum()).unsqueeze(0)

        _, stress = compute_forces_and_stress(
            energy_part, pos, cell, batch, training=True
        )

        # Reshape to 3x3 and check symmetry
        stress_matrix = stress.view(3, 3)
        assert torch.allclose(stress_matrix, stress_matrix.T, atol=1e-6)
