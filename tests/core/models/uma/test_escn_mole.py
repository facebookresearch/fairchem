"""
Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.mole import MOLE, MOLEGlobals


@pytest.mark.gpu()
def test_mole1_vs_linear_gpu():
    mole1_vs_linear("cuda")


def test_mole1_vs_linear_cpu():
    mole1_vs_linear("cpu")


def mole1_vs_linear(device):
    channels = 256

    systems_per_batch = 40

    system_sizes = (torch.rand(systems_per_batch) * 256 + 1).to(torch.int)
    edge_sizes = system_sizes * 4
    expert_embeddings = torch.ones(systems_per_batch, 1).to(device)
    total_edges = sum(edge_sizes)
    x = torch.rand(total_edges, channels).to(device)

    global_mole_tensors = MOLEGlobals(
        expert_mixing_coefficients=expert_embeddings, mole_sizes=edge_sizes
    )

    mole_linear = MOLE(
        num_experts=1,
        in_features=channels,
        out_features=channels,
        global_mole_tensors=global_mole_tensors,
        bias=True,
    ).to(device)

    linear = torch.nn.Linear(
        in_features=channels,
        out_features=channels,
        bias=True,
    ).to(device)

    with torch.no_grad():
        mole_linear.weights[0].copy_(linear.weight)
        mole_linear.bias.copy_(linear.bias)

    mole_output = mole_linear(x.clone())
    linear_output = linear(x.clone())

    assert mole_output.isclose(linear_output, atol=0.0001, rtol=0.001).all()


def test_1mole_merge():
    channels = 256
    device = "cpu"

    systems_per_batch = 1  # merge can only work for one system

    system_sizes = (torch.rand(systems_per_batch) * 256 + 1).to(torch.int)
    edge_sizes = system_sizes * 4
    expert_embeddings = torch.nn.functional.softmax(
        torch.rand(systems_per_batch, 4).to(device), dim=1
    )
    total_edges = sum(edge_sizes)
    x = torch.rand(total_edges, channels).to(device)

    global_mole_tensors = MOLEGlobals(
        expert_mixing_coefficients=expert_embeddings, mole_sizes=edge_sizes
    )

    mole_linear = MOLE(
        num_experts=1,
        in_features=channels,
        out_features=channels,
        global_mole_tensors=global_mole_tensors,
        bias=True,
    ).to(device)

    linear = mole_linear.merged_linear_layer()

    mole_output = mole_linear(x.clone())
    linear_output = linear(x.clone())

    assert mole_output.isclose(linear_output, atol=0.0001, rtol=0.001).all()


def get_moe_backbone(composition_dropout: float = 0.0):
    """Create a minimal eSCNMDMoeBackbone for testing."""
    return eSCNMDMoeBackbone(
        max_num_elements=100,
        sphere_channels=16,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=16,
        num_distance_basis=8,
        use_dataset_embedding=False,
        always_use_pbc=False,
        num_experts=4,
        use_composition_embedding=True,
        composition_dropout=composition_dropout,
    )


class TestCompositionDropout:
    """Tests for composition_dropout feature."""

    def test_dropout_disabled_in_eval_mode(self):
        """Verify dropout has no effect in eval mode - expert coefficients should match."""
        torch.manual_seed(42)
        model_no_dropout = get_moe_backbone(composition_dropout=0.0)
        torch.manual_seed(42)
        model_with_dropout = get_moe_backbone(composition_dropout=0.5)

        # Both models in eval mode
        model_no_dropout.eval()
        model_with_dropout.eval()

        # Create test inputs for set_MOLE_coefficients
        atomic_numbers = torch.tensor([6, 1, 1, 1, 1])  # CH4
        batch = torch.zeros(5, dtype=torch.long)  # single system
        csd_mixed_emb = torch.randn(1, model_no_dropout.sphere_channels)

        # Call set_MOLE_coefficients directly
        model_no_dropout.set_MOLE_coefficients(atomic_numbers, batch, csd_mixed_emb)
        model_with_dropout.set_MOLE_coefficients(atomic_numbers, batch, csd_mixed_emb)

        # In eval mode, expert_mixing_coefficients should be identical
        assert torch.allclose(
            model_no_dropout.global_mole_tensors.expert_mixing_coefficients,
            model_with_dropout.global_mole_tensors.expert_mixing_coefficients,
        )

    def test_dropout_active_in_train_mode(self):
        """Verify dropout causes different expert coefficients in train mode."""
        torch.manual_seed(42)
        model_no_dropout = get_moe_backbone(composition_dropout=0.0)
        torch.manual_seed(42)
        model_with_dropout = get_moe_backbone(composition_dropout=0.5)

        # Both models in train mode
        model_no_dropout.train()
        model_with_dropout.train()

        # Create test inputs for set_MOLE_coefficients
        atomic_numbers = torch.tensor([6, 1, 1, 1, 1])  # CH4
        batch = torch.zeros(5, dtype=torch.long)  # single system
        torch.manual_seed(123)
        csd_mixed_emb = torch.randn(1, model_no_dropout.sphere_channels)

        # Call set_MOLE_coefficients directly
        model_no_dropout.set_MOLE_coefficients(atomic_numbers, batch, csd_mixed_emb)
        model_with_dropout.set_MOLE_coefficients(atomic_numbers, batch, csd_mixed_emb)

        # In train mode with dropout, expert_mixing_coefficients should differ
        assert not torch.allclose(
            model_no_dropout.global_mole_tensors.expert_mixing_coefficients,
            model_with_dropout.global_mole_tensors.expert_mixing_coefficients,
        )

    def test_mask_drops_atoms_statistically(self):
        """Verify mask drops approximately correct percentage of atoms."""
        dropout_rate = 0.5
        n_atoms = 1000
        atomic_numbers = torch.arange(n_atoms)

        torch.manual_seed(42)
        mask = torch.rand_like(atomic_numbers, dtype=torch.float) > dropout_rate
        kept_ratio = mask.sum().item() / n_atoms

        # Should keep approximately 50% (within 10% tolerance)
        assert 0.4 < kept_ratio < 0.6


if __name__ == "__main__":
    mole1_vs_linear()
