"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from fairchem.core.models.uma.escn_moe import DatasetSpecificMoEWrapper
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


@patch("fairchem.core.models.uma.escn_moe.registry.get_model_class")
@patch("fairchem.core.models.uma.escn_moe.recursive_replace_all_linear")
def test_dataset_mapping_wrapper_with_mapping(mock_replace, mock_registry):
    """
    Test that DatasetSpecificMoEWrapper with a dataset_mapping dict
    correctly maps datasets to expert indices.
    """
    mock_backbone = MagicMock()
    mock_backbone.regress_stress = False
    mock_backbone.regress_forces = True
    mock_registry.return_value = MagicMock()

    dataset_mapping = {
        "omol": "omol",
        "oc20_subset1": "oc20",
        "oc20_subset2": "oc20",
        "omat": "omat",
        "oc20": "oc20",
    }

    wrapper = DatasetSpecificMoEWrapper(
        backbone=mock_backbone,
        head_cls="some_head",
        dataset_mapping=dataset_mapping,
    )

    expected = {
        "oc20": 0,
        "oc20_subset1": 0,
        "oc20_subset2": 0,
        "omat": 1,
        "omol": 2,
    }
    assert wrapper.dataset_name_to_exp == expected


@patch("fairchem.core.models.uma.escn_moe.registry.get_model_class")
@patch("fairchem.core.models.uma.escn_moe.recursive_replace_all_linear")
def test_dataset_mapping_wrapper_with_deprecated_list(mock_replace, mock_registry):
    """
    Test that DatasetSpecificMoEWrapper with the deprecated dataset_names list
    correctly maps datasets to expert indices.
    """
    mock_backbone = MagicMock()
    mock_backbone.regress_stress = False
    mock_backbone.regress_forces = True
    mock_registry.return_value = MagicMock()

    dataset_names = ["omol", "omat", "oc20"]

    wrapper = DatasetSpecificMoEWrapper(
        backbone=mock_backbone,
        head_cls="some_head",
        dataset_names=dataset_names,
    )

    expected = {
        "oc20": 0,
        "omat": 1,
        "omol": 2,
    }
    assert wrapper.dataset_name_to_exp == expected


if __name__ == "__main__":
    mole1_vs_linear()
