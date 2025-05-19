"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
from contextlib import suppress

import torch
import torch.nn as nn

from fairchem.core.models.uma.nn.mole import (
    MOLE,
    MOLEDGL,
    MOLEGlobals,
    norm_str_to_fn,
)
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution

fairchem_cpp_found = False
with suppress(ModuleNotFoundError):
    fairchem_cpp_found = True


class MOLEInterface:
    def set_MOLE_coefficients(
        self, atomic_numbers_full, batch_full, csd_mixed_emb
    ) -> None:
        return None

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index) -> None:
        return None

    def log_MOLE_stats(self) -> None:
        return None

    def merge_MOLE_model(self, data):
        return self


def recursive_replace_so2m0_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2m0_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)


def recursive_replace_so2_MOLE(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2_MOLE(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weights.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)
            for so2_module in child.so2_m_conv:
                so2_module.fc = replacement_factory(so2_module.fc).to(target_device)


def recursive_replace_so2_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)
            for so2_module in child.so2_m_conv:
                so2_module.fc = replacement_factory(so2_module.fc).to(target_device)


def recursive_replace_all_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_all_linear(child, replacement_factory)


def recursive_replace_notso2_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, SO2_Convolution):
            continue
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_notso2_linear(child, replacement_factory)


def model_search_and_replace(
    model, module_search_function, replacement_factory, layers=None
):
    if layers is None:
        layers = list(range(len(model.blocks)))
    for layer_idx in layers:
        module_search_function(model.blocks[layer_idx], replacement_factory)


def replace_linear_with_shared_linear(
    existing_linear_module,
    cache,
):
    layer_identifier = (
        existing_linear_module.in_features,
        existing_linear_module.out_features,
        existing_linear_module.bias is not None,
    )
    if layer_identifier in cache:
        return cache[layer_identifier]

    cache[layer_identifier] = existing_linear_module
    return existing_linear_module


def replace_MOLE_with_linear(
    existing_mole_module: MOLE,
):
    return existing_mole_module.merged_linear_layer()


def replace_linear_with_MOLE(
    existing_linear_module,
    global_mole_tensors,
    num_experts,
    mole_layer_type,
    cache=None,
):
    layer_identifier = (
        existing_linear_module.in_features,
        existing_linear_module.out_features,
        existing_linear_module.bias,
    )
    if cache is not None and layer_identifier in cache:
        return cache[layer_identifier]

    if mole_layer_type == "dgl":
        assert (
            fairchem_cpp_found
        ), "Cannot use DGL layer type if fairchem_cpp package is not available"
        layer = MOLEDGL(
            num_experts=num_experts,
            global_mole_tensors=global_mole_tensors,
            in_features=existing_linear_module.in_features,
            out_features=existing_linear_module.out_features,
            bias=existing_linear_module.bias is not None,
        )
    elif mole_layer_type == "pytorch":
        layer = MOLE(
            num_experts=num_experts,
            global_mole_tensors=global_mole_tensors,
            in_features=existing_linear_module.in_features,
            out_features=existing_linear_module.out_features,
            bias=existing_linear_module.bias is not None,
        )
    else:
        raise ValueError("mole_layer_type must be pytorch")
    if cache is not None:
        cache[layer_identifier] = layer
    return layer


def convert_model_to_MOLE_model(
    model,
    num_experts: int = 8,
    mole_dropout: float = 0.0,
    mole_expert_coefficient_norm: str = "softmax",
    act=torch.nn.SiLU,
    layers_mole=None,
    use_composition_embedding: bool = False,
    mole_layer_type: str = "pytorch",
    mole_single: bool = False,
    mole_type: str = "so2",
):
    model.num_experts = num_experts
    if model.num_experts == 0:
        return

    model.mole_type = mole_type

    routing_mlp_dim = (
        use_composition_embedding + 1  # always use dataset/csd_mixed_emb
    ) * model.sphere_channels

    model.routing_mlp = nn.Sequential(
        nn.Linear(
            routing_mlp_dim,
            num_experts * 2,
            bias=True,
        ),
        nn.SiLU(),
        nn.Linear(
            num_experts * 2,
            num_experts * 2,
            bias=True,
        ),
        nn.SiLU(),
        nn.Linear(
            num_experts * 2,
            num_experts,
            bias=True,
        ),
        nn.SiLU(),
    )

    #
    model.use_composition_embedding = use_composition_embedding
    model.global_mole_tensors = MOLEGlobals(
        expert_mixing_coefficients=None, mole_sizes=None
    )

    model.mole_dropout = torch.nn.Dropout(mole_dropout)
    model.mole_expert_coefficient_norm = norm_str_to_fn(mole_expert_coefficient_norm)
    model.act = act()

    if model.use_composition_embedding:
        model.composition_embedding = nn.Embedding(
            model.max_num_elements, model.sphere_channels
        )

    # plotting
    model.counter = 0

    # replace modules
    replacement_factory = functools.partial(
        replace_linear_with_MOLE,
        num_experts=model.num_experts,
        global_mole_tensors=model.global_mole_tensors,
        mole_layer_type=mole_layer_type,
        cache={} if mole_single else None,
    )

    if mole_type == "so2":
        model_search_and_replace(
            model, recursive_replace_so2_linear, replacement_factory, layers=layers_mole
        )
    elif mole_type == "so2m0":
        model_search_and_replace(
            model,
            recursive_replace_so2m0_linear,
            replacement_factory,
            layers=layers_mole,
        )
    elif mole_type == "all":
        model_search_and_replace(
            model, recursive_replace_all_linear, replacement_factory, layers=layers_mole
        )
    elif mole_type == "notso2":
        model_search_and_replace(
            model,
            recursive_replace_notso2_linear,
            replacement_factory,
            layers=layers_mole,
        )
    else:
        raise ValueError(f"Not a valid mole_type {mole_type}")
