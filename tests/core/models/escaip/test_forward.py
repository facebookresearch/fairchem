"""
Modified from tests/core/models/uma/test_compile.py
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.escaip.EScAIP import (
    EScAIPBackbone,
    EScAIPGradientEnergyForceStressHead,
)

MAX_ELEMENTS = 100


def make_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # set before any CUDA init
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_data(num_atoms: int):
    samples = get_fcc_carbon_xtal(num_atoms)
    return AtomicData.from_ase(samples)


def get_backbone_config(
    cutoff: float, use_compile: bool, otf_graph=False, autograd: bool = True
):
    return {
        "regress_stress": True,
        "direct_forces": not autograd,
        "regress_forces": True,
        "hidden_size": 8,
        "activation": "gelu",
        "use_compile": use_compile,
        "use_padding": use_compile,
        "use_pbc": True,
        "max_num_elements": MAX_ELEMENTS,
        "max_atoms": 1000,
        "max_batch_size": 64,
        "max_radius": cutoff,
        "knn_k": 20,
        "knn_soft": True,
        "knn_sigmoid_scale": 0.2,
        "knn_lse_scale": 0.1,
        "knn_use_low_mem": True,
        "knn_pad_size": 30,
        "distance_function": "sigmoid",
        "use_envelope": True,
        "use_angle_embedding": "none",
        "num_layers": 2,
        "atom_embedding_size": 8,
        "node_direction_embedding_size": 8,
        "node_direction_expansion_size": 4,
        "edge_distance_expansion_size": 8,
        "edge_distance_embedding_size": 8,
        "readout_hidden_layer_multiplier": 1,
        "output_hidden_layer_multiplier": 1,
        "ffn_hidden_layer_multiplier": 1,
        "atten_name": "memory_efficient",
        "atten_num_heads": 2,
        "use_frequency_embedding": False,
        "energy_reduce": "sum",
        "normalization": "rmsnorm",
    }


def get_escaip_backbone(
    cutoff: float,
    use_compile: bool,
    otf_graph=False,
    device="cuda",
    autograd: bool = True,
):
    backbone_config = get_backbone_config(
        cutoff=cutoff, use_compile=use_compile, otf_graph=otf_graph, autograd=autograd
    )
    model = EScAIPBackbone(**backbone_config)
    model.to(device)
    model.eval()
    return model


def get_escaip_full(
    cutoff: float,
    use_compile: bool,
    otf_graph=False,
    device="cuda",
    autograd: bool = True,
):
    backbone = get_escaip_backbone(
        cutoff=cutoff,
        use_compile=use_compile,
        otf_graph=otf_graph,
        device=device,
        autograd=autograd,
    )
    heads = {
        "efs_head": EScAIPGradientEnergyForceStressHead(backbone, wrap_property=False)
    }
    model = HydraModelV2(backbone, heads).to(device)
    model.eval()
    return model


@pytest.mark.gpu()
def test_compile_full_gpu():
    # make_deterministic()
    torch.compiler.reset()
    device = "cuda"
    cutoff = 6.0
    model_compile = get_escaip_full(cutoff=cutoff, use_compile=True, device=device)
    model_no_compile = get_escaip_full(cutoff=cutoff, use_compile=False, device=device)
    # copy model parameters from model_compile to model_no_compile
    for param, param_compile in zip(
        model_no_compile.parameters(), model_compile.parameters()
    ):
        param.data = param_compile.data.clone()
    for size in range(3, 10):
        data = get_sample_data(size).to(device)
        seed_everywhere()
        output = model_no_compile(data)["efs_head"]
        seed_everywhere()
        output_compiled = model_compile(data)["efs_head"]
        assert torch.allclose(output["energy"], output_compiled["energy"], atol=1e-5)
        assert torch.allclose(output["forces"], output_compiled["forces"], atol=1e-4)
        assert torch.allclose(output["stress"], output_compiled["stress"], atol=1e-5)


@pytest.mark.gpu()
def test_fixed_forward_full_gpu():
    # make_deterministic()
    torch.compiler.reset()
    device = "cuda"
    cutoff = 6.0
    seed_everywhere()
    # get model
    model = get_escaip_full(cutoff=cutoff, use_compile=False, device=device)
    model.train()
    seed_everywhere()
    # get optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)
    seed_everywhere()
    # get data
    data = get_sample_data(10).to(device)
    seed_everywhere()
    # get output
    output = model(data)["efs_head"]
    seed_everywhere()
    # get loss and backward (dummy loss)
    loss = output["energy"] + output["forces"].sum() + output["stress"].sum()
    loss.backward()
    seed_everywhere()
    optimizer.step()
    seed_everywhere()

    # load fixed results
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixed_results.pt"
    )
    fixed_results = torch.load(results_path)
    # compare fixed_results with output
    model_output = output
    assert torch.allclose(
        fixed_results["model_output"]["energy"], model_output["energy"], atol=1e-5
    )
    assert torch.allclose(
        fixed_results["model_output"]["forces"], model_output["forces"], atol=1e-4
    )
    assert torch.allclose(
        fixed_results["model_output"]["stress"], model_output["stress"], atol=1e-5
    )
