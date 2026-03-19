"""
Modified from tests/core/models/uma/test_compile.py
"""

from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal
from fairchem.core.models.allscaip.AllScAIP import (
    AllScAIPBackbone,
    AllScAIPGradientEnergyForceStressHead,
)
from fairchem.core.models.base import HydraModelV2

MAX_ELEMENTS = 100
DATASET_LIST = ["oc20", "omol", "osc", "omat", "odac"]


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
    data_object = AtomicData.from_ase(samples)
    data_object.natoms = torch.tensor(len(samples))
    data_object.charge = torch.LongTensor([0])
    data_object.spin = torch.LongTensor([0])
    data_object.dataset = "omol"
    data_object.pos.requires_grad = True
    data_loader = torch.utils.data.DataLoader(
        [data_object],
        collate_fn=partial(data_list_collater, otf_graph=True),
        batch_size=1,
        shuffle=False,
    )
    return next(iter(data_loader))


def get_backbone_config(
    cutoff: float, use_compile: bool, otf_graph=False, autograd: bool = True
):
    return {
        "regress_stress": True,
        "direct_forces": not autograd,
        "regress_forces": True,
        "hidden_size": 8,
        "dataset_list": DATASET_LIST,
        "use_compile": use_compile,
        "use_padding": use_compile,
        "max_num_elements": MAX_ELEMENTS,
        "max_atoms": 30,
        "max_batch_size": 8,
        "max_radius": cutoff,
        "knn_k": 20,
        "knn_pad_size": 30,
        "num_layers": 2,
        "atten_name": "memory_efficient",
        "atten_num_heads": 2,
        "freequency_list": [2, 2],
        "use_freq_mask": True,
        "use_sincx_mask": True,
    }


def get_allscaip_backbone(
    cutoff: float,
    use_compile: bool,
    otf_graph=False,
    device="cuda",
    autograd: bool = True,
):
    backbone_config = get_backbone_config(
        cutoff=cutoff, use_compile=use_compile, otf_graph=otf_graph, autograd=autograd
    )
    model = AllScAIPBackbone(**backbone_config)
    model.to(device)
    model.eval()
    return model


def get_allscaip_full(
    cutoff: float,
    use_compile: bool,
    otf_graph=False,
    device="cuda",
    autograd: bool = True,
):
    backbone = get_allscaip_backbone(
        cutoff=cutoff,
        use_compile=use_compile,
        otf_graph=otf_graph,
        device=device,
        autograd=autograd,
    )
    heads = {
        "efs_head": AllScAIPGradientEnergyForceStressHead(backbone, wrap_property=False)
    }
    model = HydraModelV2(backbone, heads).to(device)
    model.eval()
    return model


@pytest.mark.gpu()
def test_fixed_forward_full_gpu():
    # make_deterministic()
    torch.compiler.reset()
    device = "cuda"
    cutoff = 6.0
    seed_everywhere()
    # get model
    model = get_allscaip_full(cutoff=cutoff, use_compile=False, device=device)
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
    assert torch.allclose(fixed_results["energy"], model_output["energy"], atol=1e-5)
    assert torch.allclose(fixed_results["forces"], model_output["forces"], atol=1e-4)
    assert torch.allclose(fixed_results["stress"], model_output["stress"], atol=1e-5)
