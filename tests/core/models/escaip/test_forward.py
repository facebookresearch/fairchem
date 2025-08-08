"""
Modified from tests/core/models/uma/test_compile.py
"""

from __future__ import annotations

import itertools
import random
from functools import partial

import numpy as np
import pytest
import torch
from ase import build

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.escaip.EScAIP import (
    EScAIPBackbone,
    EScAIPGradientEnergyForceStressHead,
)

MAX_ELEMENTS = 100


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ase_to_graph(atoms, neighbors: int, cutoff: float):
    data_object = AtomicData.from_ase(
        atoms,
        max_neigh=neighbors,
        radius=cutoff,
        r_edges=True,
    )
    data_object.natoms = torch.tensor(len(atoms))
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


def get_diamond_tg_data(neighbors: int, cutoff: float, size: int, device: str):
    # get torch geometric data object for diamond
    # atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = build.bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms = atoms.repeat((size, size, size))
    return ase_to_graph(atoms, neighbors, cutoff).to(device)


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
    sizes = range(3, 7)
    neighbors = range(30, 100, 5)
    for size, neigh in list(itertools.product(sizes, neighbors)):
        data = get_diamond_tg_data(neigh, cutoff, size, device)
        seed_everywhere()
        output = model_no_compile(data)["efs_head"]
        seed_everywhere()
        output_compiled = model_compile(data)["efs_head"]
        assert torch.allclose(output["energy"], output_compiled["energy"], atol=1e-5)
        assert torch.allclose(output["forces"], output_compiled["forces"], atol=1e-4)
        assert torch.allclose(output["stress"], output_compiled["stress"], atol=1e-5)
