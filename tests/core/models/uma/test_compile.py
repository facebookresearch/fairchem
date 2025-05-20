"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
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
from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone

MAX_ELEMENTS = 100
DATASET_LIST = ["oc20", "omol", "osc", "omat", "odac"]


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ase_to_graph(atoms, neighbors: int, cutoff: float):
    data_object = AtomicData.from_ase(atoms,
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


def get_backbone_config(cutoff: float, otf_graph=False, autograd: bool = True):
    return {
        "max_num_elements": MAX_ELEMENTS,
        "sphere_channels": 16,
        "lmax": 2,
        "mmax": 2,
        "otf_graph": otf_graph,
        "max_neighbors": 300,
        "cutoff": cutoff,
        "edge_channels": 16,
        "num_layers": 2,
        "hidden_channels": 16,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "ff_type": "spectral",
        "activation_checkpointing": False,
        "chg_spin_emb_type": "pos_emb",
        "cs_emb_grad": False,
        "dataset_emb_grad": False,
        "dataset_list": DATASET_LIST,
        "regress_stress": True,
        "direct_forces": not autograd,
        "regress_forces": True,
    }


def get_escn_md_backbone(
    cutoff: float, otf_graph=False, device="cuda", autograd: bool = True
):
    backbone_config = get_backbone_config(
        cutoff=cutoff, otf_graph=otf_graph, autograd=autograd
    )
    model = eSCNMDBackbone(**backbone_config)
    model.to(device)
    model.eval()
    return model


def get_escn_md_full(
    cutoff: float, otf_graph=False, device="cuda", autograd: bool = True
):
    backbone = get_escn_md_backbone(
        cutoff=cutoff, otf_graph=otf_graph, device=device, autograd=autograd
    )
    heads = {"efs_head": MLP_EFS_Head(backbone, wrap_property=False)}
    model = HydraModelV2(backbone, heads).to(device)
    model.eval()
    return model


# compile tests take a long time
@pytest.mark.skip()
@pytest.mark.gpu()
def test_compile_backbone_gpu():
    torch.compiler.reset()
    device = "cuda"
    cutoff = 6.0
    model = get_escn_md_backbone(cutoff=cutoff, device=device)
    compiled = torch.compile(model, dynamic=True)
    sizes = range(3, 10)
    neighbors = range(30, 100, 10)
    for size, neigh in zip(sizes, neighbors):
        print("SIZE", size, neigh)
        data = get_diamond_tg_data(neigh, cutoff, size, device)
        seed_everywhere()
        output = model(data)
        seed_everywhere()
        output_compiled = compiled(data)
        assert torch.allclose(
            output["node_embedding"], output_compiled["node_embedding"], atol=1e-5
        )


@pytest.mark.gpu()
def test_compile_full_gpu():
    torch.compiler.reset()
    device = "cuda"
    cutoff = 6.0
    model = get_escn_md_full(cutoff=cutoff, device=device)
    compiled = torch.compile(model, dynamic=True)
    sizes = range(3, 10)
    neighbors = range(30, 100, 5)
    for size, neigh in list(itertools.product(sizes, neighbors)):
        data = get_diamond_tg_data(neigh, cutoff, size, device)
        seed_everywhere()
        output = model(data)["efs_head"]
        seed_everywhere()
        output_compiled = compiled(data)["efs_head"]
        assert torch.allclose(output["energy"], output_compiled["energy"], atol=1e-5)
        assert torch.allclose(output["forces"], output_compiled["forces"], atol=1e-4)
        assert torch.allclose(output["stress"], output_compiled["stress"], atol=1e-5)
