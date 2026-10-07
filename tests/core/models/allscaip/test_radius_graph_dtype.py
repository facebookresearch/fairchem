"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.models.allscaip.utils.allscaip_radius_graph import (
    biknn_radius_graph,
)


def _get_batch(num_atoms: int = 8):
    atoms = get_fcc_crystal_by_num_atoms(num_atoms)
    data_object = AtomicData.from_ase(atoms)
    data_object.natoms = torch.tensor(len(atoms))
    data_object.charge = torch.LongTensor([0])
    data_object.spin = torch.LongTensor([0])
    data_object.dataset = "omol"
    loader = torch.utils.data.DataLoader(
        [data_object],
        collate_fn=partial(data_list_collater, otf_graph=True),
        batch_size=1,
        shuffle=False,
    )
    return next(iter(loader))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pbc", [True, False])
def test_biknn_radius_graph_follows_cell_dtype(dtype, pbc):
    """The PBC image offsets must follow the batch dtype, not the global default.

    float64 inference (InferenceSettings base_precision_dtype=torch.float64)
    hands this function a double batch while torch's default dtype is still
    float32; a default-dtype image_id then crashes torch.mm(image_id, cell)
    in build_radius_graph with "expected mat1 and mat2 to have the same
    dtype, but got: float != double".
    """
    batch = _get_batch()
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    if not pbc:
        batch.pbc = torch.zeros_like(batch.pbc)

    outputs = biknn_radius_graph(
        batch,
        cutoff=6.0,
        knn_k=20,
        knn_soft=False,
        knn_sigmoid_scale=0.2,
        knn_lse_scale=0.1,
        knn_use_low_mem=False,
        knn_pad_size=None,
        device=torch.device("cpu"),
    )
    disp = outputs[1]
    assert disp.dtype == dtype
