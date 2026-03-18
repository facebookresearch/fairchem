"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.gemnet_oc.gemnet_oc import (
    GemNetOCBackbone,
    GemNetOCEnergyAndGradForceHead,
    GemNetOCForceHead,
)


@pytest.fixture()
def data():
    atoms = get_fcc_crystal_by_num_atoms(100)
    return AtomicData.from_ase(atoms)


def construct_backbone(regress_forces, direct_forces, otf_graph):
    torch.manual_seed(0)
    return GemNetOCBackbone(
        num_spherical=4,
        num_radial=8,
        num_blocks=2,
        emb_size_atom=64,
        emb_size_edge=64,
        emb_size_trip_in=16,
        emb_size_trip_out=16,
        emb_size_quad_in=8,
        emb_size_quad_out=8,
        emb_size_aint_in=16,
        emb_size_aint_out=16,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_sbf=16,
        num_before_skip=1,
        num_after_skip=1,
        num_concat=1,
        num_atom=2,
        num_output_afteratom=2,
        num_atom_emb_layers=1,
        regress_forces=regress_forces,
        direct_forces=direct_forces,
        use_pbc=True,
        cutoff=12.0,
        max_neighbors=30,
        max_neighbors_qint=8,
        max_neighbors_aeaint=20,
        max_neighbors_aint=100,
        otf_graph=otf_graph,
        chg_spin_emb_type="rand_emb",
        cs_emb_grad=True,
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
    )


@pytest.mark.parametrize(
    "regress_forces, direct_forces, otf_graph",
    [
        (True, True, True),
        (True, False, True),
    ],
)
def test_forward_pass_basic(regress_forces, direct_forces, otf_graph, data):
    """Test that the model can perform a forward pass without errors."""
    backbone = construct_backbone(regress_forces, direct_forces, otf_graph)
    heads = {
        "ef_head": GemNetOCEnergyAndGradForceHead(backbone, 2),
        "f_head": GemNetOCForceHead(backbone, 2),
    }
    model = HydraModelV2(backbone, heads)
    model.train()
    with torch.no_grad():
        output = model(data)

    # Check that outputs are returned
    assert output is not None
    assert isinstance(output, dict)

    # Check that expected keys are present
    assert "energy" in output["ef_head"]
    if regress_forces:
        if direct_forces:
            assert "forces" in output["f_head"]
        else:
            assert "forces" in output["ef_head"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
