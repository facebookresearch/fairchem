"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.common.utils import load_state_dict
from fairchem.core.datasets.solvent import SOLVENT_DIM
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.units.mlip_unit.utils import expand_mix_csd_state_dict

SPHERE_CHANNELS = 4


def _make_backbone(use_solvent_embedding: bool) -> eSCNMDBackbone:
    return eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=SPHERE_CHANNELS,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        use_solvent_embedding=use_solvent_embedding,
        solvent_emb_hidden=8,
        always_use_pbc=False,
    )


def test_expand_mix_csd_state_dict_pads_trailing_solvent_block():
    """The pretrained mix_csd weight maps onto the leading columns of the wider
    solvent-enabled layer; the new solvent block is zero."""
    torch.manual_seed(0)
    pretrained = _make_backbone(use_solvent_embedding=False)
    solvent_model = _make_backbone(use_solvent_embedding=True)

    assert pretrained.mix_csd.in_features == 2 * SPHERE_CHANNELS
    assert solvent_model.mix_csd.in_features == 3 * SPHERE_CHANNELS

    expanded = expand_mix_csd_state_dict(solvent_model, pretrained.state_dict())
    load_state_dict(solvent_model, expanded, strict=True)

    pretrained_w = pretrained.mix_csd.weight
    grafted_w = solvent_model.mix_csd.weight
    assert torch.allclose(grafted_w[:, : 2 * SPHERE_CHANNELS], pretrained_w)
    assert torch.count_nonzero(grafted_w[:, 2 * SPHERE_CHANNELS :]) == 0
    # bias has unchanged shape and is copied as-is
    assert torch.allclose(solvent_model.mix_csd.bias, pretrained.mix_csd.bias)


def test_solvent_graft_is_identity_warm_start():
    """After grafting, the system embedding is identical to the pretrained model
    for any solvent input, because the solvent columns of mix_csd are zero."""
    torch.manual_seed(0)
    pretrained = _make_backbone(use_solvent_embedding=False)
    solvent_model = _make_backbone(use_solvent_embedding=True)

    expanded = expand_mix_csd_state_dict(solvent_model, pretrained.state_dict())
    load_state_dict(solvent_model, expanded, strict=True)

    charge = torch.tensor([0.0])
    spin = torch.tensor([2.0])
    solvent = torch.randn(1, SOLVENT_DIM)

    base = pretrained.csd_embedding(charge, spin, dataset=None)
    grafted = solvent_model.csd_embedding(charge, spin, dataset=None, solvent=solvent)
    assert torch.allclose(base, grafted, atol=1e-6)


def test_strict_load_allows_missing_solvent_embedding_only():
    """Loading a pretrained (no-solvent) checkpoint into a solvent-enabled model
    must succeed under strict loading (only solvent_embedding keys are missing),
    but still raise if a genuine non-solvent key is missing."""
    torch.manual_seed(0)
    pretrained = _make_backbone(use_solvent_embedding=False)
    solvent_model = _make_backbone(use_solvent_embedding=True)

    expanded = expand_mix_csd_state_dict(solvent_model, pretrained.state_dict())
    missing, unexpected = load_state_dict(solvent_model, expanded, strict=True)
    assert missing == []
    assert unexpected == []

    broken = dict(expanded)
    a_real_key = next(k for k in broken if k.startswith("sphere_embedding"))
    del broken[a_real_key]
    with pytest.raises(RuntimeError):
        load_state_dict(solvent_model, broken, strict=True)
