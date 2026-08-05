"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

# PARQ depends on torchao's prototype module (the `quant` extra). Skip the whole
# file cleanly if it is not installed.
pytest.importorskip("torchao.prototype.parq")

from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone  # noqa: E402
from fairchem.core.modules.quantization.parq import (  # noqa: E402
    build_parq_optimizer,
    build_parq_param_groups,
    is_quantizable,
)

# A deliberately tiny backbone so the tests run in a few seconds on CPU.
_TINY_BACKBONE_KWARGS = dict(
    moe_layer_type="pytorch",
    num_experts=0,
    use_composition_embedding=True,
    use_global_embedding=False,
    max_num_elements=100,
    sphere_channels=16,
    lmax=2,
    mmax=2,
    otf_graph=True,
    max_neighbors=300,
    use_pbc=True,
    use_pbc_single=True,
    cutoff=6,
    edge_channels=16,
    distance_function="gaussian",
    num_distance_basis=32,
    regress_forces=True,
    regress_stress=False,
    num_layers=2,
    hidden_channels=16,
    norm_type="rms_norm_sh",
    act_type="gate",
    ff_type="spectral",
    chg_spin_emb_type="rand_emb",
    cs_emb_grad=True,
    dataset_list=["omol"],
    moe_dropout=0.0,
    direct_forces=False,
)


@pytest.fixture()
def tiny_backbone():
    torch.manual_seed(0)
    return eSCNMDMoeBackbone(**_TINY_BACKBONE_KWARGS)


def test_quantizable_filter_partition(tiny_backbone):
    """The filter must select only the equivariance-safe linear weights and put
    them in exactly one quant group, leaving embeddings / norms / biases out."""
    groups = build_parq_param_groups(tiny_backbone, quant_bits=8, weight_decay=1e-3)

    quant_groups = [g for g in groups if "quant_bits" in g]
    assert len(quant_groups) == 1, "expected exactly one quantized param group"

    # Map param -> name so we can inspect what landed in the quant group.
    name_of = {p.data_ptr(): n for n, p in tiny_backbone.named_parameters()}
    quant_names = [name_of[p.data_ptr()] for p in quant_groups[0]["params"]]

    # Every quant-group member must satisfy is_quantizable...
    assert quant_names, "quant group is empty"
    assert all(is_quantizable(n) for n in quant_names)

    # ...and nothing excluded may leak in: no head, no norm affine params, no
    # biases, and no actual nn.Embedding lookup tables. (Note the substring
    # "embedding" is NOT a valid exclusion test: edge_degree_embedding.rad_func
    # is a quantizable radial MLP, so we match true nn.Embedding modules.)
    embedding_param_names = {
        f"{mod_name}.weight"
        for mod_name, mod in tiny_backbone.named_modules()
        if isinstance(mod, torch.nn.Embedding)
    }
    for n in quant_names:
        assert "energy_block" not in n
        assert ".affine_" not in n
        assert not n.endswith(".bias")
        assert n not in embedding_param_names

    # Partition completeness: the quant group must contain *all* trainable
    # params the filter matches (no quantizable weight silently dropped).
    expected = {
        n
        for n, p in tiny_backbone.named_parameters()
        if p.requires_grad and is_quantizable(n)
    }
    assert set(quant_names) == expected


def test_parq_collapses_quantizable_weights_only(tiny_backbone):
    """After annealing to hard quantization, a quantizable weight collapses to a
    <= 2^bits-value grid while an excluded (embedding) weight stays full precision."""
    bits = 4
    total_steps = 60
    opt = build_parq_optimizer(
        tiny_backbone,
        lr=1e-3,
        weight_decay=1e-3,
        quant_bits=bits,
        total_steps=total_steps,
        anneal_start_frac=0.1,  # start soft->hard at step ~6
        anneal_end_frac=0.5,  # fully hard by step ~30
        anneal_steepness=10.0,
        warmup_steps=0,
        quant_period=5,
    )

    # Pick a tracked quantizable SO(2) fc weight and an excluded embedding weight.
    quant_w = quant_name = None
    embed_w = embed_name = None
    for n, p in tiny_backbone.named_parameters():
        if (
            quant_w is None
            and n.endswith("so2_m_conv.0.fc.weight")
            and is_quantizable(n)
        ):
            quant_w, quant_name = p, n
        if embed_w is None and isinstance(
            dict(tiny_backbone.named_modules()).get(n.rsplit(".", 1)[0]),
            torch.nn.Embedding,
        ):
            embed_w, embed_name = p, n
    assert quant_w is not None, "no SO(2) fc weight found in tiny backbone"
    assert embed_w is not None, "no embedding weight found in tiny backbone"

    # Fake training steps: give every trainable param a gradient and step.
    for _ in range(total_steps):
        for p in tiny_backbone.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 1e-2
        opt.step()

    assert not torch.isnan(quant_w).any()
    assert not torch.isinf(quant_w).any()

    n_unique_quant = quant_w.detach().unique().numel()
    assert n_unique_quant <= 2**bits, (
        f"{quant_name}: expected <= {2**bits} unique values after hard "
        f"quantization, got {n_unique_quant}"
    )

    n_unique_embed = embed_w.detach().unique().numel()
    assert n_unique_embed > 2**bits, (
        f"{embed_name}: excluded param should stay full precision, but has only "
        f"{n_unique_embed} unique values"
    )
