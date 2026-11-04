"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
from e3nn import o3

from fairchem.core.models.uma.nn.dropout import (
    DropPath,
    EquivariantDropout,
    EquivariantDropoutArraySphericalHarmonics,
    EquivariantScalarsDropout,
    GraphDropPath,
    drop_path,
)


def test_drop_path_preserves_expectation_and_shared_graph_masks(seed_fixture):
    x = torch.ones(512, 3, 2)
    layer = DropPath(0.5)
    result = layer(x)
    assert set(result.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(result, result[:, :1, :1].expand_as(result))
    assert abs(result.mean().item() - 1.0) < 0.15
    layer.eval()
    assert layer(x) is x
    assert drop_path(x, drop_prob=0, training=True) is x
    assert "drop_prob=0.5" in repr(layer)
    graph_layer = GraphDropPath(0.5)
    batch = torch.arange(512).repeat_interleave(2)
    graph_result = graph_layer(torch.ones(1024, 3), batch).reshape(512, 2, 3)
    assert torch.equal(graph_result[:, 0], graph_result[:, 1])
    assert set(graph_result.unique().tolist()) == {0.0, 2.0}
    assert abs(graph_result.mean().item() - 1.0) < 0.15
    graph_layer.eval()
    assert torch.equal(graph_layer(torch.ones(1024, 3), batch), torch.ones(1024, 3))
    assert "drop_prob=0.5" in repr(graph_layer)


def test_equivariant_dropout_preserves_vector_components(seed_fixture):
    irreps = o3.Irreps("2x0e + 1x1o")
    x = torch.ones(32, irreps.dim)
    layer = EquivariantDropout(irreps, 0.5)
    result = layer(x)
    assert set(result.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(result[:, 2:], result[:, 2:3].expand(-1, 3))
    layer.eval()
    assert layer(x) is x
    assert EquivariantDropout(irreps, 0)(x) is x
    scalar_layer = EquivariantScalarsDropout(irreps, 0.5)
    scalar_result = scalar_layer(x)
    assert torch.equal(scalar_result[:, 2:], x[:, 2:])
    assert set(scalar_result[:, :2].unique().tolist()) == {0.0, 2.0}
    scalar_layer.eval()
    assert scalar_layer(x) is x
    assert EquivariantScalarsDropout(irreps, 0)(x) is x
    assert "irreps=" in repr(scalar_layer)


def test_spherical_dropout_shares_masks_across_components_and_graphs(seed_fixture):
    x = torch.ones(32, 9, 4)
    batch = torch.arange(16).repeat_interleave(2)
    for drop_graph in (False, True):
        layer = EquivariantDropoutArraySphericalHarmonics(0.5, drop_graph=drop_graph)
        result = layer(x, batch=batch)
        assert torch.equal(result, result[:, :1, :].expand_as(result))
        assert set(result.unique().tolist()) == {0.0, 2.0}
        if drop_graph:
            assert torch.equal(result[::2], result[1::2])
        layer.eval()
        assert layer(x) is x
        assert "drop_graph=" in repr(layer)
    assert EquivariantDropoutArraySphericalHarmonics(0)(x) is x
