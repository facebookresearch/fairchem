"""Tests for padded neighbor edges in UMA."""

from __future__ import annotations

import torch

from fairchem.core.models.uma.escn_md import eSCNMDBackbone


def test_precomputed_padding_is_moved_beyond_cutoff():
    backbone = object.__new__(eSCNMDBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.cutoff = 5.0
    backbone.otf_graph = False
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": torch.eye(3).unsqueeze(0),
        "natoms": torch.tensor([2]),
        "edge_index": torch.tensor([[0, 0], [1, 0]]),
        "cell_offsets": torch.zeros(2, 3),
        "edge_valid_mask": torch.tensor([True, False]),
    }

    graph = backbone._generate_graph(data)

    torch.testing.assert_close(graph["edge_distance"][0], torch.tensor(1.0))
    torch.testing.assert_close(graph["edge_distance"][1], torch.tensor(6.0))
    torch.testing.assert_close(data["scatter_target"], torch.tensor([1, 0]))


def test_precomputed_periodic_shifts_match_internal_graph_math():
    backbone = object.__new__(eSCNMDBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.cutoff = 5.0
    backbone.otf_graph = False
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": torch.diag(torch.tensor([3.0, 4.0, 5.0])).unsqueeze(0),
        "natoms": torch.tensor([2]),
        "edge_index": torch.tensor([[0], [1]]),
        "cell_offsets": torch.tensor([[1.0, 0.0, 0.0]]),
    }

    graph = backbone._generate_graph(data)

    torch.testing.assert_close(
        graph["edge_distance_vec"], torch.tensor([[2.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(graph["edge_distance"], torch.tensor([2.0]))
