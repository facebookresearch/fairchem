"""Tests for bucketed NVIDIA neighbor graphs."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import fairchem.core.graph.padded_nvidia_graph as padded_graph
from fairchem.core.graph.padded_nvidia_graph import PaddedNvidiaGraphGenerator


class _GraphData(SimpleNamespace):
    def clone(self):
        return _GraphData(
            **{
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in vars(self).items()
            }
        )


def _generator(bucket_size=4):
    generator = object.__new__(PaddedNvidiaGraphGenerator)
    generator.cutoff = 5.0
    generator.max_neighbors = 32
    generator.enforce_max_neighbors_strictly = False
    generator.edge_bucket_size = bucket_size
    return generator


def test_generate_pads_edges_to_bucket(monkeypatch):
    edge_index = torch.tensor([[1, 0, 2], [0, 1, 1]])
    cell_offsets = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    neighbors = torch.tensor([2, 1])

    def graph(*args, **kwargs):
        return edge_index, cell_offsets, neighbors

    monkeypatch.setattr(padded_graph, "radius_graph_pbc_nvidia", graph)
    data = _GraphData(
        pos=torch.empty(3, 3, dtype=torch.float32),
        pbc=torch.ones(2, 3, dtype=torch.bool),
    )

    result = _generator().generate(data)

    assert result is data
    torch.testing.assert_close(result.edge_index[:, :3], edge_index)
    torch.testing.assert_close(
        result.edge_index[:, 3], torch.zeros(2, dtype=torch.long)
    )
    torch.testing.assert_close(result.cell_offsets[:3], cell_offsets.float())
    torch.testing.assert_close(result.cell_offsets[3], torch.zeros(3))
    torch.testing.assert_close(result.nedges, torch.tensor([2, 2]))
    torch.testing.assert_close(
        result.edge_valid_mask, torch.tensor([True, True, True, False])
    )


def test_generate_keeps_exact_bucket_shape(monkeypatch):
    edge_index = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]])
    cell_offsets = torch.zeros(4, 3)
    neighbors = torch.tensor([4])
    monkeypatch.setattr(
        padded_graph,
        "radius_graph_pbc_nvidia",
        lambda *args, **kwargs: (edge_index, cell_offsets, neighbors),
    )
    data = _GraphData(pos=torch.empty(3, 3), pbc=torch.ones(1, 3, dtype=torch.bool))

    result = _generator().generate(data)

    torch.testing.assert_close(result.edge_index, edge_index)
    torch.testing.assert_close(result.nedges, neighbors)
    assert result.edge_valid_mask.all()
