"""Bucketed NVIDIA neighbor graphs for compiled inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairchem.core.common import gp_utils
from fairchem.core.graph.radius_graph_pbc_nvidia import radius_graph_pbc_nvidia


class PaddedNvidiaGraphGenerator:
    """Generate v3 neighbor graphs and replace data's edges with a padded bucket."""

    def __init__(self, settings, backbone):
        if gp_utils.initialized():
            raise ValueError(
                "padded NVIDIA graph generation does not support graph parallelism"
            )
        if not getattr(backbone, "supports_padded_edges", False):
            raise ValueError(
                f"{type(backbone).__name__} does not support padded neighbor edges"
            )
        if torch.device(next(backbone.parameters()).device).type != "cuda":
            raise ValueError("padded NVIDIA graph generation requires CUDA")
        self.cutoff = float(backbone.cutoff)
        self.max_neighbors = int(backbone.max_neighbors)
        self.enforce_max_neighbors_strictly = bool(
            backbone.enforce_max_neighbors_strictly
        )
        self.edge_bucket_size = settings.internal_graph_edge_bucket_size

    def generate(self, data):
        edge_index, cell_offsets, neighbors = radius_graph_pbc_nvidia(
            data,
            self.cutoff,
            self.max_neighbors,
            self.enforce_max_neighbors_strictly,
            pbc=data.pbc,
        )
        num_edges = edge_index.shape[1]
        edge_capacity = (
            (num_edges + self.edge_bucket_size - 1) // self.edge_bucket_size
        ) * self.edge_bucket_size
        pad = edge_capacity - num_edges

        data.edge_index = F.pad(edge_index, (0, pad))
        data.cell_offsets = F.pad(cell_offsets.to(data.pos.dtype), (0, 0, 0, pad))
        data.nedges = neighbors.clone()
        data.nedges[-1] += pad
        data.edge_valid_mask = (
            torch.arange(edge_capacity, device=edge_index.device) < num_edges
        )
        return data
