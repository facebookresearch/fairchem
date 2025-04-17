"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

# TODO lets move these functions to this file
from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)


@dataclass
class GraphData:
    """Class to keep graph attributes nicely packaged."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_distance_vec: torch.Tensor
    cell_offsets: torch.Tensor
    offset_distances: torch.Tensor
    neighbors: torch.Tensor
    batch_full: torch.Tensor  # used for GP functionality
    atomic_numbers_full: torch.Tensor  # used for GP functionality
    node_offset: int = 0  # used for GP functionality


def generate_graph(
    data: dict,
    cutoff: float,
    max_neighbors: int,
    use_pbc: bool,
    otf_graph: bool,
    enforce_max_neighbors_strictly: bool,
    use_pbc_single: bool,
) -> GraphData:
    """Generate a graph representation from atomic structure data.

    Args:
        data: Dictionary containing atomic structure data (positions, cell, etc.)
        cutoff: Cutoff radius for neighbor search
        max_neighbors: Maximum number of neighbors per atom
        use_pbc: Whether to use periodic boundary conditions
        otf_graph: Whether to generate the graph on-the-fly or use pre-computed edges
        enforce_max_neighbors_strictly: Whether to strictly enforce the maximum number of neighbors
        use_pbc_single: Whether to process each system separately when using PBC

    Returns:
        GraphData: Object containing graph representation with edge indices, distances, etc.
    """
    if not otf_graph:
        try:
            edge_index = data.edge_index

            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            logging.warning(
                "Turning otf_graph=True as required attributes not present in data object"
            )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            if use_pbc_single:
                (
                    edge_index_per_system,
                    cell_offsets_per_system,
                    neighbors_per_system,
                ) = list(
                    zip(
                        *[
                            radius_graph_pbc(
                                data[idx],
                                cutoff,
                                max_neighbors,
                                enforce_max_neighbors_strictly,
                            )
                            for idx in range(len(data))
                        ]
                    )
                )

                # atom indexs in the edge_index need to be offset
                atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
                atom_index_offset[0] = 0
                edge_index = torch.hstack(
                    [
                        edge_index_per_system[idx] + atom_index_offset[idx]
                        for idx in range(len(data))
                    ]
                )
                cell_offsets = torch.vstack(cell_offsets_per_system)
                neighbors = torch.hstack(neighbors_per_system)
            else:
                ## TODO this is the original call, but blows up with memory
                ## using two different samples
                ## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
                ## sid='75396' (OC22)
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:  # TODO radius_graph no longer exists
            pass
            # edge_index = radius_graph(
            #     data.pos,
            #     r=cutoff,
            #     batch=data.batch,
            #     max_num_neighbors=max_neighbors,
            # )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)

    return GraphData(
        edge_index=edge_index,
        edge_distance=edge_dist,
        edge_distance_vec=distance_vec,
        cell_offsets=cell_offsets,
        offset_distances=cell_offset_distances,
        neighbors=neighbors,
        node_offset=0,
        batch_full=data.batch,
        atomic_numbers_full=data.atomic_numbers.long(),
    )
