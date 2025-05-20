"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc, radius_graph_pbc_v2


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


# TODO: compiling internal graph gen is not supported right now
@torch.compiler.disable()
def generate_graph(
    data: dict,  # this is still a torch geometric batch object currently, turn this into a dict
    cutoff: float,
    max_neighbors: int,
    enforce_max_neighbors_strictly: bool,
    radius_pbc_version: int,
    pbc: torch.Tensor,
) -> dict:
    """Generate a graph representation from atomic structure data.

    Args:
        data (dict): A dictionary containing a batch of molecular structures.
            It should have the following keys:
                - 'pos' (torch.Tensor): Positions of the atoms.
                - 'cell' (torch.Tensor): Cell vectors of the molecular structures.
                - 'natoms' (torch.Tensor): Number of atoms in each molecular structure.
        cutoff (float): The maximum distance between atoms to consider them as neighbors.
        max_neighbors (int): The maximum number of neighbors to consider for each atom.
        enforce_max_neighbors_strictly (bool): Whether to strictly enforce the maximum number of neighbors.
        radius_pbc_version: the version of radius_pbc impl
        pbc (list[bool]): The periodic boundary conditions in 3 dimensions, defaults to [True,True,True] for 3D pbc

    Returns:
        dict: A dictionary containing the generated graph with the following keys:
            - 'edge_index' (torch.Tensor): Indices of the edges in the graph.
            - 'edge_distance' (torch.Tensor): Distances between the atoms connected by the edges.
            - 'edge_distance_vec' (torch.Tensor): Vectors representing the distances between the atoms connected by the edges.
            - 'cell_offsets' (torch.Tensor): Offsets of the cell vectors for each edge.
            - 'offset_distances' (torch.Tensor): Distances between the atoms connected by the edges, including the cell offsets.
            - 'neighbors' (torch.Tensor): Number of neighbors for each atom.
    """

    if radius_pbc_version == 1:
        radius_graph_pbc_fn = radius_graph_pbc
    elif radius_pbc_version == 2:
        radius_graph_pbc_fn = radius_graph_pbc_v2
    else:
        raise ValueError(f"Invalid radius_pbc version {radius_pbc_version}")

    (
        edge_index_per_system,
        cell_offsets_per_system,
        neighbors_per_system,
    ) = list(
        zip(
            *[
                radius_graph_pbc_fn(
                    data[idx],  # loop over the batches?
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                    pbc=pbc[idx],
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

    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": cell_offset_distances,
        "neighbors": neighbors,
    }
