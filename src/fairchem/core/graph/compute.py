"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.graph.radius_graph_pbc import (
    radius_graph_pbc,
    radius_graph_pbc_v2,
)
from fairchem.core.graph.radius_graph_pbc_nvidia import radius_graph_pbc_nvidia


def filter_edges_by_node_partition(
    node_partition: torch.Tensor,
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor,
    neighbors: torch.Tensor,
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter edges to keep only those where the target atom belongs to the node partition.
    edge_index is shape (2, num_edges) where the first row is the source atom index and the second row is the target atom index for each edge
    cell_offsets is shape (num_edges, 3)
    neighbors is cardinality of the edge_index per system in the batch

    Args:
        node_partition: Tensor of atom indices belonging to the current rank's partition.
        edge_index: Edge index tensor of shape (2, num_edges), where row 0 is the source and 1 is the target atom.
        cell_offsets: Cell offsets tensor of shape (num_edges, 3).
        neighbors: Tensor with edge count per system in the batch (length = num_systems).
        num_atoms: Total number of atoms across all batches. Used to create a boolean mask for filtering.

    Returns:
        Filtered edge_index, cell_offsets, and neighbors tensors.
    """
    target_atoms = edge_index[1]
    node_mask = torch.zeros(num_atoms, dtype=torch.bool, device=target_atoms.device)
    node_mask[node_partition] = True
    local_edge_mask = node_mask[target_atoms]

    # Create system index for each edge to track which system each edge belongs to
    num_systems = neighbors.shape[0]
    edge_system_idx = torch.repeat_interleave(
        torch.arange(num_systems, device=neighbors.device), neighbors
    )

    # Filter edges
    edge_index = edge_index[:, local_edge_mask]
    cell_offsets = cell_offsets[local_edge_mask]
    if neighbors.shape[0] == 1:
        # If there's only one system, we can skip the scatter_add step and just return the count of remaining edges
        new_neighbors = local_edge_mask.sum(dtype=neighbors.dtype).unsqueeze(0)
        return edge_index, cell_offsets, new_neighbors

    filtered_edge_system_idx = edge_system_idx[local_edge_mask]

    # Count edges per system after filtering
    new_neighbors = torch.zeros(
        num_systems, device=neighbors.device, dtype=neighbors.dtype
    )
    new_neighbors.scatter_add_(
        0,
        filtered_edge_system_idx,
        torch.ones_like(filtered_edge_system_idx, dtype=neighbors.dtype),
    )

    return edge_index, cell_offsets, new_neighbors


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
    offsets = cell_offsets.to(dtype=cell.dtype).view(-1, 1, 3).bmm(cell).view(-1, 3)
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
    node_partition: torch.Tensor | None = None,
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
        radius_pbc_version: the version of radius_pbc impl (1, 2, or 3 for NVIDIA)
        pbc (list[bool]): The periodic boundary conditions in 3 dimensions, defaults to [True,True,True] for 3D pbc
        node_partition (torch.Tensor | None): The partitioning of the nodes (atoms) for distributed inference. If provided, returned graph will be filtered to keep only edges where the target atom (edge_index[1,:]) belongs to the current rank's partition.

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
    elif radius_pbc_version == 3:
        radius_graph_pbc_fn = radius_graph_pbc_nvidia
    else:
        raise ValueError(f"Invalid radius_pbc version {radius_pbc_version}")

    edge_index, cell_offsets, neighbors = radius_graph_pbc_fn(
        data,
        cutoff,
        max_neighbors,
        enforce_max_neighbors_strictly,
        pbc=pbc,
    )

    if node_partition is not None:
        edge_index, cell_offsets, neighbors = filter_edges_by_node_partition(
            node_partition,
            edge_index,
            cell_offsets,
            neighbors,
            num_atoms=data.pos.shape[0],
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

    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": cell_offset_distances,
        "neighbors": neighbors,
    }
