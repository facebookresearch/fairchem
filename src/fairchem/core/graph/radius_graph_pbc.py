"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch


def sum_partitions(x: torch.Tensor, partition_idxs: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros(partition_idxs.shape[0] - 1, device=x.device, dtype=x.dtype)
    for idx in range(partition_idxs.shape[0] - 1):
        sums[idx] = x[partition_idxs[idx] : partition_idxs[idx + 1]].sum()
    return sums


def get_counts(x: torch.Tensor, length: int):
    dtype = x.dtype
    device = x.device
    return torch.zeros(length, device=device, dtype=dtype).scatter_reduce(
        dim=0,
        index=x,
        src=torch.ones(x.shape[0], device=device, dtype=dtype),
        reduce="sum",
    )


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    num_neighbors = get_counts(edge_index[1], data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    return sum_partitions(num_neighbors, image_indptr)


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    num_neighbors = get_counts(index, num_atoms)

    max_num_neighbors = num_neighbors.max()
    if max_num_neighbors_threshold > 0:
        num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)
    else:
        num_neighbors_thresholded = num_neighbors

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors],
        np.inf,
        device=device,
        dtype=atom_distance.dtype,
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)
    return mask_num_neighbors, num_neighbors_image


# @torch.no_grad() # breaks torch compile test with OOM
def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
    pbc = canonical_pbc(data, pbc)

    device = data.pos.device
    batch_size = len(data.natoms)

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep.item(), rep.item() + 1, device=device, dtype=data.cell.dtype)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def canonical_pbc(data, pbc: torch.Tensor | None):
    assert hasattr(data, "pbc"), "AtomicData does not have pbc set"
    if pbc is None and hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        pbc = torch.BoolTensor([True, True, True])
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )
    # elif pbc is not None and hasattr(data, "pbc"):
    #     # This can be on a different device, deffering to a new PR to fix this TODO
    #     if (pbc != data.pbc).all():
    #         logging.warning("PBC provided to radius_graph_pbc differs from data.pbc")
    elif pbc is None:
        pbc = torch.BoolTensor([True, True, True])

    assert isinstance(pbc, torch.Tensor)
    assert pbc.ndim == 1
    assert pbc.shape[0] == 3
    return list(pbc)


def box_size_warning(cell, pos, pbc):
    if hasattr(box_size_warning, "already_printed"):
        return
    box_size_warning.already_printed = True
    logging.warning(
        f"PBCv2: graph generation encountered a very large box. The size of the cell is {cell} and min/max positions are {pos.min(),pos.max()}. Performance will be slower than optimal."
    )
    if any(pbc):
        logging.warning(
            "PBCv2: Does this system require PBC=True or will PBC=False work?"
        )


@torch.no_grad()
def radius_graph_pbc_v2(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
    pbc = canonical_pbc(data, pbc)

    device = data.pos.device
    batch_size = len(data.natoms)
    data_batch_idxs = (
        data.batch
        if data.batch is not None
        else data.pos.new_zeros(data.natoms[0], dtype=torch.int)
    )
    # Resolution of the grid cells, should be less than the radius
    grid_resolution = radius / 1.99

    # This function assumes that all atoms are within the unti cell. If atoms
    # are outside of the unit cell, it will not work correctly.
    # position of the atoms
    atom_pos = data.pos
    num_atoms = len(data.pos)
    num_atoms_per_image = data.natoms

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(batch_size)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(batch_size)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(batch_size)

    rep = torch.cat([rep_a1.view(-1, 1), rep_a2.view(-1, 1), rep_a3.view(-1, 1)], dim=1)
    cells_per_image = (
        (rep[:, 0] * 2 + 1.0) * (rep[:, 1] * 2 + 1.0) * (rep[:, 2] * 2 + 1.0)
    ).long()

    # Create a tensor of unit cells for each image
    unit_cell = torch.zeros(
        torch.sum(cells_per_image), 3, device=device, dtype=data.cell.dtype
    )
    offset = 0
    for i in range(batch_size):
        cells_x = torch.arange(
            -rep[i][0], rep[i][0] + 1, device=device, dtype=torch.float
        )
        cells_y = torch.arange(
            -rep[i][1], rep[i][1] + 1, device=device, dtype=torch.float
        )
        cells_z = torch.arange(
            -rep[i][2], rep[i][2] + 1, device=device, dtype=torch.float
        )
        unit_cell[offset : cells_per_image[i] + offset] = torch.cartesian_prod(
            cells_x, cells_y, cells_z
        )
        offset = offset + cells_per_image[i]

    # Compute the x, y, z positional offsets for each cell in each image
    cell_matrix = torch.transpose(data.cell, 1, 2)
    cell_matrix = torch.repeat_interleave(cell_matrix, cells_per_image, dim=0)
    pbc_cell_offsets = torch.bmm(cell_matrix, unit_cell.view(-1, 3, 1)).squeeze(-1)

    # Position of the target atoms for the edges
    target_atom_pos = atom_pos
    target_atom_image = data_batch_idxs

    # Compute the position of the source atoms for the edges. There are
    # more source atoms than target atoms, since the source atoms are
    # tiled by the PBC cells.
    num_cells_per_atom = torch.repeat_interleave(cells_per_image, num_atoms_per_image)
    source_atom_index = torch.repeat_interleave(
        torch.arange(num_atoms, device=device).long(), num_cells_per_atom
    )
    source_atom_image = data_batch_idxs[source_atom_index]
    source_atom_pos = atom_pos[source_atom_index]

    # For each atom the index of the PBC cell
    pbc_cell_index = torch.tensor([], device=device).long()
    offset = 0
    for i in range(batch_size):
        cell_indices = (
            torch.arange(offset, offset + cells_per_image[i], device=device)
            .repeat(num_atoms_per_image[i])
            .long()
        )
        pbc_cell_index = torch.cat([pbc_cell_index, cell_indices], dim=0)
        offset = offset + cells_per_image[i]

    # Remember the source cell for later use
    source_cell = unit_cell[pbc_cell_index]

    # Compute their PBC cell offsets
    source_pbc_cell_offsets = pbc_cell_offsets[pbc_cell_index]
    # Add on the PBC cell offsets
    source_atom_pos = source_atom_pos + source_pbc_cell_offsets

    # Given the positions of all the atoms has been computed, we
    # split them up using a cubic grid to make pairwise comparisons
    # more computationally efficient. The resolution of the grid
    # is grid_resolution.

    # Compute the grid index for each dimension for each atom
    max_internal_cell = (
        max(source_atom_pos.abs().max(), target_atom_pos.abs().max()) / grid_resolution
    )
    if max_internal_cell > 200:
        box_size_warning(data.cell, data.pos, pbc)
        grid_resolution = max(
            grid_resolution,
            max(source_atom_pos.abs().max(), target_atom_pos.abs().max()) / 200,
        )
    source_atom_grid = torch.floor(source_atom_pos / grid_resolution).long()
    target_atom_grid = torch.floor(target_atom_pos / grid_resolution).long()

    # Find the min and max grid index for each image
    unique_atom_image, num_source_atoms_per_image = torch.unique(
        source_atom_image, return_counts=True
    )
    max_num_source_atoms = torch.max(num_source_atoms_per_image)
    # Create a new array with size [batch_size, max_num_source_atoms] to hold
    # the grid indices for each atom. We can then perform max/min on each
    # image separately.
    # First, create a mapping from the array of source atoms to the 2D array.
    source_atom_offset_per_image = torch.cumsum(num_source_atoms_per_image, dim=0).roll(
        1
    )
    source_atom_offset_per_image[0] = 0
    source_atom_offset_per_image = torch.repeat_interleave(
        source_atom_offset_per_image, num_source_atoms_per_image, 0
    )
    source_atom_mapping = (
        torch.arange(len(source_atom_offset_per_image), device=device, dtype=torch.long)
        - source_atom_offset_per_image
    )
    source_atom_mapping = (
        source_atom_mapping
        + torch.repeat_interleave(unique_atom_image, num_source_atoms_per_image, 0)
        * max_num_source_atoms
    )

    # Create 2D array
    source_atom_grid_per_image = torch.zeros(
        batch_size * max_num_source_atoms, 3, device=device, dtype=torch.long
    )
    # Populate with the grid values
    source_atom_grid_per_image[source_atom_mapping] = source_atom_grid
    source_atom_grid_per_image = source_atom_grid_per_image.view(
        batch_size, max_num_source_atoms, 3
    )
    # Perform min and max operations per image
    grid_min, no_op = torch.min(source_atom_grid_per_image, dim=1)
    grid_max, no_op = torch.max(source_atom_grid_per_image, dim=1)

    # Size of grid in each dimension for each image
    grid_size = grid_max - grid_min + 1
    grid_length = grid_size[:, 0] * grid_size[:, 1] * grid_size[:, 2]
    # Offset between grids for each image
    grid_offset = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(grid_length, dim=0)], dim=0
    )

    num_grid_cells = torch.sum(grid_length)

    # Subtract the minimum grid index so they are zero indexed
    source_atom_grid = source_atom_grid - grid_min[source_atom_image]
    target_atom_grid = target_atom_grid - grid_min[target_atom_image]

    # Compute the grid id which is a combination of the grid indices in the x,y,z directions
    # Grid id is x + y*grid_size[0] + z*grid_size[0]*grid_size[1] + offset
    source_atom_grid_size = grid_size[source_atom_image]
    source_atom_grid_offset = grid_offset[source_atom_image]
    source_atom_grid_id = (
        source_atom_grid[:, 0]
        + source_atom_grid[:, 1] * source_atom_grid_size[:, 0]
        + source_atom_grid[:, 2]
        * source_atom_grid_size[:, 0]
        * source_atom_grid_size[:, 1]
    )
    source_atom_grid_id = source_atom_grid_id + source_atom_grid_offset

    target_atom_grid_size = grid_size[target_atom_image]
    target_atom_grid_offset = grid_offset[target_atom_image]
    target_atom_grid_id = (
        target_atom_grid[:, 0]
        + target_atom_grid[:, 1] * target_atom_grid_size[:, 0]
        + target_atom_grid[:, 2]
        * target_atom_grid_size[:, 0]
        * target_atom_grid_size[:, 1]
    )
    target_atom_grid_id = target_atom_grid_id + target_atom_grid_offset

    # Compute a mapping from the array of atoms to a 2D array containing
    # all the atoms in each grid cell of size [num_grid_cells, max_atoms_per_grid_cell].
    # Pad each list of atoms for each grid cell so each
    # list is of the same length - max_atoms_per_grid_cell.
    sort_grid_id, sort_indices = torch.sort(source_atom_grid_id)
    grid_cell_atom_count = torch.zeros(
        num_grid_cells, device=device, dtype=source_atom_grid_id.dtype
    )
    grid_cell_atom_count.index_add_(
        0, source_atom_grid_id, torch.ones_like(source_atom_grid_id)
    )

    # Maximum number of atoms in a grid cell used to pad the array of atoms in each
    max_atoms_per_grid_cell = torch.max(grid_cell_atom_count)

    # Compute a mapping from the grid cell lists to the atoms in that grid cell
    cum_sum_grid_cell_atom_count = torch.cumsum(grid_cell_atom_count, 0)
    cum_sum_grid_cell_atom_count = torch.roll(cum_sum_grid_cell_atom_count, 1, 0)
    cum_sum_grid_cell_atom_count[0] = 0
    cum_sum_grid_cell_offset = torch.repeat_interleave(
        cum_sum_grid_cell_atom_count, grid_cell_atom_count, dim=0
    )
    grid_cell_offset = (  # If this OOMs it could be because boxsize is large and PBC is on
        torch.arange(num_grid_cells, device=device) * max_atoms_per_grid_cell
    )
    grid_cell_offset = torch.repeat_interleave(
        grid_cell_offset, grid_cell_atom_count, dim=0
    )
    grid_atom_map = (
        torch.arange(len(grid_cell_offset), device=device)
        + grid_cell_offset
        - cum_sum_grid_cell_offset
    )

    # If an entry doesn't have an atom, set to a value of -1
    grid_atom_index = (
        torch.zeros(
            num_grid_cells * max_atoms_per_grid_cell, dtype=torch.long, device=device
        )
        - 1
    )
    # Populate the 2D array with the atom indices
    grid_atom_index[grid_atom_map] = sort_indices
    grid_atom_index = grid_atom_index.view(num_grid_cells, max_atoms_per_grid_cell)
    # Add a Null grid cell to the end
    grid_atom_index = torch.cat(
        [
            grid_atom_index,
            torch.zeros(max_atoms_per_grid_cell, device=device).view(1, -1).long() - 1,
        ],
        dim=0,
    )
    null_grid_index = num_grid_cells

    # How many grid cells do we need to include in each direction given the search radius?
    padding_size = math.floor(radius / grid_resolution) + 1
    num_padding_grid_cells = (
        (2 * padding_size + 1) * (2 * padding_size + 1) * (2 * padding_size + 1)
    )

    padding_offsets = torch.arange(
        -padding_size, padding_size + 1, device=device, dtype=torch.long
    )
    padding_offsets = padding_offsets.view(1, -1).repeat(3, 1)
    padding_offsets = torch.cartesian_prod(
        padding_offsets[0, :], padding_offsets[1, :], padding_offsets[2, :]
    )
    padding_offsets = padding_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    grid_index = target_atom_grid.view(-1, 1, 3).repeat(1, num_padding_grid_cells, 1)
    grid_index = grid_index + padding_offsets[target_atom_image]
    max_index = target_atom_grid_size.view(-1, 1, 3).expand(
        -1, num_padding_grid_cells, -1
    )
    out_of_bounds = torch.logical_and(grid_index.ge(0), grid_index.lt(max_index))
    out_of_bounds = torch.all(out_of_bounds, dim=2).ne(True)

    padding_offsets[:, :, 1] = padding_offsets[:, :, 1] * grid_size.unsqueeze(-1)[:, 0]
    padding_offsets[:, :, 2] = (
        padding_offsets[:, :, 2]
        * grid_size.unsqueeze(-1)[:, 0]
        * grid_size.unsqueeze(-1)[:, 1]
    )
    padding_offsets = torch.sum(padding_offsets, dim=2)

    # For every cell, compute a list of neighboring grid cells.
    neighboring_grid_cells = target_atom_grid_id.view(-1, 1).repeat(
        1, num_padding_grid_cells
    )
    neighboring_grid_cells = neighboring_grid_cells + padding_offsets[target_atom_image]

    # Filter neighboring cells that are out of bounds
    neighboring_grid_cells.masked_fill_(out_of_bounds, null_grid_index)

    target_atom_edge_index = (
        torch.arange(len(target_atom_pos), device=device)
        .view(-1, 1, 1)
        .repeat(1, num_padding_grid_cells, max_atoms_per_grid_cell)
    )
    source_atom_edge_index = grid_atom_index[neighboring_grid_cells]

    # Remove padded atoms
    atom_pad_mask = source_atom_edge_index.ne(-1)
    source_atom_edge_index = torch.masked_select(source_atom_edge_index, atom_pad_mask)
    target_atom_edge_index = torch.masked_select(target_atom_edge_index, atom_pad_mask)

    # Get the atom positions
    source_atom_edge_pos = source_atom_pos[source_atom_edge_index]
    target_atom_edge_pos = target_atom_pos[target_atom_edge_index]

    # Compute their distances
    atom_distance_sqr = torch.sum(
        (target_atom_edge_pos - source_atom_edge_pos) ** 2, dim=1
    )
    # Is the distance within the radius and not 0 (the same atom)
    within_radius_mask = torch.logical_and(
        atom_distance_sqr.le(radius * radius), atom_distance_sqr.ne(0.0)
    )
    source_atom_edge_index = torch.masked_select(
        source_atom_edge_index, within_radius_mask
    )
    target_atom_edge_index = torch.masked_select(
        target_atom_edge_index, within_radius_mask
    )
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, within_radius_mask)

    # Get the return values
    # The indices of the atoms for each edge
    source_idx = source_atom_index[source_atom_edge_index]
    target_idx = target_atom_edge_index
    # The cell for each source atom
    source_cell = source_cell[source_atom_edge_index]
    # The number of edge per image
    no_op, num_neighbors_image = torch.unique(
        source_atom_image[source_atom_edge_index], return_counts=True
    )

    # Reduce the number of neighbors for each atom to the
    # desired threshold max_num_neighbors_threshold
    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=target_idx,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        target_idx = torch.masked_select(target_idx, mask_num_neighbors)
        source_idx = torch.masked_select(source_idx, mask_num_neighbors)
        source_cell = torch.masked_select(
            source_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        source_cell = source_cell.view(-1, 3)

    edge_index = torch.stack((source_idx, target_idx))

    return edge_index, source_cell, num_neighbors_image
