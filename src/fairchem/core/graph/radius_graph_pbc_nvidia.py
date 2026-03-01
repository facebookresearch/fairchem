"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from monty.dev import requires

from fairchem.core.graph.radius_graph_pbc import get_max_neighbors_mask

# Try to import nvalchemiops at module load
try:
    from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors
    from nvalchemiops.neighborlist.neighborlist import neighbor_list

    def nvalchemiops_installed() -> bool:
        return True

except ImportError as e:
    logging.debug(
        f"nvalchemiops not available: {e}. Install with `pip install nvalchemiops`"
    )
    estimate_max_neighbors = None
    neighbor_list = None

    def nvalchemiops_installed() -> bool:
        return False


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.no_grad()
def get_neighbors_nvidia(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_neigh: int,
    method: str = "cell_list",
    enforce_max_neighbors_strictly: bool = False,
    batch: torch.Tensor | None = None,
    natoms: torch.Tensor | None = None,
    return_distances_sq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Performs nearest neighbor search using NVIDIA nvalchemiops and returns edge index, distances,
    and cell offsets as tensors. Supports both single structure and batched inputs.

    Args:
        positions: Atomic positions tensor (N, 3) - device and dtype determine computation
        cell: Unit cell tensor (B, 3, 3) or (3, 3) - must be on same device as positions
        pbc: Periodic boundary conditions (B, 3) or (3,) boolean tensor - must be on same device as positions
        cutoff: Cutoff radius in Angstroms
        max_neigh: Maximum number of neighbors per atom
        method: NVIDIA method to use ("naive" or "cell_list")
        enforce_max_neighbors_strictly: If True, strictly limit to max neighbors;
            if False, include additional neighbors within degeneracy tolerance
        batch: Optional batch tensor (N,) indicating which structure each atom belongs to
        natoms: Optional tensor (B,) with number of atoms per structure. If not provided,
            inferred as single structure with all atoms.
        return_distances_sq: If True, compute and return pairwise distances squared; if False, return None for distances

    Returns:
        c_index: Center atom indices (tensor, int32) - global indices if batched
        n_index: Neighbor atom indices (tensor, int32) - global indices if batched
        offsets: Cell offsets (tensor, int32, shape [num_edges, 3])
        distances_sq: Pairwise distances squared (tensor) accounting for PBC if distances_sq is True, otherwise None
    """
    # Validate input shapes
    assert (
        positions.ndim == 2
    ), f"positions must have shape (N, 3), got {positions.shape}"
    assert (
        positions.shape[1] == 3
    ), f"positions must have shape (N, 3), got {positions.shape}"
    assert cell.ndim in (
        2,
        3,
    ), f"cell must have shape (3, 3) or (B, 3, 3), got {cell.shape}"
    if cell.ndim == 2:
        assert cell.shape == (3, 3), f"cell must have shape (3, 3), got {cell.shape}"
    else:
        assert cell.shape[1:] == (
            3,
            3,
        ), f"cell must have shape (B, 3, 3), got {cell.shape}"
    assert pbc.ndim in (1, 2), f"pbc must have shape (3,) or (B, 3), got {pbc.shape}"
    if pbc.ndim == 1:
        assert pbc.shape == (3,), f"pbc must have shape (3,), got {pbc.shape}"
    else:
        assert pbc.shape[1] == 3, f"pbc must have shape (B, 3), got {pbc.shape}"
    if batch is not None:
        assert (
            batch.ndim == 1
        ), f"batch must have shape (N,) where N={positions.shape[0]}, got {batch.shape}"
        assert (
            batch.shape[0] == positions.shape[0]
        ), f"batch must have shape (N,) where N={positions.shape[0]}, got {batch.shape}"
    if natoms is not None:
        assert natoms.ndim == 1, f"natoms must have shape (B,), got {natoms.shape}"
    assert max_neigh > 0, f"max_neigh must be a positive integer, got {max_neigh}"

    device = positions.device
    total_atoms = positions.shape[0]

    # Normalize inputs to batched format
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    if pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    if batch is None:
        batch = torch.zeros(total_atoms, dtype=torch.long, device=device)
    if natoms is None:
        natoms = torch.tensor([total_atoms], dtype=torch.long, device=device)

    # Request more neighbors to handle degeneracy to allow our max neighbors filter to work properly
    # The NVIDIA neighbor list doesn't prioritize closest neighbors, so we need
    # a large buffer to ensure we capture all neighbors within the cutoff.
    # This allows the mask to correctly include degenerate edges.
    # note estimate_max_neighbors(cutoff=6.0, safety_factor=2.0) = 640 which should be overly safe.
    buffer_max_neigh = estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0)
    if max_neigh > buffer_max_neigh:
        logging.warning(
            f"max_neigh={max_neigh} is greater than the NVIDIA neighbor list buffer size of {buffer_max_neigh}. The returned neighbors will be limited to buffer_max_neigh"
        )

    # Small epsilon to ensure atoms at exactly cutoff distance are included
    nvidia_cutoff = cutoff + 1e-6

    neighbor_matrix = torch.full(
        (total_atoms, buffer_max_neigh),
        total_atoms,
        dtype=torch.int32,
        device=device,
    )
    neighbor_matrix_shifts = torch.zeros(
        (total_atoms, buffer_max_neigh, 3),
        dtype=torch.int32,
        device=device,
    )
    num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

    neighbor_list(
        positions=positions,
        cutoff=nvidia_cutoff,
        cell=cell,
        pbc=pbc,
        batch_idx=batch.int(),
        method=f"batch_{method}",
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        half_fill=False,
    )

    # Conversion from neighbor matrix to edge list
    atom_indices = torch.arange(total_atoms, device=device).unsqueeze(1)
    neigh_indices = torch.arange(buffer_max_neigh, device=device).unsqueeze(0)
    valid_mask = neigh_indices < num_neighbors.unsqueeze(1)

    c_index = atom_indices.expand(-1, buffer_max_neigh)[valid_mask]
    n_index = neighbor_matrix[valid_mask]
    offsets = neighbor_matrix_shifts[valid_mask]

    # We used to sort the edges here but the ordering of the edges is no longer required, leaving this comment here for reference

    # if max number of neighbors is less than max_neigh, we can skip the masking steps all together
    # if we don't need the distances either, then we can skip both of these steps
    filter_max_neighbors = torch.any(num_neighbors > max_neigh)
    distances_sq = None
    if return_distances_sq or filter_max_neighbors:
        # This could be added in the future so we skip this computation step here: NVIDIA/nvalchemi-toolkit-ops#14.
        # Compute squared distances with PBC corrections
        distance_vectors = positions[n_index] - positions[c_index]
        edge_cells = cell[batch[c_index]]
        offsets_cartesian = torch.bmm(
            offsets.float().unsqueeze(1),
            edge_cells.float(),
        ).squeeze(1)
        distance_vectors.add_(offsets_cartesian)
        distances_sq = (distance_vectors**2).sum(dim=-1)

    if filter_max_neighbors and c_index.shape[0] > 0:
        # Apply max neighbors mask to handle degeneracy properly
        mask_num_neighbors, _ = get_max_neighbors_mask(
            natoms=natoms,
            index=c_index,
            atom_distance=distances_sq,
            max_num_neighbors_threshold=max_neigh,
            enforce_max_strictly=enforce_max_neighbors_strictly,
        )

        c_index = c_index[mask_num_neighbors]
        n_index = n_index[mask_num_neighbors]
        distances_sq = distances_sq[mask_num_neighbors]
        offsets = offsets[mask_num_neighbors]

    return c_index, n_index, offsets, distances_sq


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.no_grad()
def radius_graph_pbc_nvidia(
    data,
    radius: float,
    max_num_neighbors_threshold: int,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVIDIA-accelerated radius graph generation with PBC support.

    This function has the same interface as radius_graph_pbc and radius_graph_pbc_v2,
    allowing it to be used as a drop-in replacement in generate_graph.

    Args:
        data: Data object with pos, cell, natoms, pbc, and optionally batch attributes
        radius: Cutoff radius for neighbor search
        max_num_neighbors_threshold: Maximum number of neighbors per atom
        enforce_max_neighbors_strictly: If True, strictly limit to max neighbors;
            if False, include additional neighbors within degeneracy tolerance
        pbc: Periodic boundary conditions tensor (optional, uses data.pbc if not provided)

    Returns:
        edge_index: (2, num_edges) tensor with [source, target] indices
        cell_offsets: (num_edges, 3) tensor with integer cell offsets
        neighbors: (batch_size,) tensor with number of edges per structure
    """
    pos = data.pos
    natoms = data.natoms
    cell = data.cell
    device = pos.device
    batch_size = len(natoms)

    # Get batch tensor
    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        batch = torch.repeat_interleave(torch.arange(batch_size, device=device), natoms)

    # Get PBC tensor
    if pbc is None:
        pbc_tensor = (
            data.pbc
            if hasattr(data, "pbc")
            else torch.tensor([True, True, True], device=device)
        )
    else:
        pbc_tensor = pbc

    c_index, n_index, offsets, _ = get_neighbors_nvidia(
        positions=pos,
        cell=cell,
        pbc=pbc_tensor,
        cutoff=radius,
        max_neigh=max_num_neighbors_threshold,
        method="cell_list",
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        batch=batch,
        natoms=natoms,
    )

    edge_batch = batch[c_index]
    num_neighbors_image = torch.zeros(batch_size, dtype=torch.long, device=device)
    num_neighbors_image.scatter_add_(0, edge_batch, torch.ones_like(edge_batch))

    edge_index = torch.stack([n_index, c_index], dim=0)

    return edge_index, offsets, num_neighbors_image


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.inference_mode()
def get_neighbors_nvidia_atoms(
    atoms, cutoff: float, max_neigh: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs nearest neighbor search using NVIDIA nvalchemiops and returns edge index, distances,
    and cell offsets.

    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff radius in Angstroms
        max_neigh: Maximum number of neighbors per atom

    Returns:
        c_index: Center atom indices (numpy array)
        n_index: Neighbor atom indices (numpy array)
        distances: Pairwise distances (numpy array) accounting for PBC
        offsets: Cell offsets (numpy array)
    """
    positions = torch.from_numpy(atoms.get_positions()).float()
    cell = torch.from_numpy(np.array(atoms.get_cell(complete=True))).float()
    pbc = torch.from_numpy(np.array(atoms.pbc)).bool()

    c_index, n_index, offsets, distances_sq = get_neighbors_nvidia(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_neigh=max_neigh,
        method="cell_list",
        enforce_max_neighbors_strictly=True,
        return_distances_sq=True,
    )

    return (
        c_index.numpy(),
        n_index.numpy(),
        np.sqrt(distances_sq.numpy()),
        offsets.numpy(),
    )
