"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch
from monty.dev import requires

from fairchem.core.graph.radius_graph_pbc import get_max_neighbors_mask

try:
    from nvalchemiops.torch.neighbors import neighbor_list
    from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

    nvidia_installed = True

except ImportError:
    get_neighbors_nvidia_atoms = None
    nvidia_installed = False


@requires(nvidia_installed, message="Requires `nvalchemiops` to be installed")
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
):
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

    Returns:
        c_index: Center atom indices (tensor, int64) - global indices if batched
        n_index: Neighbor atom indices (tensor, int64) - global indices if batched
        distances: Pairwise distances (tensor) accounting for PBC
        offsets: Cell offsets (tensor, int64, shape [num_edges, 3])
    """
    device = positions.device
    dtype = positions.dtype
    total_atoms = positions.shape[0]

    # Normalize inputs to batched format
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)  # (3, 3) -> (1, 3, 3)
    if pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)  # (3,) -> (1, 3)
    if batch is None:
        batch = torch.zeros(total_atoms, dtype=torch.long, device=device)
    if natoms is None:
        natoms = torch.tensor([total_atoms], dtype=torch.long, device=device)

    if max_neigh < 0:
        max_neigh = estimate_max_neighbors(cutoff)
    # When not enforcing strictly, request more neighbors to handle degeneracy
    # The NVIDIA neighbor list doesn't prioritize closest neighbors, so we need
    # a large buffer to ensure we capture all neighbors within the cutoff.
    # This allows the mask to correctly include degenerate edges.
    if not enforce_max_neighbors_strictly:
        buffer_max_neigh = max(300, max_neigh * 2)
    else:
        buffer_max_neigh = max_neigh

    # Small epsilon to ensure atoms at exactly cutoff distance are included
    nvidia_cutoff = cutoff + 1e-6

    # Allocate output tensors for neighbor_list
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

    # Always use batched method
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

    # Convert neighbor matrix to edge list format
    total_edges = num_neighbors.sum().item()
    if total_edges == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=dtype, device=device),
            torch.empty((0, 3), dtype=torch.int64, device=device),
        )

    # Conversion from neighbor matrix to edge list
    atom_indices = torch.arange(total_atoms, device=device).unsqueeze(1)
    neigh_indices = torch.arange(buffer_max_neigh, device=device).unsqueeze(0)
    valid_mask = neigh_indices < num_neighbors.unsqueeze(1)

    c_index = atom_indices.expand(-1, buffer_max_neigh)[valid_mask]
    n_index = neighbor_matrix[valid_mask]
    offsets = neighbor_matrix_shifts[valid_mask]

    # Sort by center atom for consistency
    sort_idx = torch.argsort(c_index, stable=True)
    c_index = c_index[sort_idx].long()
    n_index = n_index[sort_idx].long()
    offsets = offsets[sort_idx].long()

    # Compute distances with PBC corrections
    distance_vectors = positions[n_index] - positions[c_index]

    # Apply cell offsets using per-edge cells
    edge_cells = cell[batch[c_index]]  # [num_edges, 3, 3]
    offsets_cartesian = torch.bmm(
        offsets.float().unsqueeze(1),  # [num_edges, 1, 3]
        edge_cells.float(),  # [num_edges, 3, 3]
    ).squeeze(1)  # [num_edges, 3]
    distance_vectors = distance_vectors + offsets_cartesian

    # Compute Euclidean distances
    distances = distance_vectors.norm(dim=-1)

    # Apply max neighbors mask to handle degeneracy properly
    if max_neigh > 0 and len(c_index) > 0:
        # Use squared distances for consistency with v1/v2 implementations
        atom_distance_sqr = distances**2

        mask_num_neighbors, _ = get_max_neighbors_mask(
            natoms=natoms,
            index=c_index,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_neigh,
            enforce_max_strictly=enforce_max_neighbors_strictly,
        )

        c_index = c_index[mask_num_neighbors]
        n_index = n_index[mask_num_neighbors]
        distances = distances[mask_num_neighbors]
        offsets = offsets[mask_num_neighbors]

    return c_index, n_index, distances, offsets


@requires(nvidia_installed, message="Requires `nvalchemiops` to be installed")
def radius_graph_pbc_nvidia(
    data,
    radius: float,
    max_num_neighbors_threshold: int,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
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
    device = data.pos.device
    batch_size = len(data.natoms)

    # Get batch tensor
    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), data.natoms
        )

    # Get PBC tensor
    if pbc is None:
        pbc_tensor = (
            data.pbc
            if hasattr(data, "pbc")
            else torch.tensor([True, True, True], device=device)
        )
    else:
        pbc_tensor = pbc

    # Call core neighbor search function (handles max_neighbors mask internally)
    c_index, n_index, distances, offsets = get_neighbors_nvidia(
        positions=data.pos,
        cell=data.cell,
        pbc=pbc_tensor,
        cutoff=radius,
        max_neigh=max_num_neighbors_threshold,
        method="cell_list",
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        batch=batch,
        natoms=data.natoms,
    )

    # Compute neighbors per image
    edge_batch = batch[c_index]
    num_neighbors_image = torch.zeros(batch_size, dtype=torch.long, device=device)
    num_neighbors_image.scatter_add_(0, edge_batch, torch.ones_like(edge_batch))

    # Format edge_index to match internal methods: [source, target] = [neighbor, center]
    edge_index = torch.stack([n_index, c_index], dim=0)

    return edge_index, offsets, num_neighbors_image


@requires(nvidia_installed, message="Requires `nvalchemiops` to be installed")
def get_neighbors_nvidia_atoms(
    atoms, cutoff: float, max_neigh: int, method: str = "cell_list"
):
    """Performs nearest neighbor search using NVIDIA nvalchemiops and returns edge index, distances,
    and cell offsets.

    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff radius in Angstroms
        max_neigh: Maximum number of neighbors per atom
        method: NVIDIA method to use ("naive" or "cell_list")

    Returns:
        c_index: Center atom indices (numpy array)
        n_index: Neighbor atom indices (numpy array)
        distances: Pairwise distances (numpy array) accounting for PBC
        offsets: Cell offsets (numpy array)
    """
    # Convert Atoms to tensors
    positions = torch.from_numpy(atoms.get_positions()).float()
    cell = torch.from_numpy(np.array(atoms.get_cell(complete=True))).float()
    pbc = torch.from_numpy(np.array(atoms.pbc)).bool()

    # Call tensor function (handles max_neighbors mask internally)
    c_index, n_index, distances, offsets = get_neighbors_nvidia(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_neigh=max_neigh,
        method=method,
        enforce_max_neighbors_strictly=True,
    )

    # Convert back to numpy for backward compatibility
    return (
        c_index.numpy(),
        n_index.numpy(),
        distances.numpy(),
        offsets.numpy(),
    )
