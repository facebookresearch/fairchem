from fairchem.core.common import gp_utils
import numpy as np
import torch
import logging

from dataclasses import dataclass
from enum import Enum


def check_or_get_rank_world_size(
    rank: int | None = None, world_size: int | None = None
):
    if rank is None and world_size is None:
        # assume we want to automatically get rank/world_size through GP
        assert gp_utils.initialized()
        rank = gp_utils.get_gp_rank()
        world_size = gp_utils.get_gp_world_size()
    else:
        # assume the user passes in the intended information
        assert isinstance(rank, int)
        assert isinstance(world_size, int)
        assert rank < world_size
        assert rank >= 0
        assert world_size > 0
    return rank, world_size


@dataclass
class PartitionStrategy(str, Enum):
    INDEX_SPLIT = "index_split"
    SLICE = "slice"
    CUBE = "cube"
    KMEANS = "kmeans"


def partition_atoms_to_grid(coords: torch.Tensor, k: int):
    """
    Partition N atoms into a kxkxk grid and return atom indices with their cell assignments.

    Args:
        coords (torch.Tensor): [N, 3] tensor of atom coordinates
        k (int): Number of cells per dimension (creates kxkxk grid)

    Returns:
        cell_indices (torch.Tensor): [N] tensor of cell indices for each atom
    """
    N = coords.shape[0]
    k = int(k)  # Ensure k is a Python int, not numpy scalar
    total_cells = k**3

    # Always start with round-robin to guarantee each cell gets at least 1 atom
    cell_indices = torch.arange(N, device=coords.device) % total_cells

    # If we have more atoms than cells, reassign the extra atoms based on spatial location
    if N > total_cells:
        # Find bounding box of all atoms
        min_coords = torch.min(coords, dim=0)[0]  # [3]
        max_coords = torch.max(coords, dim=0)[0]  # [3]

        # Calculate cell size for each dimension
        grid_size = (max_coords - min_coords) / k  # [3]

        # Handle edge case where all atoms have same coordinate in a dimension
        grid_size = torch.where(grid_size == 0, torch.ones_like(grid_size), grid_size)

        # Normalize coordinates to [0, k) range
        normalized_coords = (coords - min_coords) / grid_size

        # Clamp to handle floating point precision issues at boundaries
        normalized_coords = torch.clamp(normalized_coords, 0, k - 1e-6)

        # Convert to grid indices
        grid_coords = normalized_coords.long()  # [N, 3]

        # Convert 3D grid coordinates to 1D cell indices
        spatial_cell_indices = (
            grid_coords[:, 0] + grid_coords[:, 1] * k + grid_coords[:, 2] * k * k
        )

        # Only reassign atoms beyond the first total_cells atoms based on spatial location
        cell_indices[total_cells:] = spatial_cell_indices[total_cells:]

    return cell_indices


def partition_atoms_to_slices(coords: torch.Tensor, K: int, axis: int = 0):
    """
    Partition N atoms into K slices along a specified axis.

    Args:
        coords (torch.Tensor): [N, 3] tensor of atom coordinates
        K (int): Number of slices to create
        axis (int): Axis along which to create slices (0=x, 1=y, 2=z)

    Returns:
        slice_indices (torch.Tensor): [N] tensor of slice indices for each atom
    """
    N = coords.shape[0]
    device = coords.device

    # Extract coordinates along the specified axis
    axis_coords = coords[:, axis]  # [N]

    # Find min and max coordinates along the axis
    min_coord = torch.min(axis_coords)
    max_coord = torch.max(axis_coords)

    # Handle edge case where all atoms have same coordinate
    if min_coord == max_coord:
        # All atoms go to slice 0
        slice_indices = torch.zeros(N, dtype=torch.long, device=device)
        return slice_indices

    # Calculate slice width
    slice_width = (max_coord - min_coord) / K
    logging.debug(f"slice_width: {slice_width}")

    # Assign atoms to slices
    # Normalize coordinates to [0, K) range
    normalized_coords = (axis_coords - min_coord) / slice_width

    # Clamp to handle floating point precision issues at boundaries
    normalized_coords = torch.clamp(normalized_coords, 0, K - 1e-6)

    # Convert to slice indices
    slice_indices = normalized_coords.long()  # [N]
    logging.debug(f"slice_indices: {slice_indices}")

    return slice_indices


def partition_atoms_kmeans(
    coords: torch.Tensor,
    k: int,
    max_iters: int = 10,
    tol: float = 1e-4,
    seed: int | None = None,
):
    """
    Partition N atoms into k clusters using k-means clustering.

    Uses fast-pytorch-kmeans for GPU-accelerated k-means clustering.

    Args:
        coords (torch.Tensor): [N, 3] tensor of atom coordinates
        k (int): Number of clusters (ranks) to create
        max_iters (int): Maximum number of k-means iterations
        tol (float): Convergence tolerance for centroid movement
        seed (int | None): Random seed for reproducible initialization

    Returns:
        cluster_indices (torch.Tensor): [N] tensor of cluster indices for each atom
    """
    # experimental
    from fast_pytorch_kmeans import KMeans

    N = coords.shape[0]
    device = coords.device

    # Handle edge case where k >= N
    if k >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    # Set seed for reproducibility (default to 42 if not provided)
    # This ensures consistent cluster assignments across calls
    # TODO Fix this so we dont reset the global seed
    torch.manual_seed(seed if seed is not None else 42)

    # Use fast-pytorch-kmeans
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iters,
        tol=tol,
        mode="euclidean",
    )

    # Fit and get cluster assignments
    cluster_indices = kmeans.fit_predict(coords)

    logging.debug(f"fast-pytorch-kmeans completed")
    return cluster_indices.long()


def partition_atoms_by_position(
    positions: torch.Tensor,
    method: PartitionStrategy,
    rank: int | None = None,
    world_size: int | None = None,
):
    rank, world_size = check_or_get_rank_world_size(rank, world_size)

    if method is PartitionStrategy.CUBE:
        rounded_cbrt = np.round(np.cbrt(world_size))
        assert (
            rounded_cbrt**3 == world_size
        ), "CUBE partitioning requires gp world size to be an integer cube root"
        # first get the assignment of atoms to cell index given k
        # where index ranges from [0, k^3 - 1]
        rank_indices = partition_atoms_to_grid(positions, rounded_cbrt)
    elif method is PartitionStrategy.SLICE:
        # partition of exactly gp_world_size horizontal slices along an axis
        rank_indices = partition_atoms_to_slices(positions, world_size)
    elif method is PartitionStrategy.INDEX_SPLIT:
        # original GP split code
        # node_partition = torch.tensor_split(
        #     torch.arange(len(positions)).to(positions.device),
        #     world_size,
        # )[rank]
        chunks = torch.tensor_split(
            torch.arange(len(positions), device=positions.device),
            world_size,
        )
        for i, t in enumerate(chunks):
            t.fill_(i)
        rank_indices = torch.cat(chunks)
    elif method is PartitionStrategy.KMEANS:
        # partition using k-means clustering into exactly world_size clusters
        rank_indices = partition_atoms_kmeans(positions, world_size)
    else:
        raise ValueError(f"method {method} not recognized")

    node_partition = (rank_indices == rank).nonzero().squeeze()
    if node_partition.dim() == 0:
        node_partition = node_partition.unsqueeze(0)
    return node_partition, rank_indices
