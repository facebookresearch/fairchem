"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from fairchem.core.common.utils import segment_coo

# Try to import actual torch_sparse, with fallback to None if not available
try:
    import torch_sparse

    TORCH_SPARSE_AVAILABLE = True
except ImportError:
    torch_sparse = None
    TORCH_SPARSE_AVAILABLE = False


def get_sparse_tensor_class(use_torch_sparse: bool = False):
    """
    Get the appropriate SparseTensor class based on the flag.

    Args:
        use_torch_sparse: If True, use actual torch_sparse.SparseTensor.
                         If False, use custom implementation.

    Returns:
        SparseTensor class to use
    """
    if use_torch_sparse:
        if not TORCH_SPARSE_AVAILABLE:
            warnings.warn(
                "torch_sparse is not available. Falling back to custom implementation. "
                "To use torch_sparse, install it with: pip install torch-sparse",
                RuntimeWarning,
            )
            return CustomSparseTensor
        return torch_sparse.SparseTensor
    else:
        return CustomSparseTensor


class CustomSparseTensor:
    """
    A simplified implementation of SparseTensor from torch_sparse.

    This class provides the basic functionality needed for the gemnet_oc model.
    """

    def __init__(self, row, col, value=None, sparse_sizes=None):
        """
        Initialize a CustomSparseTensor.

        Args:
            row: Row indices
            col: Column indices
            value: Values at the specified indices (optional)
            sparse_sizes: Size of the sparse matrix (optional)
        """
        self.row = row
        self.col = col
        self.value = value if value is not None else torch.ones_like(row)

        if sparse_sizes is None:
            sparse_sizes = (row.max().item() + 1, col.max().item() + 1)
        self.sparse_sizes = sparse_sizes

    def __getitem__(self, idx):
        """
        Get a slice of the sparse tensor.

        This is used for operations like adj[idx_t] in the code.
        The result should have:
        - value() returning the original values from the selected rows
        - row() returning the position in idx for each value
        """
        if isinstance(idx, torch.Tensor):
            # For batched indexing
            # Find all entries where row is in idx
            result_rows = []
            result_cols = []
            result_values = []

            for i, target_row in enumerate(idx):
                # Find all entries with this row index
                mask = self.row == target_row
                if torch.any(mask):
                    # Add entries with row index set to position in idx
                    result_rows.append(
                        torch.full(
                            (mask.sum(),),
                            i,
                            dtype=self.row.dtype,
                            device=self.row.device,
                        )
                    )
                    result_cols.append(self.col[mask])
                    result_values.append(self.value[mask])

            if result_rows:
                new_rows = torch.cat(result_rows)
                new_cols = torch.cat(result_cols)
                new_values = torch.cat(result_values)
            else:
                # No matches found
                new_rows = torch.empty(0, dtype=self.row.dtype, device=self.row.device)
                new_cols = torch.empty(0, dtype=self.col.dtype, device=self.col.device)
                new_values = torch.empty(
                    0, dtype=self.value.dtype, device=self.value.device
                )

            return CustomSparseStorage(
                row=new_rows,
                col=new_cols,
                value=new_values,
                sparse_sizes=(idx.size(0), self.sparse_sizes[1]),
            )
        else:
            # For single index
            mask = self.row == idx
            return CustomSparseStorage(
                row=torch.zeros_like(
                    self.row[mask]
                ),  # All rows become 0 for single index
                col=self.col[mask],
                value=self.value[mask],
                sparse_sizes=(1, self.sparse_sizes[1]),
            )

    def coo(self):
        """Return the COO representation of the sparse tensor."""
        return self.row, self.col, self.value


class CustomSparseStorage:
    """
    A helper class to mimic the storage object of torch_sparse.SparseTensor.
    """

    def __init__(self, row, col, value, sparse_sizes):
        self._row = row
        self.col = col
        self._value = value
        self.sparse_sizes = sparse_sizes
        # For compatibility with torch_sparse, storage should point to itself
        self.storage = self

    def value(self):
        """Return the values."""
        return self._value

    def row(self):
        """Return the row indices."""
        return self._row


# For backward compatibility - use get_sparse_tensor_class() instead
def SparseTensor(*args, **kwargs):
    """
    Backward compatibility function.
    Use get_sparse_tensor_class() for new code.
    """
    warnings.warn(
        "Direct use of SparseTensor is deprecated. Use get_sparse_tensor_class() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CustomSparseTensor(*args, **kwargs)


def segment_csr(src, indptr, reduce="sum"):
    """
    This function performs a segment-wise reduction of the src tensor along the first dimension,
    where the segments are defined by the indptr tensor.

    Args:
        src: Source tensor of shape [N, *]
        indptr: Index pointer tensor of shape [M+1] where M is the number of segments
        reduce: Reduction method ('sum', 'mean', 'min', 'max')

    Returns:
        Tensor of shape [M, *] with segment-wise reduction
    """

    # Create output tensor with proper gradient requirements
    dim_size = indptr.size(0) - 1
    out_shape = [dim_size] + list(src.shape[1:])

    if reduce == "sum" or reduce == "mean":
        out = torch.zeros(
            out_shape,
            dtype=src.dtype,
            device=src.device,
            requires_grad=src.requires_grad,
        )
    elif reduce == "min":
        out = torch.full(
            out_shape,
            float("inf"),
            dtype=src.dtype,
            device=src.device,
            requires_grad=src.requires_grad,
        )
    elif reduce == "max":
        out = torch.full(
            out_shape,
            float("-inf"),
            dtype=src.dtype,
            device=src.device,
            requires_grad=src.requires_grad,
        )
    else:
        raise ValueError(f"Unsupported reduction: {reduce}")

    # Perform segment-wise reduction
    for i in range(dim_size):
        start_idx = indptr[i].item()
        end_idx = indptr[i + 1].item()

        if start_idx == end_idx:
            continue

        segment = src[start_idx:end_idx]

        if reduce == "sum":
            out[i] = torch.sum(segment, dim=0)
        elif reduce == "mean":
            out[i] = torch.mean(segment, dim=0)
        elif reduce == "min":
            out[i] = torch.min(segment, dim=0)[0]
        elif reduce == "max":
            out[i] = torch.max(segment, dim=0)[0]

    return out


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    return id_steps.cumsum(0)


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: int = 0,
    repeat_inc: int = 0,
) -> torch.Tensor:
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    return id_ar.cumsum(0)


def masked_select_sparsetensor_flat(src, mask, use_torch_sparse: bool = False):
    """
    Apply a mask to a SparseTensor.

    Args:
        src: Source SparseTensor (either CustomSparseTensor or torch_sparse.SparseTensor)
        mask: Boolean mask to apply
        use_torch_sparse: Whether to use torch_sparse or custom implementation

    Returns:
        Masked SparseTensor
    """
    if use_torch_sparse and TORCH_SPARSE_AVAILABLE and hasattr(src, "coo"):
        # For actual torch_sparse.SparseTensor
        row, col, value = src.coo()
    elif hasattr(src, "coo"):
        # For CustomSparseTensor
        row, col, value = src.coo()
    else:
        # Fallback for other implementations
        row, col, value = src.row, src.col, src.value

    row = row[mask]
    col = col[mask]
    value = value[mask]

    # Get sparse_sizes - handle both property and method
    if callable(src.sparse_sizes):
        sparse_sizes = src.sparse_sizes()  # torch_sparse.SparseTensor
    else:
        sparse_sizes = src.sparse_sizes  # CustomSparseTensor

    SparseTensorClass = get_sparse_tensor_class(use_torch_sparse)
    return SparseTensorClass(row=row, col=col, value=value, sparse_sizes=sparse_sizes)


def calculate_interatomic_vectors(R, id_s, id_t, offsets_st):
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Arguments
    ---------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    # ReLU prevents negative numbers in sqrt

    V_st = Rt - Rs if offsets_st is None else Rt - Rs + offsets_st  # s -> t
    D_st = torch.sqrt(torch.sum(V_st**2, dim=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_clamped(x, y) -> torch.Tensor:
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def get_angle(R_ac, R_ab) -> torch.Tensor:
    """Calculate angles between atoms c -> a <- b.

    Arguments
    ---------
        R_ac: Tensor, shape = (N, 3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=-1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab, dim=-1).norm(dim=-1)  # shape = (N,)
    y = y.clamp(min=1e-9)  # Avoid NaN gradient for y = (0,0,0)

    return torch.atan2(y, x)


def vector_rejection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N, 3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def get_projected_angle(R_ab, P_n, eps: float = 1e-4) -> torch.Tensor:
    """
    Project the vector R_ab onto a plane with normal vector P_n,
    then calculate the angle w.r.t. the (x [cross] P_n),
    or (y [cross] P_n) if the former would be ill-defined/numerically unstable.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.
        eps: float
            Norm of projection below which to use the y-axis instead of x.

    Returns
    -------
        angle_ab: Tensor, shape = (N)
            Angle on plane w.r.t. x- or y-axis.
    """
    R_ab_proj = torch.cross(R_ab, P_n, dim=-1)

    # Obtain axis defining the angle=0
    x = P_n.new_tensor([[1, 0, 0]]).expand_as(P_n)
    zero_angle = torch.cross(x, P_n, dim=-1)

    use_y = torch.norm(zero_angle, dim=-1) < eps
    P_n_y = P_n[use_y]
    y = P_n_y.new_tensor([[0, 1, 0]]).expand_as(P_n_y)
    y_cross = torch.cross(y, P_n_y, dim=-1)
    zero_angle[use_y] = y_cross

    angle = get_angle(zero_angle, R_ab_proj)

    # Flip sign of angle if necessary to obtain clock-wise angles
    cross = torch.cross(zero_angle, R_ab_proj, dim=-1)
    flip_sign = torch.sum(cross * P_n, dim=-1) < 0
    angle[flip_sign] = -angle[flip_sign]

    return angle


def mask_neighbors(neighbors, edge_mask):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    return segment_csr(edge_mask.long(), neighbors_old_indptr)


def get_neighbor_order(num_atoms: int, index, atom_distance) -> torch.Tensor:
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    """
    device = index.device

    # Get sorted index and inverse sorting
    # Necessary for index_sort_map
    index_sorted, index_order = torch.sort(index)
    index_order_inverse = torch.argsort(index_order)

    # Get number of neighbors
    ones = index_sorted.new_ones(1).expand_as(index_sorted)
    num_neighbors = segment_coo(ones, index_sorted, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index_sorted * max_num_neighbors
        + torch.arange(len(index_sorted), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Offset index_sort so that it indexes into index_sorted
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # Create indices specifying the order in index_sort
    order_peratom = torch.arange(max_num_neighbors, device=device)[None, :].expand_as(
        mask_finite
    )
    order_peratom = torch.masked_select(order_peratom, mask_finite)

    # Re-index to obtain order value of each neighbor in index_sorted
    order = torch.zeros(len(index), device=device, dtype=torch.long)
    order[index_sort] = order_peratom

    return order[index_order_inverse]


def get_inner_idx(idx, dim_size):
    """
    Assign an inner index to each element (neighbor) with the same index.
    For example, with idx=[0 0 0 1 1 1 1 2 2] this returns [0 1 2 0 1 2 3 0 1].
    These indices allow reshape neighbor indices into a dense matrix.
    idx has to be sorted for this to work.
    """
    ones = idx.new_ones(1).expand_as(idx)
    num_neighbors = segment_coo(ones, idx, dim_size=dim_size)
    return ragged_range(num_neighbors)


def get_edge_id(edge_idx, cell_offsets, num_atoms: int):
    cell_basis = cell_offsets.max() - cell_offsets.min() + 1
    cell_id = (
        (cell_offsets * cell_offsets.new_tensor([[1, cell_basis, cell_basis**2]]))
        .sum(-1)
        .long()
    )
    return edge_idx[0] + edge_idx[1] * num_atoms + cell_id * num_atoms**2
