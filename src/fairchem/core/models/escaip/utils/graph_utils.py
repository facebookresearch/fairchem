from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from e3nn.o3._spherical_harmonics import _spherical_harmonics


def get_node_direction_expansion_neighbor(
    direction_vec: torch.Tensor, neighbor_mask: torch.Tensor, lmax: int
):
    """
    Calculate Bond-Orientational Order (BOO) for each node in the graph.
    Ref: Steinhardt, et al. "Bond-orientational order in liquids and glasses." Physical Review B 28.2 (1983): 784.
    Input:
        direction_vec: (num_nodes, num_neighbors, 3)
        neighbor_mask: (num_nodes, num_neighbors)
    Return:
        node_boo: (num_nodes, num_neighbors, lmax + 1)
    """
    # Convert mask to float and expand dimensions
    neighbor_mask = neighbor_mask.float().unsqueeze(-1)

    # Compute spherical harmonics with proper normalization
    edge_sh = _spherical_harmonics(
        lmax=lmax,
        x=direction_vec[:, :, 0],
        y=direction_vec[:, :, 1],
        z=direction_vec[:, :, 2],
    )

    # Normalize spherical harmonics by sqrt(2l+1) to improve numerical stability
    sh_index = torch.arange(lmax + 1, device=edge_sh.device)
    sh_index = torch.repeat_interleave(sh_index, 2 * sh_index + 1)
    edge_sh = edge_sh / torch.clamp(torch.sqrt(2 * sh_index + 1), min=1e-6).unsqueeze(
        0
    ).unsqueeze(0)

    # Compute masked spherical harmonics
    masked_sh = edge_sh * neighbor_mask

    # Compute mean over neighbors with proper normalization
    neighbor_count = neighbor_mask.sum(dim=1)
    neighbor_count = torch.clamp(neighbor_count, min=1)
    node_boo = masked_sh.sum(dim=1) / neighbor_count

    # Compute final BOO with proper normalization
    node_boo_squared = node_boo**2
    # node_boo = scatter(node_boo_squared, sh_index, dim=-1, reduce="sum").sqrt()
    node_boo = compilable_scatter(
        node_boo_squared, sh_index, dim_size=lmax + 1, dim=-1, reduce="sum"
    )
    node_boo = torch.clamp(node_boo, min=1e-6).sqrt()

    return node_boo


def map_sender_receiver_feature(sender_feature, receiver_feature, neighbor_list):
    """
    Map from node features to edge features.
    sender_feature, receiver_feature: (num_nodes, h)
    neighbor_list: (num_nodes, max_neighbors)
    return: sender_features, receiver_features (num_nodes, max_neighbors, h)
    """
    # sender feature
    sender_feature = sender_feature[neighbor_list.flatten()].view(
        neighbor_list.shape[0], neighbor_list.shape[1], -1
    )

    # receiver features
    receiver_feature = receiver_feature.unsqueeze(1).expand(
        -1, neighbor_list.shape[1], -1
    )

    return (sender_feature, receiver_feature)


def legendre_polynomials(x: torch.Tensor, lmax: int) -> torch.Tensor:
    """
    Compute Legendre polynomials P_0..P_{lmax} for each element in x,
    using the standard recursion:
      P_0(x) = 1
      P_1(x) = x
      (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
    x can have any shape; output will have an extra dimension (lmax+1).
    """
    # x shape could be (N, max_neighbors, max_neighbors) in your case
    shape = x.shape  # e.g. (N, M, M)
    P = x.new_zeros(*shape, lmax + 1)  # => (N, M, M, lmax+1)

    # P_0
    P[..., 0] = 1.0

    if lmax >= 1:
        # P_1
        P[..., 1] = x

        for n in range(1, lmax):
            P[..., n + 1] = ((2 * n + 1) * x * P[..., n] - n * P[..., n - 1]) / (n + 1)

    return P


def get_compact_frequency_vectors(
    edge_direction: torch.Tensor, lmax: int, repeating_dimensions: torch.Tensor | list
):
    """
    Calculate a compact representation of frequency vectors.
    Args:
        edge_direction: (N, k, 3) normalized direction vectors
        lmax: maximum l value for spherical harmonics
        repeating_dimensions: (lmax+1,) tensor or list with repeat counts for each l value
    Returns:
        frequency_vectors: (N, k, sum_{l=0..lmax} rep_l * (2l+1))
        flat tensor containing the spherical harmonics matched to repeating dimensions
    """
    # Convert to Python list if tensor to ensure it's treated as a compile-time constant
    if isinstance(repeating_dimensions, torch.Tensor):
        repeat_dims = repeating_dimensions.tolist()
    else:
        repeat_dims = repeating_dimensions

    edge_direction = edge_direction.to(torch.float32)
    # (edge_direction: N, k, 3)
    harmonics = _spherical_harmonics(
        lmax, edge_direction[..., 0], edge_direction[..., 1], edge_direction[..., 2]
    )  # (N, k, (lmax + 1)**2)

    # Create list to hold components
    components = []
    curr_idx = 0

    # Process each l value based on repeating dimensions
    for _l in range(lmax + 1):
        # Get the (2l+1) components for this l value
        sh_dim = 2 * _l + 1
        curr_irrep = harmonics[:, :, curr_idx : curr_idx + sh_dim] / math.sqrt(sh_dim)

        # Get repeat count - use Python list instead of tensor access
        rep_count = repeat_dims[_l]

        # Only add component if rep_count > 0
        if rep_count > 0:
            # Create a component that will match with the expanded q and k
            # (N, k, 2l+1) -> (N, k, rep_count * (2l+1))
            component = curr_irrep.unsqueeze(2).expand(-1, -1, rep_count, -1)
            component = component.reshape(component.shape[0], component.shape[1], -1)

            # Add component to list
            components.append(component)

        # Update index
        curr_idx += sh_dim

    # Concatenate components if we have any, otherwise return empty tensor
    if components:
        return torch.cat(components, dim=-1)
    else:
        # Return empty tensor with proper shape if no components
        return torch.zeros(
            (edge_direction.shape[0], edge_direction.shape[1], 0),
            device=edge_direction.device,
        )


def get_attn_mask(
    edge_direction: torch.Tensor,
    neighbor_mask: torch.Tensor,
    num_heads: int,
    lmax: int,
    use_angle_embedding: str,
):
    """
    Args:
        edge_direction: (num_nodes, max_neighbors, 3)
        neighbor_mask: (num_nodes, max_neighbors)
        num_heads: number of attention heads
    """
    # create a mask for empty neighbors
    batch_size, max_neighbors = neighbor_mask.shape
    attn_mask = torch.zeros(
        batch_size, max_neighbors, max_neighbors, device=neighbor_mask.device
    )
    attn_mask = attn_mask.masked_fill(~neighbor_mask.unsqueeze(1), float("-inf"))

    # repeat the mask for each head
    attn_mask = (
        attn_mask.unsqueeze(1)
        .expand(batch_size, num_heads, max_neighbors, max_neighbors)
        .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
    )

    # get the angle embeddings
    dot_product = torch.matmul(edge_direction, edge_direction.transpose(1, 2))
    dot_product = dot_product.clamp(-1.0, 1.0)
    if use_angle_embedding == "bias":
        angle_embedding = legendre_polynomials(dot_product, lmax)
    else:
        angle_embedding = (
            dot_product.unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(batch_size * num_heads, max_neighbors, max_neighbors)
        )
    return attn_mask, angle_embedding


def get_attn_mask_env(
    src_mask: torch.Tensor,
    num_heads: int,
):
    """
    Args:
        src_mask: (num_nodes, num_neighbors)
        num_heads: number of attention heads
    Output:
        attn_mask: (num_nodes * num_heads, num_neighbors, num_neighbors)
    """
    batch_size, num_neighbors = src_mask.shape
    # broadcast src_mask to attention mask shape
    attn_mask = (
        src_mask.unsqueeze(1)
        .unsqueeze(2)  # (num_nodes, 1, 1, num_neighbors)
        .expand(
            -1, num_heads, num_neighbors, -1
        )  # (num_nodes, num_heads, num_neighbors, num_neighbors)
        .reshape(batch_size * num_heads, num_neighbors, num_neighbors)
    )
    return attn_mask


def pad_batch(
    max_atoms,
    max_batch_size,
    atomic_numbers,
    node_direction_expansion,
    edge_distance_expansion,
    edge_direction,
    neighbor_list,
    neighbor_mask,
    node_batch,
    num_graphs,
    src_mask=None,
):
    """
    Pad the batch to have the same number of nodes in total.
    Needed for torch.compile

    Note: the sampler for multi-node training could sample batchs with different number of graphs.
    The number of sampled graphs could be smaller or larger than the batch size.
    This would cause the model to recompile or core dump.
    Temporarily, setting the max number of graphs to be twice the batch size to mitigate this issue.
    TODO: look into a better way to handle this.
    """
    device = atomic_numbers.device
    num_nodes, _ = neighbor_list.shape
    pad_size = max_atoms - num_nodes
    assert (
        pad_size >= 0
    ), "Number of nodes exceeds the maximum number of nodes per batch"

    # pad the features
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    node_direction_expansion = F.pad(
        node_direction_expansion, (0, 0, 0, pad_size), value=0
    )
    edge_distance_expansion = F.pad(
        edge_distance_expansion, (0, 0, 0, 0, 0, pad_size), value=0
    )
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    neighbor_list = F.pad(neighbor_list, (0, 0, 0, pad_size), value=-1)
    neighbor_mask = F.pad(neighbor_mask, (0, 0, 0, pad_size), value=0)
    node_batch = F.pad(node_batch, (0, pad_size), value=num_graphs)
    if src_mask is not None:
        src_mask = F.pad(src_mask, (0, 0, 0, pad_size), value=0)

    # create the padding mask
    node_padding_mask = torch.ones(max_atoms, dtype=torch.bool, device=device)
    node_padding_mask[num_nodes:] = False

    # TODO look into a better way to handle this
    graph_padding_mask = torch.ones(max_batch_size, dtype=torch.bool, device=device)
    graph_padding_mask[num_graphs:] = False

    return (
        atomic_numbers,
        node_direction_expansion,
        edge_distance_expansion,
        edge_direction,
        neighbor_list,
        neighbor_mask,
        src_mask,
        node_batch,
        node_padding_mask,
        graph_padding_mask,
    )


def unpad_results(results, node_padding_mask, graph_padding_mask):
    """
    Unpad the results to remove the padding.
    """
    unpad_results = {}
    for key in results:
        if results[key].shape[0] == node_padding_mask.shape[0]:
            unpad_results[key] = results[key][node_padding_mask]
        elif results[key].shape[0] == graph_padding_mask.shape[0]:
            unpad_results[key] = results[key][graph_padding_mask]
        elif (
            results[key].shape[0] == node_padding_mask.sum()
            or results[key].shape[0] == graph_padding_mask.sum()
        ):
            unpad_results[key] = results[key]
        else:
            raise ValueError("Unknown padding mask shape")
    return unpad_results


def patch_singleton_atom(edge_direction, neighbor_list, neighbor_mask):
    """
    Patch the singleton atoms in the neighbor_list and neighbor_mask.
    Add a self-loop to the singleton atom
    """

    # Find the singleton atoms
    idx = torch.where(neighbor_mask.sum(dim=-1) == 0)[0]

    # patch edge_direction to unit vector
    edge_direction[idx, 0] = torch.tensor(
        [1.0, 0.0, 0.0], device=edge_direction.device, dtype=edge_direction.dtype
    )

    # patch neighbor_list to itself
    neighbor_list[idx, 0] = idx

    # patch neighbor_mask to itself
    neighbor_mask[idx, 0] = 1

    return edge_direction, neighbor_list, neighbor_mask


def compilable_scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    torch_scatter scatter function with compile support.
    Modified from torch_geometric.utils.scatter_.
    """

    def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
        dim = ref.dim() + dim if dim < 0 else dim
        size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
        return src.view(size).expand_as(ref)

    dim = src.dim() + dim if dim < 0 else dim
    size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]

    if reduce == "sum" or reduce == "add":
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == "mean":
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    raise ValueError(f"Invalid reduce option '{reduce}'.")


def get_displacement_and_cell(data, regress_stress, regress_forces, direct_forces):
    """
    Get the displacement and cell from the data.
    For gradient-based forces/stress
    ref: https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/escn_md.py#L298
    """
    displacement = None
    orig_cell = None
    if regress_stress and not direct_forces:
        displacement = torch.zeros(
            (3, 3),
            dtype=data["pos"].dtype,
            device=data["pos"].device,
        )
        num_batch = len(data["natoms"])
        displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
        displacement.requires_grad = True
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        if data["pos"].requires_grad is False:
            data["pos"].requires_grad = True
        data["pos_original"] = data["pos"]
        data["pos"] = data["pos"] + torch.bmm(
            data["pos"].unsqueeze(-2),
            torch.index_select(symmetric_displacement, 0, data["batch"]),
        ).squeeze(-2)

        orig_cell = data["cell"]
        data["cell"] = data["cell"] + torch.bmm(data["cell"], symmetric_displacement)
    if (
        not regress_stress
        and regress_forces
        and not direct_forces
        and data["pos"].requires_grad is False
    ):
        data["pos"].requires_grad = True
    return displacement, orig_cell
