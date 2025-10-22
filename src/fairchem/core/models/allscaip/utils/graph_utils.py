from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fairchem.core.models.allscaip.custom_types import GraphAttentionData


def pad_batch(
    max_atoms,
    max_batch_size,
    atomic_numbers,
    charge,
    spin,
    edge_direction,
    edge_distance,
    neighbor_index,
    node_batch,
    num_graphs,
    src_mask,
    dst_mask,
    src_index,
    dst_index,
    dist_pairwise,
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
    _, num_nodes, _ = neighbor_index.shape
    pad_size = max_atoms - num_nodes
    assert (
        pad_size >= 0
    ), "Number of nodes exceeds the maximum number of nodes per batch"
    assert (
        max_batch_size >= num_graphs
    ), "Number of graphs exceeds the maximum batch size"

    # pad the features
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    edge_distance = F.pad(edge_distance, (0, 0, 0, pad_size), value=0)
    neighbor_index = F.pad(neighbor_index, (0, 0, 0, pad_size), value=-1)
    node_batch = F.pad(node_batch, (0, pad_size), value=-1)
    src_mask = F.pad(src_mask, (0, 0, 0, pad_size), value=-torch.inf)
    dst_mask = F.pad(dst_mask, (0, 0, 0, pad_size), value=-torch.inf)
    src_index = F.pad(src_index, (0, 0, 0, pad_size), value=-1)
    dst_index = F.pad(dst_index, (0, 0, 0, pad_size), value=-1)
    dist_pairwise = F.pad(dist_pairwise, (0, pad_size, 0, pad_size), value=0)
    if charge is not None:
        charge = F.pad(charge, (0, max_batch_size - num_graphs), value=0)
    else:
        charge = torch.zeros(max_batch_size, dtype=torch.float, device=device)
    if spin is not None:
        spin = F.pad(spin, (0, max_batch_size - num_graphs), value=0)
    else:
        spin = torch.zeros(max_batch_size, dtype=torch.float, device=device)

    return (
        atomic_numbers,
        charge,
        spin,
        edge_direction,
        edge_distance,
        neighbor_index,
        src_mask,
        dst_mask,
        src_index,
        dst_index,
        dist_pairwise,
        node_batch,
    )


def unpad_results(results: dict, data: GraphAttentionData):
    """
    Unpad the results to remove the padding.
    """
    unpad_results = {}
    for key in results:
        if results[key].shape[0] == data.max_num_nodes:
            # Node-level results
            unpad_results[key] = results[key][: data.num_nodes]
        elif results[key].shape[0] == data.max_batch_size:
            # Graph-level results
            unpad_results[key] = results[key][: data.num_graphs]
        elif (
            results[key].shape[0] == data.num_nodes
            or results[key].shape[0] == data.num_graphs
        ):
            # Results already unpadded
            unpad_results[key] = results[key]
        else:
            raise ValueError(
                f"Unknown padding mask shape for key '{key}': "
                f"result shape {results[key].shape}, "
                f"data shape {data.num_nodes}, {data.num_graphs}"
            )
    return unpad_results


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
