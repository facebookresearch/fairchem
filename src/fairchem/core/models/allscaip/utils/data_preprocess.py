from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from e3nn.o3._spherical_harmonics import _spherical_harmonics

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.allscaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
    )

from fairchem.core.models.allscaip.custom_types import GraphAttentionData
from fairchem.core.models.allscaip.utils.allscaip_radius_graph import (
    biknn_radius_graph,
)
from fairchem.core.models.escaip.utils.graph_utils import compilable_scatter
from fairchem.core.models.escaip.utils.radius_graph import (
    envelope_fn,
    safe_norm,
    safe_normalize,
)
from fairchem.core.models.escaip.utils.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)


def get_edge_distance_expansion(
    molecular_graph_cfg: MolecularGraphConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    edge_distance: torch.Tensor,
    device: torch.device,
):
    # edge distance expansion
    expansion_func = {
        "gaussian": GaussianSmearing,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[molecular_graph_cfg.distance_function]

    edge_distance_expansion_func = expansion_func(
        0.0,
        molecular_graph_cfg.max_radius,
        gnn_cfg.edge_distance_expansion_size,
        basis_width_scalar=2.0,
    ).to(device)

    # edge distance expansion (ref: scn)
    # (num_nodes, num_neighbors, edge_distance_expansion_size)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance.flatten())
    return edge_distance_expansion


def get_frequency_vectors(
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    edge_direction: torch.Tensor,
):
    """
    Calculate frequency vectors for neighbor attention using spherical harmonics.

    This function generates compact frequency vector representations by computing spherical
    harmonics for edge directions and organizing them according to specified repeat patterns.
    The spherical harmonics are normalized and expanded based on the frequency list configuration
    to create attention-compatible feature vectors.

    The function validates that the sum of frequency list values equals the head dimension
    (hidden_size / attention_heads) to ensure proper attention head compatibility.

    Args:
        global_cfg: Global configuration containing hidden_size
        gnn_cfg: GNN configuration containing freequency_list and atten_num_heads
        edge_direction: (N, k, 3) normalized direction vectors between atoms
        device: torch device for tensor operations

    Returns:
        frequency_vectors: (N, k, sum_{l=0..lmax} rep_l * (2l+1)) tensor containing
            spherical harmonics organized by the frequency list pattern

    Raises:
        AssertionError: If sum of freequency_list doesn't equal head_dim
    """
    # Validate configuration compatibility
    head_dim = global_cfg.hidden_size // gnn_cfg.atten_num_heads
    sum_repeats = sum(gnn_cfg.freequency_list)
    assert sum_repeats == head_dim, (
        f"Sum of freequency_list must equal head_dim ({head_dim}), "
        f"but got sum={sum_repeats}. Please adjust freequency_list."
    )

    # Use the Python list directly for better torch.compile compatibility
    lmax = len(gnn_cfg.freequency_list) - 1
    repeat_dims = gnn_cfg.freequency_list

    # Convert edge direction to float32 for spherical harmonics computation
    edge_direction = edge_direction.to(torch.float32)

    # Compute spherical harmonics for all l values up to lmax
    # (edge_direction: N, k, 3) -> (N, k, (lmax + 1)**2)
    harmonics = _spherical_harmonics(
        lmax, edge_direction[..., 0], edge_direction[..., 1], edge_direction[..., 2]
    )

    # Create list to hold components for each l value
    components = []
    curr_idx = 0

    # Process each l value based on repeating dimensions
    for _l in range(lmax + 1):
        # Get the (2l+1) components for this l value
        sh_dim = 2 * _l + 1
        curr_irrep = harmonics[:, :, curr_idx : curr_idx + sh_dim] / math.sqrt(sh_dim)

        # Get repeat count from frequency list
        rep_count = repeat_dims[_l]

        # Only add component if rep_count > 0
        if rep_count > 0:
            # Create a component that will match with the expanded q and k
            # (N, k, 2l+1) -> (N, k, rep_count * (2l+1))
            component = curr_irrep.unsqueeze(2).expand(-1, -1, rep_count, -1)
            component = component.reshape(component.shape[0], component.shape[1], -1)

            # Add component to list
            components.append(component)

        # Update index for next l value
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


def get_node_attention_mask(
    node_batch: torch.Tensor,
    dist_pairwise: torch.Tensor | None,
    n_freq: int = 32,
    r_min: float = 0.25,
    r_max: float = 30.0,
    use_sincx_mask: bool = True,
    single_system_no_padding: bool = False,
):
    N_pad = node_batch.size(0)

    # Skip base_mask for single system without padding (no graph masking needed)
    if single_system_no_padding:
        if not use_sincx_mask:
            return None, None, None
        # Still need sincx computation but no base_mask
        base_mask = None
        valid_mask = None
    else:
        # get base attention mask
        # (N_pad,1) == (1,N_pad) -> (N_pad,N_pad) bool
        same_graph = node_batch.unsqueeze(1) == node_batch.unsqueeze(0)
        real = node_batch.unsqueeze(0) != -1  # (1,N_pad) bool
        real2 = node_batch.unsqueeze(1) != -1  # (N_pad,1)

        valid_mask = same_graph & real & real2  # True where attention allowed

        # filter distance larger than 15 A
        # valid_mask = valid_mask & (dist_pairwise < 15.0)

        base_mask = torch.zeros(
            (N_pad, N_pad), dtype=torch.float32, device=node_batch.device
        )
        neg_inf = torch.finfo(base_mask.dtype).min
        base_mask = base_mask.masked_fill(~valid_mask, neg_inf)  # (N_pad,N_pad)
        base_mask = base_mask.unsqueeze(0)  # (1,N_pad,N_pad)

        # Skip sincx computation if not needed (lazy evaluation)
        if not use_sincx_mask:
            return None, base_mask, None

    # Euclidean Rotary Encoding (Sinc Kernels)
    # Frequencies
    omega_min = math.pi / (4.0 * r_max)
    omega_max = math.pi / (r_min)
    omega = torch.logspace(
        math.log10(omega_min),
        math.log10(omega_max),
        n_freq,
        device=node_batch.device,
        dtype=torch.float32,
    )  # (K,)

    # x = r * ω
    x = dist_pairwise.unsqueeze(-1) * omega.view(1, 1, -1)  # (N,N,K)

    # Stable sinc in fp32 (Taylor Expansion 4th order)
    sincx = torch.empty_like(x)
    small = x.abs() < 1e-4
    x_small = x[small]
    x2 = x_small * x_small
    sincx[small] = 1 - x2 / 6 + (x2 * x2) / 120
    sincx[~small] = torch.sin(x[~small]) / x[~small]

    return sincx, base_mask, valid_mask


def data_preprocess_radius_graph(
    data: AtomicData,
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    molecular_graph_cfg: MolecularGraphConfigs,
) -> GraphAttentionData:
    # Check if we should preprocess on CPU
    original_device = data.pos.device
    preprocess_on_cpu = (
        molecular_graph_cfg.preprocess_on_cpu and original_device.type == "cuda"
    )

    if preprocess_on_cpu:
        # Move data to CPU for preprocessing
        data = data.cpu()
        preprocess_device = torch.device("cpu")
    else:
        preprocess_device = original_device

    # atomic numbers
    atomic_numbers = data.atomic_numbers.long()

    # Only compute dist_pairwise if needed for sincx mask in node attention
    need_dist_pairwise = global_cfg.use_node_path and gnn_cfg.use_sincx_mask

    # generate graph
    (
        dist_pairwise,  # (num_nodes, num_nodes) or None
        disp,  # (num_nodes, num_neighbors, 3)
        src_env,
        dst_env,
        src_index,  # (2, num_nodes, num_neighbors)
        dst_index,
        neighbor_index,  # (2, num_nodes, num_neighbors)
    ) = biknn_radius_graph(  # type: ignore
        data,
        molecular_graph_cfg.max_radius,  # type: ignore[arg-type]
        molecular_graph_cfg.knn_k,
        molecular_graph_cfg.knn_soft,
        molecular_graph_cfg.knn_sigmoid_scale,
        molecular_graph_cfg.knn_lse_scale,
        molecular_graph_cfg.knn_use_low_mem,
        molecular_graph_cfg.knn_pad_size if global_cfg.use_padding else None,
        preprocess_device,
        compute_dist_pairwise=need_dist_pairwise,
        use_chunked=molecular_graph_cfg.use_chunked_graph,
        chunk_size=molecular_graph_cfg.graph_chunk_size,
    )

    num_nodes, max_num_neighbors, _ = disp.shape
    edge_direction = safe_normalize(disp)  # (num_nodes, num_neighbors, 3)
    edge_distance = safe_norm(disp)  # (num_nodes, num_neighbors)
    src_mask = envelope_fn(src_env, molecular_graph_cfg.use_envelope)
    dst_mask = envelope_fn(dst_env, molecular_graph_cfg.use_envelope)

    # pad batch
    if global_cfg.use_padding:
        num_graphs = data.num_graphs
        num_nodes = atomic_numbers.shape[0]
        max_num_nodes = molecular_graph_cfg.max_atoms
        max_batch_size = molecular_graph_cfg.max_batch_size
        (
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
        ) = pad_batch(
            max_atoms=molecular_graph_cfg.max_atoms,
            max_batch_size=molecular_graph_cfg.max_batch_size,
            atomic_numbers=atomic_numbers,
            charge=data.charge,
            spin=data.spin,
            edge_direction=edge_direction,
            edge_distance=edge_distance,
            neighbor_index=neighbor_index,
            src_mask=src_mask,
            dst_mask=dst_mask,
            src_index=src_index,
            dst_index=dst_index,
            dist_pairwise=dist_pairwise,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
        )
    else:
        num_graphs = data.num_graphs
        num_nodes = atomic_numbers.shape[0]
        max_num_nodes = num_nodes
        max_batch_size = num_graphs
        node_batch = data.batch
        charge = data.charge
        spin = data.spin

    # edge distance expansion (num_nodes, num_neighbors, edge_distance_expansion_size)
    edge_distance_expansion = get_edge_distance_expansion(
        molecular_graph_cfg, gnn_cfg, edge_distance, data.pos.device
    ).view(max_num_nodes, max_num_neighbors, gnn_cfg.edge_distance_expansion_size)

    # Compute spherical harmonics for edge direction
    edge_direction_expansion = _spherical_harmonics(
        lmax=gnn_cfg.edge_direction_expansion_size - 1,
        x=edge_direction[:, :, 0],
        y=edge_direction[:, :, 1],
        z=edge_direction[:, :, 2],
    )

    # node direction expansion (num_nodes, num_neighbors, lmax + 1)
    node_direction_expansion = get_node_direction_expansion_neighbor(
        direction_vec=edge_direction,
        neighbor_mask=src_mask != -torch.inf,
        lmax=gnn_cfg.node_direction_expansion_size - 1,
    )

    # get frequency vectors for neighbor attention
    if gnn_cfg.use_freq_mask:
        frequency_vectors = get_frequency_vectors(global_cfg, gnn_cfg, edge_direction)
    else:
        frequency_vectors = None

    # get attention mask for node attention
    if global_cfg.use_node_path:
        sincx, base_mask, valid_mask = get_node_attention_mask(
            node_batch,
            dist_pairwise,
            gnn_cfg.attn_num_freq,
            use_sincx_mask=gnn_cfg.use_sincx_mask,
            single_system_no_padding=global_cfg.single_system_no_padding,
        )
    else:
        sincx = None
        base_mask = None
        valid_mask = None

    # change the -1 in node_batch to avoid indexing error
    node_batch = node_batch.masked_fill(node_batch == -1, 0)

    # Create node padding mask: 1.0 for real nodes, 0.0 for padded nodes
    # Use arange comparison which is compile-friendly
    node_indices = torch.arange(max_num_nodes, device=node_batch.device)
    node_padding_mask = (node_indices < num_nodes).to(torch.float32)

    if gnn_cfg.atten_name in ["memory_efficient", "flash", "math"]:
        if (
            gnn_cfg.atten_name in ["memory_efficient", "flash"]
            and not global_cfg.direct_forces
        ):
            logging.warning(
                "Fallback to math attention for gradient based force prediction"
            )
            gnn_cfg.atten_name = "math"
        torch.backends.cuda.enable_flash_sdp(gnn_cfg.atten_name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(
            gnn_cfg.atten_name == "memory_efficient"
        )
        # torch.backends.cuda.enable_math_sdp(gnn_cfg.atten_name == "math")
    else:
        raise NotImplementedError(
            f"Attention name {gnn_cfg.atten_name} not implemented"
        )

    # construct input data
    x = GraphAttentionData(
        atomic_numbers=atomic_numbers,
        charge=charge,
        spin=spin,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction_expansion=edge_direction_expansion,
        edge_direction=edge_direction,
        src_neighbor_attn_mask=src_mask,
        dst_neighbor_attn_mask=dst_mask,
        src_index=src_index,
        dst_index=dst_index,
        frequency_vectors=frequency_vectors,
        node_base_attn_mask=base_mask,
        node_sincx_matrix=sincx,
        node_valid_mask=valid_mask,
        neighbor_index=neighbor_index,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        max_batch_size=max_batch_size,
        num_graphs=num_graphs,
        max_num_nodes=max_num_nodes,
        num_nodes=num_nodes,
    )

    # Move results back to original device if preprocessed on CPU
    if preprocess_on_cpu:
        x = x.to(original_device)

    return x


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
    if dist_pairwise is not None:
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
