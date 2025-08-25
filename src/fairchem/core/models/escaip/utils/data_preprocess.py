from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.escaip.configs import (
        GlobalConfigs,
        GraphNeuralNetworksConfigs,
        MolecularGraphConfigs,
    )
from fairchem.core.models.escaip.custom_types import GraphAttentionData
from fairchem.core.models.escaip.utils.graph_utils import (
    get_attn_mask,
    get_attn_mask_env,
    get_compact_frequency_vectors,
    get_node_direction_expansion_neighbor,
    pad_batch,
)
from fairchem.core.models.escaip.utils.radius_graph import (
    biknn_radius_graph,
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


def data_preprocess_radius_graph(
    data: AtomicData,
    # generate_graph_fn: callable,
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    molecular_graph_cfg: MolecularGraphConfigs,
) -> GraphAttentionData:
    # atomic numbers
    atomic_numbers = data.atomic_numbers.long()

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
    ).to(data.pos.device)

    # generate graph
    (
        disp,  # (num_nodes, num_neighbors, 3)
        src_env,
        _,
        _,
        _,
        neighbor_index,  # (2, num_nodes, num_neighbors)
    ) = biknn_radius_graph(  # type: ignore
        data,
        molecular_graph_cfg.max_radius,  # type: ignore[arg-type]
        molecular_graph_cfg.knn_k,
        molecular_graph_cfg.knn_soft,
        molecular_graph_cfg.knn_sigmoid_scale,
        molecular_graph_cfg.knn_lse_scale,
        molecular_graph_cfg.knn_use_low_mem,
        molecular_graph_cfg.knn_pad_size,
        molecular_graph_cfg.use_pbc,
        data.pos.device,
    )

    num_nodes, num_neighbors, _ = disp.shape
    edge_direction = safe_normalize(disp)  # (num_nodes, num_neighbors, 3)
    edge_distance = safe_norm(disp)  # (num_nodes, num_neighbors)
    neighbor_mask = src_env != torch.inf  # (num_nodes, num_neighbors)
    src_mask = envelope_fn(src_env, molecular_graph_cfg.use_envelope)

    # edge distance expansion (ref: scn)
    # (num_nodes, num_neighbors, edge_distance_expansion_size)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance.flatten())
    edge_distance_expansion = edge_distance_expansion.view(
        num_nodes, num_neighbors, gnn_cfg.edge_distance_expansion_size
    )

    # node direction expansion (num_nodes, num_neighbors, lmax + 1)
    node_direction_expansion = get_node_direction_expansion_neighbor(
        direction_vec=edge_direction,
        neighbor_mask=neighbor_mask,
        lmax=gnn_cfg.node_direction_expansion_size - 1,
    )

    neighbor_list = neighbor_index[1]

    # pad batch
    if global_cfg.use_padding:
        (
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
        ) = pad_batch(
            max_atoms=molecular_graph_cfg.max_atoms,
            max_batch_size=molecular_graph_cfg.max_batch_size,
            atomic_numbers=atomic_numbers,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            src_mask=src_mask,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
        )
    else:
        node_padding_mask = torch.ones_like(atomic_numbers, dtype=torch.bool)
        graph_padding_mask = torch.ones(
            data.num_graphs, dtype=torch.bool, device=data.batch.device
        )
        node_batch = data.batch

    # patch singleton atom (TODO: check if this is needed)
    # if global_cfg.use_padding:
    #     edge_direction, neighbor_list, neighbor_mask = patch_singleton_atom(
    #         edge_direction, neighbor_list, neighbor_mask
    #     )

    # get attention mask
    attn_mask = get_attn_mask_env(src_mask, gnn_cfg.atten_num_heads)  # type: ignore
    if gnn_cfg.use_angle_embedding == "none":
        angle_embedding = None
    else:
        attn_mask, angle_embedding = get_attn_mask(
            edge_direction=edge_direction,
            neighbor_mask=neighbor_mask,
            num_heads=gnn_cfg.atten_num_heads,
            lmax=gnn_cfg.node_direction_expansion_size,
            use_angle_embedding=gnn_cfg.use_angle_embedding,
        )

    # get frequency vectors
    if gnn_cfg.use_frequency_embedding:
        # Create repeating dimensions tensor for assertion checks
        repeating_dimensions = torch.tensor(
            gnn_cfg.freequency_list, dtype=torch.long, device=data.pos.device
        )

        # Add assertions to validate repeating_dimensions
        # Check that values sum to head_dim
        head_dim = global_cfg.hidden_size // gnn_cfg.atten_num_heads
        sum_repeats = torch.sum(repeating_dimensions).item()
        assert sum_repeats == head_dim, (
            f"Sum of freequency_list must equal head_dim ({head_dim}), "
            f"but got sum={sum_repeats}. Please adjust freequency_list."
        )

        # Use the Python list directly for better torch.compile compatibility
        frequency_vectors = get_compact_frequency_vectors(
            edge_direction=edge_direction,
            lmax=len(gnn_cfg.freequency_list) - 1,
            repeating_dimensions=gnn_cfg.freequency_list,  # Pass list instead of tensor
        )
    else:
        frequency_vectors = None

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
        # enable math attention for fallbacks
        torch.backends.cuda.enable_math_sdp(True)
    else:
        raise NotImplementedError(
            f"Attention name {gnn_cfg.atten_name} not implemented"
        )

    # construct input data
    x = GraphAttentionData(
        atomic_numbers=atomic_numbers,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction=edge_direction,
        attn_mask=attn_mask,
        angle_embedding=angle_embedding,
        frequency_vectors=frequency_vectors,
        neighbor_list=neighbor_list,
        neighbor_mask=neighbor_mask,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        graph_padding_mask=graph_padding_mask,
    )
    return x
