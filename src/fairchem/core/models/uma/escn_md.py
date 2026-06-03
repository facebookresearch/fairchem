"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.distributed.nn.functional import all_reduce as all_reduce_with_grad
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.parallelism.graph_parallel_a2a import (
    GPContext,
    build_gp_context,
)
from fairchem.core.common.parallelism.graph_partition import (
    PartitionStrategy,
    partition_atoms_index_split,
    partition_atoms_spatial,
)
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.common.quaternion.quaternion_wigner_utils import (
    create_wigner_data_module,
)
from fairchem.core.models.uma.common.quaternion.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
)
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.embedding import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.execution_backends import (
    get_execution_backend,
)
from fairchem.core.models.uma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole_utils import MOLEInterface
from fairchem.core.models.uma.nn.radial import GaussianSmearing, PolynomialEnvelope
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    compute_hessian,
    get_l_component_range,
    reduce_node_to_system,
)
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum
from fairchem.core.units.mlip_unit.api.inference import (
    validate_uma_atoms_data,
)
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task

from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE = 1024 * 128
AUTO_EDGE_CHUNK_FRACTION = 0.05


@dataclass
class GradRegressConfig:
    """
    Configuration for gradient-based computation of forces and stress.
    """

    direct_forces: bool = False
    direct_stress: bool = False
    forces: bool = False
    stress: bool = False
    hessian: bool = False
    hessian_vmap: bool = True


def add_n_empty_edges(graph_dict: dict, edges_to_add: int, cutoff: float):
    graph_dict["edge_index"] = torch.cat(
        (
            graph_dict["edge_index"].new_zeros(2, edges_to_add),
            graph_dict["edge_index"],
        ),
        dim=1,
    )

    self_edge_distance_vec = graph_dict["edge_distance_vec"].new_ones(1, 3) + cutoff
    graph_dict["edge_distance_vec"] = torch.cat(
        (
            self_edge_distance_vec.expand(edges_to_add, 3),
            graph_dict["edge_distance_vec"],
        ),
        dim=0,
    )

    edge_distance = torch.linalg.norm(self_edge_distance_vec, dim=-1, keepdim=False)
    graph_dict["edge_distance"] = torch.cat(
        (edge_distance.expand(edges_to_add), graph_dict["edge_distance"]), dim=0
    )


def validate_contiguous_channels(channels: list[int], name: str) -> tuple[int, int]:
    """Validate channels are contiguous, return (start, end) slice indices.

    Args:
        channels: List of channel indices to validate
        name: Name of the channel list for error messages

    Returns:
        Tuple of (start_idx, end_idx) for slicing. Returns (0, 0) if channels is empty.

    Raises:
        ValueError: If channels are not contiguous
    """
    if not channels:
        return 0, 0
    sorted_channels = sorted(channels)
    expected = list(range(sorted_channels[0], sorted_channels[-1] + 1))
    if sorted_channels != expected:
        raise ValueError(f"{name} must be contiguous (e.g., [0, 1, 2]). Got {channels}")
    return sorted_channels[0], sorted_channels[-1] + 1


def balance_channels_batched(
    emb: torch.Tensor,
    target: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    target_offset: float = 0.0,
) -> torch.Tensor:
    """Balance a contiguous range of channels to target sum per system.

    This batched version processes all channels in a contiguous range in a single
    call, which is more efficient than processing each channel individually.

    Args:
        emb: Node embeddings of shape [num_atoms, sph_features, channels]
        target: Target sum per system of shape [num_systems]
        natoms: Number of atoms per system of shape [num_systems]
        batch: Batch indices mapping atoms to systems of shape [num_atoms]
        start_idx: Start index of channel range (inclusive)
        end_idx: End index of channel range (exclusive)
        target_offset: Offset to subtract from target (e.g., 1.0 for spin)

    Returns:
        Modified embeddings with the specified channel range balanced to sum to target.

    Supports graph parallel (GP) mode using torch.distributed.nn.functional.all_reduce
    which provides correct gradients in both forward and backward passes.
    """
    out_emb = emb.clone()
    num_systems = len(natoms)
    n_channels = end_idx - start_idx

    # Batched extraction: [num_atoms, n_channels]
    channels_to_balance = emb[:, 0, start_idx:end_idx]

    # Batched sum: [num_systems, n_channels]
    system_sums = torch.zeros(
        num_systems, n_channels, device=emb.device, dtype=emb.dtype
    )
    system_sums.index_add_(0, batch, channels_to_balance)

    # Reduce partial sums across all graph parallel ranks
    if gp_utils.initialized():
        system_sums = all_reduce_with_grad(system_sums, group=gp_utils.get_gp_group())

    # Batched correction: broadcast target to all channels
    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    corrections = (system_sums - target_sums) / natoms.unsqueeze(1)

    out_emb[:, 0, start_idx:end_idx] = channels_to_balance - corrections[batch]
    return out_emb


def resolve_dataset_mapping(
    deprecated_list: list[str] | None,
    dataset_mapping: dict[str, str] | None,
    deprecated_param_name: str = "dataset_list",
) -> dict[str, str]:
    """
    Validate and resolve dataset mapping from either a deprecated list or a mapping dict.

    Args:
        deprecated_list: Deprecated list of dataset names. If provided, it is
            converted to a mapping where each name maps to itself.
        dataset_mapping: Mapping from the config dataset name to desired dataset name for embeddings and heads.
            Allows multiple subsets to share the same dataset embedding and/or output head by mapping
            them to the same identifier.
        deprecated_param_name: Name of the deprecated parameter, used in
            warning/error messages.

    Returns:
        The resolved dataset mapping dict.

    Raises:
        ValueError: If both or neither arguments are provided, if the mapping
            is not a non-empty dict, or if mapping values are not a subset of
            mapping keys.
    """
    if deprecated_list is not None and dataset_mapping is not None:
        msg = (
            f"Both '{deprecated_param_name}' (={deprecated_list}) and "
            f"'dataset_mapping' (={dataset_mapping}) have been provided. "
            f"Please provide 'dataset_mapping' only in the config as '{deprecated_param_name}' is deprecated."
        )
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is None and dataset_mapping is None:
        msg = "'dataset_mapping' must be provided in the config to use dataset embeddings."
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is not None:
        if not isinstance(deprecated_list, (list, ListConfig)):
            msg = f"If '{deprecated_param_name}' is provided in the config, it must be a list of dataset names. Got: {deprecated_list!r}"
            logging.error(msg, stack_info=True)
            raise ValueError(msg)
        dataset_mapping = {name: name for name in deprecated_list}
        logging.warning(
            f"If '{deprecated_param_name}' is provided in the config, the code assumes that each dataset "
            f"maps to itself. Please use 'dataset_mapping' as '{deprecated_param_name}' "
            "is deprecated and will be removed in the future."
        )
    if not isinstance(dataset_mapping, (dict, DictConfig)) or not dataset_mapping:
        msg = f"'dataset_mapping' must be a non-empty dictionary, got: {dataset_mapping!r}"
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if not set(dataset_mapping.values()) <= set(dataset_mapping.keys()):
        missing = set(dataset_mapping.values()) - set(dataset_mapping.keys())
        msg = (
            f"dataset_mapping values {missing} are not present in "
            f"dataset_mapping keys {set(dataset_mapping.keys())}. "
            f"Values must be a subset of keys. Full mapping provided: {dataset_mapping}"
        )
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    return dataset_mapping


@registry.register_model("escnmd_backbone")
class eSCNMDBackbone(nn.Module, MOLEInterface):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,  # NOTE not used
        # NOTE: graph construction related, to remove
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,  # deprecated
        use_pbc_single: bool = True,  # deprecated
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: Literal["gaussian"] = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        direct_stress: bool = False,
        regress_stress: bool = False,
        regress_hessian: bool = False,
        hessian_vmap: bool = True,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: (
            list[str] | None
        ) = None,  # deprecated, use dataset_mapping instead
        dataset_mapping: (
            dict[str, str] | None
        ) = None,  # mapping from config dataset name to dataset embedding name e.g. {"omol": "omol", "oc20": "oc20", "oc20_subset": "oc20"}, this allows multiple subsets to use the same dataset embedding.
        use_dataset_embedding: bool = True,
        use_cuda_graph_wigner: bool = False,
        use_quaternion_wigner: bool = True,
        radius_pbc_version: int = 2,
        always_use_pbc: bool = True,
        charge_balanced_channels: list[int] | None = None,
        spin_balanced_channels: list[int] | None = None,
        edge_chunk_size: int = 1,
        execution_mode: str = "general",
        use_all_to_all_gp: bool = False,
        gp_partition_strategy: str = "index_split",
    ) -> None:
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples
        # set this True if we want to ALWAYS use pbc for internal graph gen
        # despite what's in the input data this only affects when otf_graph is True
        # in this mode, the user must be responsible for providing a large vaccum box
        # for aperiodic systems
        self.always_use_pbc = always_use_pbc

        # energy conservation related
        self.regress_config = GradRegressConfig(
            direct_forces=direct_forces,
            forces=regress_forces,
            stress=regress_stress,
            direct_stress=direct_stress,
            hessian=regress_hessian,
            hessian_vmap=hessian_vmap,
        )

        # which channels to balance - validate contiguity and store slice indices
        charge_channels = (
            list(charge_balanced_channels) if charge_balanced_channels else []
        )
        spin_channels = list(spin_balanced_channels) if spin_balanced_channels else []

        self.charge_channel_start, self.charge_channel_end = (
            validate_contiguous_channels(charge_channels, "charge_balanced_channels")
        )
        self.spin_channel_start, self.spin_channel_end = validate_contiguous_channels(
            spin_channels, "spin_balanced_channels"
        )

        # NOTE: graph construction related, to remove, except for cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.use_quaternion_wigner = use_quaternion_wigner
        self.enforce_max_neighbors_strictly = False

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            # The size of edge blocks to use in activation checkpointing
            activation_checkpoint_chunk_size = (
                ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE
            )
        self.edge_chunk_size = edge_chunk_size
        self.use_all_to_all_gp = use_all_to_all_gp
        self.gp_partition_strategy = PartitionStrategy(gp_partition_strategy)

        # Allgather+spatial is not supported because allgather concatenates
        # per-rank tensors in rank order, which only matches global atom order
        # for contiguous (index_split) partitions.  Spatial partitions are
        # non-contiguous, so edge_index lookups into the allgathered tensor
        # would reference wrong atoms.  Spatial partitioning is designed for
        # A2A where only boundary atoms are exchanged.
        if (
            not self.use_all_to_all_gp
            and self.gp_partition_strategy == PartitionStrategy.SPATIAL
        ):
            raise ValueError(
                "Spatial partitioning is not supported with all-gather GP. "
                "Allgather concatenates per-rank tensors in rank order, which "
                "only matches global atom order for contiguous (index_split) "
                "partitions. Use use_all_to_all_gp=True with spatial, or "
                "gp_partition_strategy='index_split' with allgather."
            )

        self.backend = get_execution_backend(execution_mode)

        # related to charge spin dataset system embedding
        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_mapping = dataset_mapping
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        if self.use_dataset_embedding:
            self.dataset_mapping = resolve_dataset_mapping(
                self.dataset_list, dataset_mapping, "dataset_list"
            )
        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])

        # Precompute Wigner coefficients for quaternion path (like Jd for Euler path)
        if self.use_quaternion_wigner:
            # lmin=5 because l=0,1,2,3,4 use custom kernels in the hybrid method
            self.wigner_data = create_wigner_data_module(lmax=self.lmax, lmin=5)

        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        # charge / spin embedding
        self.charge_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "charge",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.spin_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "spin",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )

        # dataset embedding
        if self.use_dataset_embedding:
            self.dataset_embedding = DatasetEmbedding(
                self.sphere_channels,
                enable_grad=self.dataset_emb_grad,
                dataset_mapping=self.dataset_mapping,
            )
            # mix charge, spin, dataset embeddings
            self.mix_csd = nn.Linear(3 * self.sphere_channels, self.sphere_channels)
        else:
            # mix charge, spin
            self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,  # NOTE: sqrt avg degree
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=self.backend,
        )

        self.envelope = PolynomialEnvelope(exponent=5)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSCNMD_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.ff_type,
                activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
                backend=self.backend,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    def balance_channels(
        self,
        x_message_prime: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        natoms: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.charge_channel_end > self.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms=natoms,
                batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end,
                target_offset=0.0,
            )
        if self.spin_channel_end > self.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms=natoms,
                batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end,
                target_offset=1.0,
            )
        return x_message_prime

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_quaternion_wigner:
            with record_function("obtain rotmat wigner quaternion"):
                wigner, wigner_inv = axis_angle_wigner_hybrid(
                    edge_distance_vecs,
                    self.lmax,
                    coeffs=self.wigner_data.coeffs,
                    U_blocks=self.wigner_data.U_blocks,
                    custom_kernels=self.wigner_data.custom_kernels,
                )
        else:
            Jd_buffers = [
                getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
                for l in range(self.lmax + 1)
            ]

            with record_function("obtain rotmat wigner original"):
                euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
                wigner = eulers_to_wigner(
                    euler_angles,
                    0,
                    self.lmax,
                    Jd_buffers,
                )
                wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # Both axis_angle_wigner_hybrid and eulers_to_wigner return contiguous D
        # (created via torch.zeros + slice assignment)
        # wigner_inv is made contiguous by .transpose().contiguous() above
        return wigner, wigner_inv

    def csd_embedding(self, charge, spin, dataset):
        with record_function("charge spin dataset embeddings"):
            # Add charge, spin, and dataset embeddings
            chg_emb = self.charge_embedding(charge)
            spin_emb = self.spin_embedding(spin)
            if self.use_dataset_embedding:
                assert dataset is not None
                dataset_emb = self.dataset_embedding(dataset)
                return torch.nn.SiLU()(
                    self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
                )
            return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    @torch.compiler.disable
    def _compute_a2a_partition(
        self,
        pos: torch.Tensor,
        total_atoms: int,
        device: torch.device,
        world_size: int,
        rank: int,
        strategy: PartitionStrategy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute A2A rank assignments and node partition.

        Separated from _generate_graph so that only the A2A-specific
        partitioning is excluded from torch.compile.  The BL (all-gather)
        path stays fully compilable.
        """
        with record_function("a2a_partition"):
            if strategy == PartitionStrategy.SPATIAL:
                rank_assignments = partition_atoms_spatial(pos, world_size)
            else:
                rank_assignments = partition_atoms_index_split(
                    total_atoms, world_size, device
                )
        node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]

        return rank_assignments, node_partition

    @torch.compiler.disable
    def _compute_aabb_halo(
        self,
        pos: torch.Tensor,
        node_partition: torch.Tensor,
        pbc: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AABB halo mask for graph generation filtering.

        Builds an axis-aligned bounding box (AABB) around the local
        partition's atoms, expanded by the interaction cutoff, then
        identifies ALL atoms (including PBC images) that fall within
        this box. Graph generation then operates on ~N_halo atoms
        instead of N_total, significantly reducing cost at scale.

        No NCCL communication — purely local computation.

        Note: Only supports single-system inputs (not batched).
        For multi-system batches, the caller should skip halo
        filtering and fall back to full graph generation.

        Returns:
            Tuple of (halo_mask, shift_vecs) where:
            - halo_mask: Boolean mask over all atoms in the AABB halo.
            - shift_vecs: PBC shift vectors (n_shifts, 3).
        """
        device = pos.device
        n_total = len(pos)

        # --- Compute our AABB (expanded by cutoff) ---
        local_pos = pos[node_partition]
        lo = local_pos.min(dim=0)[0] - self.cutoff
        hi = local_pos.max(dim=0)[0] + self.cutoff

        cell_sq = cell.view(3, 3) if cell.dim() == 3 else cell
        pbc_flat = pbc.view(3) if pbc.dim() == 2 else pbc

        # Build shift vectors for periodic images (up to 27).
        shift_components = []
        for d in range(3):
            if pbc_flat[d]:
                shift_components.append(torch.tensor([-1, 0, 1], device=device))
            else:
                shift_components.append(torch.tensor([0], device=device))
        grid = torch.cartesian_prod(*shift_components)
        shift_vecs = grid.float() @ cell_sq

        # --- Compute our halo mask (atoms we need) ---
        halo_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        for shift in shift_vecs:
            shifted = pos + shift
            in_box = ((shifted >= lo) & (shifted <= hi)).all(dim=-1)
            halo_mask |= in_box

        return halo_mask, shift_vecs

    @torch.compiler.disable
    def _compute_halo_graph(
        self,
        data_dict: dict,
        node_partition: torch.Tensor,
        rank_assignments: torch.Tensor,
        pbc: torch.Tensor,
        halo_mask: torch.Tensor,
    ) -> dict | None:
        """
        Try to generate graph using AABB halo filtering.

        Uses the pre-computed halo_mask from _compute_aabb_halo to
        filter graph gen input from N_total to ~N_halo atoms.
        Returns the graph_dict if the halo achieves significant
        reduction, or None to fall back to full graph generation.

        The edge_index in the returned graph_dict is remapped to
        global coordinates for use by the backbone forward pass.
        Additionally, halo-local metadata is stored so that
        ``build_gp_context`` can work with O(N_halo)-sized tensors
        instead of O(N_total).

        Args:
            data_dict: Full data dictionary with pos, cell, etc.
            node_partition: Local atom indices (global coords).
            rank_assignments: Rank assignment per atom (global).
            pbc: Periodic boundary conditions.
            halo_mask: Pre-computed boolean halo mask.

        Returns:
            graph_dict with edge_index in global coordinates and
            halo metadata for build_gp_context,
            or None if halo didn't help.
        """
        with record_function("a2a_halo_filter"):
            pos = data_dict["pos"]
            cell = data_dict["cell"]
            n_total = len(pos)

            n_halo = halo_mask.sum().item()

            # Only use halo filtering if it reduces atoms enough.
            if n_halo >= n_total * 0.95:
                return None

            halo_indices = halo_mask.nonzero(as_tuple=True)[0]

            # Map global indices to halo-local indices
            global_to_halo = torch.full(
                (n_total,), -1, dtype=torch.long, device=pos.device
            )
            global_to_halo[halo_indices] = torch.arange(n_halo, device=pos.device)

            # Create subset data for graph generation.
            data_subset = types.SimpleNamespace()
            data_subset.pos = pos[halo_indices]
            data_subset.cell = cell
            data_subset.natoms = torch.tensor([n_halo], device=pos.device)
            data_subset.batch = torch.zeros(n_halo, dtype=torch.long, device=pos.device)
            data_subset.pbc = pbc

            # Remap partition and rank_assignments to halo-local
            node_partition_local = global_to_halo[node_partition]
            assert (node_partition_local >= 0).all(), (
                "Local partition atoms not found in halo — "
                "AABB expansion may be too small"
            )
            rank_assignments_local = rank_assignments[halo_indices]

            graph_dict = generate_graph(
                data_subset,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=(self.enforce_max_neighbors_strictly),
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
                node_partition=node_partition_local,
                rank_assignments=rank_assignments_local,
                rank=gp_utils.get_gp_rank(),
                world_size=gp_utils.get_gp_world_size(),
            )

            # Remap edge_index from halo-local to global for use
            # by the backbone forward pass (source/target embedding
            # lookups index into atomic_numbers_full).
            graph_dict["edge_index"] = halo_indices[graph_dict["edge_index"]]

            # Remap send_info indices from halo-local to global so
            # build_gp_context can use them with global-coord
            # rank_assignments and edge_index.
            if "send_info" in graph_dict:
                si = graph_dict["send_info"]
                if si["send_indices_global"].numel() > 0:
                    si["send_indices_global"] = halo_indices[si["send_indices_global"]]

            return graph_dict

    def _generate_graph(self, data_dict):
        node_partition = None
        rank_assignments = None
        if gp_utils.initialized():
            # create the partitions
            atomic_numbers_full = data_dict["atomic_numbers_full"]

            if self.use_all_to_all_gp:
                # All-to-all: compute rank_assignments FIRST, then derive
                # node_partition from them.  This ensures the
                # graph-generation partition and the GPContext partition
                # are identical, avoiding index mismatches that cause
                # OOB crashes.
                natoms = len(atomic_numbers_full)
                rank_assignments, node_partition = self._compute_a2a_partition(
                    pos=data_dict["pos"],
                    total_atoms=natoms,
                    device=atomic_numbers_full.device,
                    world_size=gp_utils.get_gp_world_size(),
                    rank=gp_utils.get_gp_rank(),
                    strategy=self.gp_partition_strategy,
                )
            else:
                # All-gather: only supports contiguous (index_split)
                # partitioning. __init__ blocks spatial+allgather.
                node_partition = torch.tensor_split(
                    torch.arange(
                        len(atomic_numbers_full),
                        device=atomic_numbers_full.device,
                    ),
                    gp_utils.get_gp_world_size(),
                )[gp_utils.get_gp_rank()]

            assert (
                node_partition.numel() > 0
            ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"

        if self.otf_graph:
            pbc = None
            if self.always_use_pbc:
                pbc = torch.ones(len(data_dict), 3, dtype=torch.bool)
            else:
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            assert (
                pbc.all() or (~pbc).all()
            ), "We can only accept pbc that is all true or all false"

            # AABB halo optimization for A2A:
            # Compute AABB bounding box around local partition,
            # expanded by cutoff, to filter graph gen input from
            # N_total to ~N_halo atoms. Pure local computation,
            # no NCCL needed.
            # Note: AABB halo only supports single-system inputs.
            # Multi-system batches skip halo and use full graph gen.
            graph_dict = None
            is_single_system = data_dict["cell"].dim() == 2 or (
                data_dict["cell"].dim() == 3 and data_dict["cell"].shape[0] == 1
            )
            if (
                self.use_all_to_all_gp
                and rank_assignments is not None
                and is_single_system
            ):
                halo_mask, _ = self._compute_aabb_halo(
                    data_dict["pos"],
                    node_partition,
                    pbc,
                    data_dict["cell"],
                )
                graph_dict = self._compute_halo_graph(
                    data_dict,
                    node_partition,
                    rank_assignments,
                    pbc,
                    halo_mask,
                )

            if graph_dict is None:
                # Full graph gen (no halo filter, or halo didn't help)
                graph_dict = generate_graph(
                    data_dict,
                    cutoff=self.cutoff,
                    max_neighbors=self.max_neighbors,
                    enforce_max_neighbors_strictly=(
                        self.enforce_max_neighbors_strictly
                    ),
                    radius_pbc_version=self.radius_pbc_version,
                    pbc=pbc,
                    node_partition=node_partition,
                    rank_assignments=(
                        rank_assignments if self.use_all_to_all_gp else None
                    ),
                    rank=(gp_utils.get_gp_rank() if self.use_all_to_all_gp else None),
                    world_size=(
                        gp_utils.get_gp_world_size() if self.use_all_to_all_gp else None
                    ),
                )
        else:
            # this assume edge_index is provided
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"

            # Compute shifts from cell offsets
            if len(data_dict["natoms"]) == 1:
                # Single system: use matmul (compile-friendly, no data-dependent ops)
                shifts = data_dict["cell_offsets"].to(
                    data_dict["cell"].dtype
                ) @ data_dict["cell"].squeeze(0)
            else:
                # Batched: need repeat_interleave for variable edges per system
                cell_per_edge = data_dict["cell"].repeat_interleave(
                    data_dict["nedges"], dim=0
                )
                shifts = torch.einsum(
                    "ij,ijk->ik",
                    data_dict["cell_offsets"].to(cell_per_edge.dtype),
                    cell_per_edge,
                )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )  # [n_edges, 3]
            # pylint: disable=E1102
            edge_distance = torch.linalg.norm(
                edge_distance_vec, dim=-1, keepdim=False
            )  # [n_edges, 1]

            graph_dict = {
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
            }

        if gp_utils.initialized():
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                node_partition
            ]
            data_dict["batch"] = data_dict["batch_full"][node_partition]

            # Build GPContext for all-to-all communication
            if self.use_all_to_all_gp:
                with record_function("a2a_build_gp_context"):
                    gp_ctx = build_gp_context(
                        edge_index=graph_dict["edge_index"],
                        rank_assignments=rank_assignments,
                        rank=gp_utils.get_gp_rank(),
                        world_size=gp_utils.get_gp_world_size(),
                        send_info=graph_dict.get("send_info"),
                        node_partition=node_partition,
                    )
                data_dict["gp_ctx"] = gp_ctx
                # Store rank_assignments so output heads can reorder
                # gathered forces/stress from partition-concatenated order
                # back to global index order. Only needed for A2A where
                # partitions are non-consecutive (spatial).
                data_dict["gp_rank_assignments"] = rank_assignments
                # A2A scatter_target comes from gp_ctx.edge_index_local[1]
                # which is set dynamically in InteractionBlock.forward().
                data_dict["scatter_target"] = None
            else:
                # Allgather: pre-compute local target indices for scatter
                # operations. Maps global edge targets to 0-based local
                # partition indices. Works for both contiguous (index_split)
                # and non-contiguous (spatial) partitions.
                total_atoms = len(data_dict["atomic_numbers_full"])
                device = graph_dict["edge_index"].device
                global_to_local = torch.zeros(
                    total_atoms, dtype=torch.long, device=device
                )
                global_to_local[node_partition] = torch.arange(
                    len(node_partition), device=device
                )
                data_dict["scatter_target"] = global_to_local[
                    graph_dict["edge_index"][1]
                ]

        if graph_dict["edge_index"].shape[1] == 0:
            add_n_empty_edges(graph_dict, 1, self.cutoff)
            # Also update scatter_target for the dummy edge (target=0)
            if (
                "scatter_target" in data_dict
                and data_dict["scatter_target"] is not None
            ):
                data_dict["scatter_target"] = torch.cat(
                    [
                        data_dict["scatter_target"].new_zeros(1),
                        data_dict["scatter_target"],
                    ]
                )

        return graph_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict: AtomicData) -> dict[str, torch.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )

        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )

        # Enable gradients for autograd-based force/stress computation.
        # Must be set before graph generation so the computation graph
        # tracks positions and cell through edge distance calculations.
        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        with record_function("obtain wigner"):
            wigner, wigner_inv = self._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"],
            )
            coefficient_index = (
                self.coefficient_index if self.mmax != self.lmax else None
            )
            wigner, wigner_inv = self.backend.prepare_wigner(
                wigner,
                wigner_inv,
                self.mappingReduced,
                coefficient_index,
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        ###
        # Hook to allow MOLE
        ###
        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()

        # Retrieve precomputed all-to-all context (needed for edge embedding
        # and message passing layers)
        gp_ctx: GPContext | None = data_dict.get("gp_ctx", None)
        send_indices: torch.Tensor | None = None
        if gp_ctx is not None:
            send_indices = gp_ctx.send_indices

        # edge degree embedding
        with record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding),
                dim=1,
            )

            # Pre-fuse envelope into wigner_inv
            wigner_inv_envelope = wigner_inv * edge_envelope

            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                # scatter_target contains pre-computed local target indices
                # for scattering edge messages to nodes. For A2A, use
                # gp_ctx.edge_index_local[1] (already local). For allgather,
                # use pre-computed global→local mapped targets.
                gp_ctx.edge_index_local[1]
                if gp_ctx is not None
                else data_dict.get(
                    "scatter_target", default=graph_dict["edge_index"][1]
                ),
                wigner_inv_envelope,
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        # Get edge embeddings for each layer
        # General backend: raw x_edge (rad_func computed inside SO2_Convolution)
        # Fast backends: precomputed radials
        with record_function("layer_radial_emb"):
            x_edge_per_layer = self.backend.get_layer_radial_emb(x_edge, self)

        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge_per_layer[i],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    scatter_target=data_dict.get("scatter_target", default=None),
                    gp_ctx=gp_ctx,
                    send_indices=send_indices,
                )
                # balance any channels requested
                x_message = self.balance_channels(
                    x_message,
                    charge=data_dict["charge"],
                    spin=data_dict["spin"],
                    natoms=data_dict["natoms"],
                    batch=data_dict["batch"],
                )

        # Final layer norm
        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
        }
        return out

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    @classmethod
    def build_inference_settings(cls, settings: InferenceSettings) -> dict:
        """Build backbone config overrides from inference settings."""
        overrides = {}

        # Always disable PBC wrapping for inference
        overrides["always_use_pbc"] = False

        if settings.activation_checkpointing is not None:
            overrides["activation_checkpointing"] = settings.activation_checkpointing
        if settings.edge_chunk_size is not None:
            overrides["edge_chunk_size"] = settings.edge_chunk_size
        if settings.external_graph_gen is not None:
            overrides["otf_graph"] = not settings.external_graph_gen
        if settings.internal_graph_gen_version is not None:
            overrides["radius_pbc_version"] = settings.internal_graph_gen_version
        if settings.use_quaternion_wigner is not None:
            overrides["use_quaternion_wigner"] = settings.use_quaternion_wigner
        if settings.execution_mode is not None:
            overrides["execution_mode"] = settings.execution_mode

        return overrides

    def get_default_untrained_tasks(
        self,
        checkpoint_tasks: dict[str, Task],
        inference_settings: InferenceSettings,
    ) -> list[Task]:
        """
        Return default untrained tasks for eSCNMDBackbone.

        For this backbone, we add stress tasks for all energy datasets
        that don't already have stress (either trained or explicitly requested).
        Stress can be computed via autograd from energy predictions.

        Returns empty list if the model uses direct forces, since autograd-based
        stress computation requires energy-conserving force computation.
        """
        # Direct force models can't compute stress via autograd
        if self.regress_config.direct_forces:
            return []

        tasks = []

        # Find datasets with energy but no stress
        energy_datasets = set()
        stress_datasets = set()
        energy_task_by_dataset = {}

        for task in checkpoint_tasks.values():
            if task.property == "energy":
                for dataset in task.datasets:
                    energy_datasets.add(dataset)
                    energy_task_by_dataset[dataset] = task
            elif task.property == "stress":
                stress_datasets.update(task.datasets)

        # Also exclude datasets already in predict_untrained_stress
        stress_datasets.update(inference_settings.predict_untrained_stress)

        # Create stress tasks for missing datasets
        missing_stress_datasets = energy_datasets - stress_datasets

        for dataset in missing_stress_datasets:
            energy_task = energy_task_by_dataset[dataset]
            # Infer task name prefix from energy task naming convention
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            tasks.append(
                Task(
                    name=f"{task_prefix}stress",
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(
                        dim=[1, 9], dtype=inference_settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )

        return tasks

    def validate_tasks(self, dataset_to_tasks: dict[str, list]) -> None:
        """
        Validate that task datasets are compatible with this backbone.
        """
        if self.use_dataset_embedding:
            assert set(dataset_to_tasks.keys()).issubset(
                set(self.dataset_mapping.keys())
            ), "Datasets in tasks is not a strict subset of datasets in backbone."

    def prepare_for_inference(self, data: AtomicData, settings: InferenceSettings):
        """
        Prepare model for inference. Called once on first prediction.
        """
        self._inference_settings = settings
        self.backend.validate(self.lmax, self.mmax, settings)
        self.backend.prepare_model_for_inference(self)
        return self

    def on_predict_check(self, data: AtomicData) -> None:
        """
        Called before each prediction.
        """

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        UMA-specific validation: handle charge/spin for OMOL task.

        Uses the shared validation logic from the api.inference module.
        """
        validate_uma_atoms_data(atoms, task_name)


class MLP_EFS_Head(nn.Module, HeadInterface):
    """MLP head for predicting energy, forces, and stress using autograd derivatives.

    This head computes forces and stress by taking gradients of the energy with respect to
    atomic positions and cell displacement.
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: str | None = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()

        self.reduce = reduce
        self.prefix = prefix
        self.wrap_property = wrap_property

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        backbone.energy_block = None
        backbone.force_block = None
        self.regress_config = backbone.regress_config

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        energy_key = f"{self.prefix}_energy" if self.prefix else "energy"
        forces_key = f"{self.prefix}_forces" if self.prefix else "forces"
        stress_key = f"{self.prefix}_stress" if self.prefix else "stress"
        hessian_key = f"{self.prefix}_hessian" if self.prefix else "hessian"

        outputs = {}

        # Use shared energy computation from parent class
        energy, energy_part = compute_energy(
            emb,
            self.energy_block,
            data["batch"],
            len(data["natoms"]),
            natoms=data["natoms"],
            reduce=self.reduce,
        )

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = emb["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )

        # Determine if we need create_graph for higher-order derivatives
        # Hessian computation requires second derivatives, so we need create_graph=True
        create_graph = self.training or self.regress_config.hessian

        if self.regress_config.stress and not self.regress_config.direct_stress:
            forces, stress = compute_forces_and_stress(
                energy_part,
                data["pos"],
                data["cell"],
                batch=data["batch_full"],  # use batch_full to work with GP reduction
                training=create_graph,
            )
            # TODO should we assume gradient forces always when stress is requested?
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
        elif self.regress_config.forces and not self.regress_config.direct_forces:
            forces = compute_forces(energy_part, data["pos"], training=self.training)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        else:
            forces = None

        if self.regress_config.hessian:
            if forces is None:
                raise ValueError(
                    "Hessian computation requires forces. "
                    "Please enable regress_forces or regress_stress."
                )
            if data["natoms"].numel() != 1:
                raise ValueError(
                    f"Hessian computation requires exactly 1 system in batch, "
                    f"found {data['natoms'].numel()}"
                )

            hessian = compute_hessian(
                forces,
                data["pos"],
                vmap=self.regress_config.hessian_vmap,
                training=create_graph,
            )
            outputs[hessian_key] = (
                {"hessian": hessian} if self.wrap_property else hessian
            )

        return outputs


# Deprecate this head in favor of MLP_EFS_Head with a regress_config.forces=False and regress_config.stress=False.
class MLP_Energy_Head(MLP_EFS_Head):
    """MLP head for predicting energy."""

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: str | None = None,
        wrap_property: bool = False,
    ) -> None:
        super().__init__(backbone, reduce, prefix, wrap_property)
        assert (
            backbone.regress_config.forces is False
            and backbone.regress_config.stress is False
        ) or (
            backbone.regress_config.direct_forces is True
            or backbone.regress_config.direct_stress is True
        ), (
            "regress_forces and regress_stress must be False or direct_forces must be True to use an MLP_Energy_Head. "
            "Use an MLP_EFS_Head if you want to predict gradient forces and stress."
        )


class Linear_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        energy, _ = compute_energy(
            emb,
            self.energy_block,
            data_dict["batch"],
            len(data_dict["natoms"]),
            natoms=data_dict["natoms"],
            reduce=self.reduce,
        )
        return {"energy": energy}


class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone) -> None:
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]):
        # SO3_Linear with lmax=1 requires both L=0 and L=1 as input
        l0_l1_embedding = get_l_component_range(emb["node_embedding"], l_min=0, l_max=1)
        forces_output = self.linear(l0_l1_embedding)

        # Extract L=1 (vector) component from the output
        forces = get_l_component_range(forces_output, l_min=1, l_max=1)
        forces = forces.view(-1, 3).contiguous()

        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(
                forces, data_dict["atomic_numbers_full"].shape[0]
            )
            # A2A spatial partitions are non-consecutive, so the
            # gathered forces are in partition-concatenated order
            # (NOT global index order). Reorder to match positions.
            ra = data_dict.get("gp_rank_assignments", None)
            if ra is not None:
                ws = gp_utils.get_gp_world_size()
                perm = torch.cat(
                    [(ra == r).nonzero(as_tuple=True)[0] for r in range(ws)]
                )
                forces_ordered = torch.empty_like(forces)
                forces_ordered[perm] = forces
                forces = forces_ordered

        return {"forces": forces}


def compose_tensor(
    trace: torch.Tensor,
    l2_symmetric: torch.Tensor,
) -> torch.Tensor:
    """Re-compose a tensor from its decomposition

    Args:
        trace: a tensor with scalar part of the decomposition of r2 tensors in the batch
        l2_symmetric: tensor with the symmetric/traceless part of decomposition

    Returns:
        tensor: rank 2 tensor
    """

    if trace.shape[1] != 1:
        raise ValueError("batch of traces must be shape (batch size, 1)")

    if l2_symmetric.shape[1] != 5:
        raise ValueError("batch of l2_symmetric tensors must be shape (batch size, 5)")

    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError(
            "Shape missmatch between trace and l2_symmetric parts. The first dimension is the batch dimension"
        )

    batch_size = trace.shape[0]
    decomposed_preds = torch.zeros(
        batch_size, irreps_sum(2), device=trace.device
    )  # rank 2
    decomposed_preds[:, : irreps_sum(0)] = trace
    decomposed_preds[:, irreps_sum(1) : irreps_sum(2)] = l2_symmetric

    r2_tensor = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(2, device=trace.device),
        decomposed_preds,
    )
    return r2_tensor


class MLP_Stress_Head(nn.Module, HeadInterface):
    """MLP head for predicting the stress tensor.

    Predicts the isotropic (L=0) and anisotropic (L=2) parts of the stress tensor
    separately to ensure symmetry, then recomposes back to the full stress tensor.
    """

    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "mean") -> None:
        super().__init__()
        self.reduce = reduce
        assert reduce in ["sum", "mean"]
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.scalar_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self.l2_linear = SO3_Linear(backbone.sphere_channels, 1, lmax=2)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        num_systems = len(data_dict["natoms"])
        batch = data_dict["batch"]

        # Compute isotropic (L=0) part of stress using MLP on scalar embedding
        scalar_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=0
        ).squeeze(1)
        node_scalar = self.scalar_block(scalar_embedding).view(-1)
        iso_stress, _ = reduce_node_to_system(node_scalar, batch, num_systems)

        if self.reduce == "mean":
            iso_stress = iso_stress / data_dict["natoms"]

        # Compute anisotropic (L=2) part of stress using SO3_Linear
        l0l1l2_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=2
        )
        l2_output = self.l2_linear(l0l1l2_embedding)

        node_l2 = (
            get_l_component_range(l2_output, l_min=2, l_max=2).view(-1, 5).contiguous()
        )
        aniso_stress, _ = reduce_node_to_system(node_l2, batch, num_systems)

        if self.reduce == "mean":
            aniso_stress = aniso_stress / data_dict["natoms"].unsqueeze(1)

        # Recompose the full stress tensor from isotropic and anisotropic parts
        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)

        return {"stress": stress}
