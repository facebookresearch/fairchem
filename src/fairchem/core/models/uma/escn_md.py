"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.distributed.nn.functional import all_reduce as all_reduce_with_grad
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
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
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    UMATask,
)

from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE = 1024 * 128


def add_n_empty_edges(
    graph_dict: dict, edges_to_add: int, cutoff: float, node_offset: int = 0
):
    graph_dict["edge_index"] = torch.cat(
        (
            graph_dict["edge_index"].new_ones(2, edges_to_add) * node_offset,
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


def get_balanced_attribute(
    emb: torch.Tensor,
    target_sum: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    balance_attribute_offset: float = 0,
    balance_channel_idx: int = 0,
) -> torch.Tensor:
    """Balance per-atom attributes (charge/spin) to sum to system target.

    Args:
        emb: Node embeddings of shape [num_atoms, sph_features, channels]
        target_sum: Target sum per system of shape [num_systems]
        natoms: Number of atoms per system of shape [num_systems]
        batch: Batch indices mapping atoms to systems of shape [num_atoms]
        balance_attribute_offset: Offset to subtract from target (e.g., 1 for spin)
        balance_channel_idx: Which channel index to balance

    Returns:
        Modified embeddings with the specified channel balanced to sum to target.

    Supports graph parallel (GP) mode using torch.distributed.nn.functional.all_reduce
    which provides correct gradients in both forward and backward passes.
    """
    out_emb = emb.clone()

    charge_unbalanced = emb[:, 0, balance_channel_idx]

    system_scalars_part = torch.zeros(
        len(natoms),
        device=emb.device,
        dtype=emb.dtype,
    )

    system_scalars_part.index_add_(0, batch, charge_unbalanced.view(-1))

    # Reduce partial sums across all graph parallel ranks
    # Use all_reduce_with_grad which has all_reduce in both forward AND backward,
    # ensuring correct gradient computation when atoms are split across ranks.
    if gp_utils.initialized():
        system_scalar = all_reduce_with_grad(
            system_scalars_part, group=gp_utils.get_gp_group()
        )
    else:
        system_scalar = system_scalars_part

    correction = (system_scalar - (target_sum - balance_attribute_offset)) / natoms

    balanced_node_scalar = charge_unbalanced - correction[batch]

    out_emb[:, 0, balance_channel_idx] = (
        out_emb[:, 0, balance_channel_idx] * 0 + balanced_node_scalar
    )

    return out_emb


@torch.compiler.disable
def pad_edges(graph_dict, edge_chunk_size: int, cutoff: float, node_offset: int = 0):
    n_edges = n_edges_post = graph_dict["edge_index"].shape[1]

    if edge_chunk_size > 0 and n_edges_post % edge_chunk_size != 0:
        # make sure we have a multiple of self.edge_chunk_size edges
        n_edges_post += edge_chunk_size - n_edges_post % edge_chunk_size

    n_edges_post = max(n_edges_post, 1)  # at least 1 edge to avoid empty "edge" case
    if n_edges_post > n_edges:
        # We append synthetic padding edges whose distance vector has norm > cutoff
        # (see add_n_empty_edges where distance_vec is set to 1+cutoff). The radial
        # polynomial envelope returns 0 for distances >= cutoff, so these edges never
        # contribute to embeddings or message passing; they only ensure the edge count
        # is a multiple of edge_chunk_size (or at least one edge), aiding chunked
        # activation checkpointing and avoiding empty tensor edge cases.
        add_n_empty_edges(graph_dict, n_edges_post - n_edges, cutoff, node_offset)


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
        regress_stress: bool = False,
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
        radius_pbc_version: int = 2,
        always_use_pbc: bool = True,
        charge_balanced_channels: list[int] | None = None,
        spin_balanced_channels: list[int] | None = None,
        edge_chunk_size: int | None = None,
        execution_mode: str = "general",
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
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress

        # which channels to balance
        self.charge_balanced_channels = (
            charge_balanced_channels if charge_balanced_channels is not None else []
        )
        self.spin_balanced_channels = (
            spin_balanced_channels if spin_balanced_channels is not None else []
        )

        # NOTE: graph construction related, to remove, except for cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.enforce_max_neighbors_strictly = False

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            # The size of edge blocks to use in activation checkpointing
            activation_checkpoint_chunk_size = (
                ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE
            )
        self.edge_chunk_size = edge_chunk_size

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
        for channel_idx in self.charge_balanced_channels:
            x_message_prime = get_balanced_attribute(
                emb=x_message_prime,
                target_sum=charge,
                natoms=natoms,
                batch=batch,
                balance_channel_idx=channel_idx,
            )
        for channel_idx in self.spin_balanced_channels:
            x_message_prime = get_balanced_attribute(
                emb=x_message_prime,
                target_sum=spin,
                natoms=natoms,
                batch=batch,
                balance_attribute_offset=1,
                balance_channel_idx=channel_idx,
            )
        return x_message_prime

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
        )
        return wigner_and_M_mapping, wigner_and_M_mapping_inv

    def _get_displacement_and_cell(
        self, data_dict: AtomicData
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ###############################################################
        # gradient-based forces/stress
        ###############################################################
        displacement = None
        orig_cell = None
        if self.regress_stress and not self.direct_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=data_dict["pos"].dtype,
                device=data_dict["pos"].device,
            )
            num_batch = len(data_dict["natoms"])
            displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            if data_dict["pos"].requires_grad is False:
                data_dict["pos"].requires_grad = True
            data_dict["pos_original"] = data_dict["pos"]
            data_dict["pos"] = data_dict["pos"] + torch.bmm(
                data_dict["pos"].unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data_dict["batch"]),
            ).squeeze(-2)

            orig_cell = data_dict["cell"]
            data_dict["cell"] = data_dict["cell"] + torch.bmm(
                data_dict["cell"], symmetric_displacement
            )

        if (
            not self.regress_stress
            and self.regress_forces
            and not self.direct_forces
            and data_dict["pos"].requires_grad is False
        ):
            data_dict["pos"].requires_grad = True
        return displacement, orig_cell

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

    def _generate_graph(self, data_dict):
        data_dict["gp_node_offset"] = 0
        node_partition = None
        if gp_utils.initialized():
            # create the partitions
            atomic_numbers_full = data_dict["atomic_numbers_full"]
            node_partition = torch.tensor_split(
                torch.arange(
                    len(atomic_numbers_full), device=atomic_numbers_full.device
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
            # for v2 graph gen we used to pass node_partition as part of the data_dict directly to radius_pbc to allow it generate partial graphs
            # to make it more general to accomodate v3, we scrapped and instead have generate_graph handle the partitioning after the graph has been generated
            graph_dict = generate_graph(
                data_dict,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
                node_partition=node_partition,
            )
        else:
            # this assume edge_index is provided
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"
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
            data_dict["gp_node_offset"] = node_partition.min().item()

        if self.edge_chunk_size is not None:
            pad_edges(
                graph_dict,
                self.edge_chunk_size,
                self.cutoff,
                data_dict["gp_node_offset"],
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

        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = self._get_displacement_and_cell(data_dict)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        if graph_dict["edge_index"].numel() == 0:
            raise ValueError(
                f"No edges found in input system, this means either you have a single atom in the system or the atoms are farther apart than the radius cutoff of the model of {self.cutoff} Angstroms. We don't know how to handle this case. Check the positions of system: {data_dict['pos']}"
            )

        with record_function("obtain wigner"):
            (wigner_and_M_mapping, wigner_and_M_mapping_inv) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec"],
                )
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

            # Pre-fuse envelope into wigner_inv for edge degree embedding
            wigner_and_M_mapping_inv_envelope_for_edge_degree = (
                wigner_and_M_mapping_inv * edge_envelope
            )

            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv_envelope_for_edge_degree,
                data_dict["gp_node_offset"],
            )

        # Pre-fuse envelope into wigner_inv for block message passing
        wigner_and_M_mapping_inv_envelope = wigner_and_M_mapping_inv * edge_envelope

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner_and_M_mapping,
                    wigner_and_M_mapping_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    node_offset=data_dict["gp_node_offset"],
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
            "displacement": displacement,
            "orig_cell": orig_cell,
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
        if settings.execution_mode is not None:
            overrides["execution_mode"] = settings.execution_mode

        return overrides

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

        For UMA: handles MOLE merging if settings.merge_mole is True.
        Stores initial composition for consistency checking.

        Returns:
            self or a new merged backbone if MOLE merging was performed. We return
            because type could have changed due to merging MOLE.
        """
        self._inference_settings = settings
        self._merged_composition = None

        # Validate settings against backend requirements (fail early)
        self.backend.validate(settings)

        if settings.merge_mole:
            assert (
                data.natoms.numel() == 1
            ), "Cannot merge model with multiple systems in batch"
            # Store composition we merged on
            self._merged_composition = self._get_composition_info(data)
            # Merge the model - returns new merged backbone
            new_backbone = self.merge_MOLE_model(data)
            # Transfer inference state to new backbone
            new_backbone._inference_settings = settings
            new_backbone._merged_composition = self._merged_composition
            self.backend.prepare_model_for_inference(new_backbone)
            return new_backbone

        self.backend.prepare_model_for_inference(self)
        return self

    def on_predict_check(self, data: AtomicData) -> None:
        """
        Called before each prediction. UMA checks MOLE consistency here.
        """
        if not getattr(self, "_inference_settings", None):
            return  # Not initialized yet

        if self._inference_settings.merge_mole and self._merged_composition is not None:
            assert (
                data.natoms.numel() == 1
            ), "Cannot run merged model on batch with multiple systems"
            current = self._get_composition_info(data)
            self._assert_composition_matches(current)

    def _get_composition_info(self, data) -> tuple:
        """
        Get composition info for MOLE consistency checking.
        """
        composition = data.atomic_numbers.new_zeros(
            self.max_num_elements, dtype=torch.int
        ).index_add(
            0,
            data.atomic_numbers.to(torch.int),
            data.atomic_numbers.new_ones(len(data.atomic_numbers), dtype=torch.int),
        )
        return (
            composition,
            getattr(data, "charge", None),
            getattr(data, "spin", None),
            getattr(data, "dataset", [None]),
        )

    def _assert_composition_matches(self, current: tuple) -> None:
        """
        Assert current composition matches what model was merged on.
        """
        merged = self._merged_composition
        # Move current tensors to same device as merged (CPU) for comparison
        device = merged[0].device

        merged_norm = merged[0].float() / merged[0].sum()
        curr_norm = current[0].float().to(device) / current[0].sum().to(device)

        assert merged_norm.isclose(
            curr_norm, rtol=1e-5
        ).all(), "Compositions differ from merged model"

        # Charge and spin are tensors that need device alignment
        merged_charge = merged[1]
        curr_charge = (
            current[1].to(device)
            if isinstance(current[1], torch.Tensor)
            else current[1]
        )
        assert (
            (merged_charge == curr_charge).all()
            if isinstance(merged_charge, torch.Tensor)
            else merged_charge == curr_charge
        ), f"Charge differs: {merged_charge} vs {current[1]}"

        merged_spin = merged[2]
        curr_spin = (
            current[2].to(device)
            if isinstance(current[2], torch.Tensor)
            else current[2]
        )
        assert (
            (merged_spin == curr_spin).all()
            if isinstance(merged_spin, torch.Tensor)
            else merged_spin == curr_spin
        ), f"Spin differs: {merged_spin} vs {current[2]}"

        assert merged[3] == current[3], f"Dataset differs: {merged[3]} vs {current[3]}"

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        UMA-specific validation: handle charge/spin for OMOL task.

        Sets default values for charge and spin in atoms.info and validates
        they are within acceptable ranges.
        """
        # Set charge defaults
        if "charge" not in atoms.info:
            if task_name == UMATask.OMOL.value:
                logging.warning(
                    "task_name='omol' detected, but charge is not set in atoms.info. "
                    "Defaulting to charge=0. Ensure charge is an integer representing "
                    "the total charge on the system and is within the range -100 to 100."
                )
            atoms.info["charge"] = DEFAULT_CHARGE

        # Set spin defaults (OMOL uses spin=1, others use spin=0)
        if "spin" not in atoms.info:
            if task_name == UMATask.OMOL.value:
                atoms.info["spin"] = DEFAULT_SPIN_OMOL
                logging.warning(
                    "task_name='omol' detected, but spin multiplicity is not set in "
                    "atoms.info. Defaulting to spin=1. Ensure spin is an integer "
                    "representing the spin multiplicity from 0 to 100."
                )
            else:
                atoms.info["spin"] = DEFAULT_SPIN

        # Validate charge range
        charge = atoms.info["charge"]
        if not isinstance(charge, (int, np.integer)):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. "
                "Charge must be an integer representing the total charge on the system."
            )
        if not (CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]):
            raise ValueError(
                f"Invalid value for charge: {charge}. "
                f"Charge must be within the range {CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )

        # Validate spin range
        spin = atoms.info["spin"]
        if not isinstance(spin, (int, np.integer)):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. "
                "Spin must be an integer representing the spin multiplicity."
            )
        if not (SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]):
            raise ValueError(
                f"Invalid value for spin: {spin}. "
                f"Spin must be within the range {SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )


class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone: eSCNMDBackbone,
        prefix: str | None = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces
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

        # TODO: this is not very clean, bug-prone.
        # but is currently necessary for finetuning pretrained models that did not have
        # the direct_forces flag set to False
        backbone.direct_forces = False
        assert (
            not backbone.direct_forces
        ), "EFS head is only used for gradient-based forces/stress."

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}
        _input = emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)
        node_energy = _output.view(-1, 1, 1)
        energy_part = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy_part.index_add_(0, data["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = emb["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy_part.sum()],
                [data["pos_original"], emb["displacement"]],
                create_graph=self.training,
            )
            if gp_utils.initialized():
                grads = (
                    gp_utils.reduce_from_model_parallel_region(grads[0]),
                    gp_utils.reduce_from_model_parallel_region(grads[1]),
                )

            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            stress = stress.view(
                -1, 9
            )  # NOTE to work better with current Multi-task trainer
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy_part.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            if gp_utils.initialized():
                forces = gp_utils.reduce_from_model_parallel_region(forces)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        return outputs


class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


class Linear_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        energy_part = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy_part.index_add_(0, data_dict["batch"], node_energy.view(-1))

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone) -> None:
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]):
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(
                forces, data_dict["atomic_numbers_full"].shape[0]
            )

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
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "mean") -> None:
        super().__init__()
        """
        predict the isotropic and anisotropic parts of the stress tensor
        to ensure symmetry and then recompose back to the full stress tensor
        """
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
        node_scalar = self.scalar_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        ).view(-1, 1, 1)

        iso_stress = torch.zeros(
            len(data_dict["natoms"]),
            device=node_scalar.device,
            dtype=node_scalar.dtype,
        )
        iso_stress.index_add_(0, data_dict["batch"], node_scalar.view(-1))

        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")
            # iso_stress = gp_utils.reduce_from_model_parallel_region(iso_stress)

        if self.reduce == "mean":
            iso_stress /= data_dict["natoms"]

        node_l2 = self.l2_linear(emb["node_embedding"].narrow(1, 0, 9))
        node_l2 = node_l2.narrow(1, 4, 5)
        node_l2 = node_l2.view(-1, 5).contiguous()

        aniso_stress = torch.zeros(
            (len(data_dict["natoms"]), 5),
            device=node_l2.device,
            dtype=node_l2.dtype,
        )
        aniso_stress.index_add_(0, data_dict["batch"], node_l2)
        if gp_utils.initialized():
            raise NotImplementedError("This code hasn't been tested yet.")
            # aniso_stress = gp_utils.reduce_from_model_parallel_region(aniso_stress)

        if self.reduce == "mean":
            aniso_stress /= data_dict["natoms"].unsqueeze(1)

        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)

        return {"stress": stress}
