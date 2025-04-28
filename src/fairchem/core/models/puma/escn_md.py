"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import cg_change_mat, conditional_grad, irreps_sum
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.puma.common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)
from fairchem.core.models.puma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.puma.nn.embedding_dev import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.puma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.puma.nn.radial import GaussianSmearing
from fairchem.core.models.puma.nn.so3_layers import SO3_Linear

from .escn_md_block import eSCNMD_Block


@registry.register_model("escnmd_backbone")
class eSCNMDBackbone(nn.Module):
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
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
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
        chg_spin_emb_type: str = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: list[str] | None = None,
        use_dataset_embedding: bool = True,
    ):
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples

        # energy conservation related
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress

        # NOTE: graph construction related, to remove, except for cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.enforce_max_neighbors_strictly = False
        self.activation_checkpointing = activation_checkpointing
        # related to charge spin dataset system embedding
        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        assert (
            self.dataset_list
        ), "the dataset list is empty, please add it to the model backbone config"

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
                grad=self.dataset_emb_grad,
                dataset_list=self.dataset_list,
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
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,  # NOTE: sqrt avg degree
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
        )

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

    def prepare_MOE(self, data, graph, csd_mixed_emb):
        pass

    def get_rotmat_and_wigner(self, edge_distance_vecs):
        with record_function("obtain rotmat"):
            edge_rot_mat = init_edge_rot_mat(
                edge_distance_vecs, rot_clip=(not self.direct_forces)
            )

        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_rot_mat.dtype)
            for l in range(self.lmax + 1)
        ]

        with record_function("obtain wigner"):
            wigner = rotation_to_wigner(
                edge_rot_mat,
                0,
                self.lmax,
                Jd_buffers,
                rot_clip=(not self.direct_forces),
            )
            wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m, wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m
        )

        return edge_rot_mat, wigner_and_M_mapping, wigner_and_M_mapping_inv

    def generate_graph(self, *args, **kwargs):
        graph = generate_graph(*args, **kwargs)
        return dict(  # noqa: C408
            edge_index=graph.edge_index,
            edge_distance=graph.edge_distance,
            edge_distance_vec=graph.edge_distance_vec,
            cell_offsets=graph.cell_offsets,
            offset_distances=None,
            neighbors=None,
            node_offset=0,
            batch_full=graph.batch_full,
            atomic_numbers_full=graph.atomic_numbers_full,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict) -> dict[str, torch.Tensor]:
        ###############################################################
        # gradient-based forces/stress
        ###############################################################
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        # TODO: this part is actually better put in the hydra base forward as
        # it is a common requirement for any energy-conserving model.
        # currently we move graph generation to the hydra part, out side backbone forward to
        # make compilation work. This will break energy-conserving models.
        # but eventually we might move the require_grad part to the hydra forward.
        # so this problem can be solved.
        displacement = None
        orig_cell = None
        if self.regress_stress and not self.direct_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=data_dict["pos"].dtype,
                device=data_dict["pos"].device,
            )
            # num_batch = data_dict["num_graphs"]
            num_batch = data_dict.get("num_graphs", len(data_dict["natoms"]))
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

        with record_function("generate_graph"):
            if self.otf_graph:
                graph_dict = self.generate_graph(
                    data_dict,
                    cutoff=self.cutoff,
                    max_neighbors=self.max_neighbors,
                    use_pbc=self.use_pbc,
                    otf_graph=self.otf_graph,
                    enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                    use_pbc_single=self.use_pbc_single,
                )
            else:
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
                    "atomic_numbers_full": data_dict["atomic_numbers_full"],
                    "batch_full": data_dict["batch_full"],
                    "edge_index": data_dict["edge_index"],
                    "edge_distance": edge_distance,
                    "edge_distance_vec": edge_distance_vec,
                    "node_offset": 0,
                }

        if gp_utils.initialized():
            graph_dict, data_dict = self._init_gp_partitions(graph_dict, data_dict)
        else:
            graph_dict["edge_distance_vec_full"] = graph_dict["edge_distance_vec"]
            graph_dict["edge_distance_full"] = graph_dict["edge_distance"]
            graph_dict["edge_index_full"] = graph_dict["edge_index"]

        with record_function("obtain wigner"):
            (_, wigner_and_M_mapping_full, wigner_and_M_mapping_inv_full) = (
                self.get_rotmat_and_wigner(graph_dict["edge_distance_vec_full"])
            )
        if gp_utils.initialized():
            wigner_and_M_mapping = wigner_and_M_mapping_full[
                graph_dict["edge_partition"]
            ]
            wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full[
                graph_dict["edge_partition"]
            ]
        else:
            wigner_and_M_mapping = wigner_and_M_mapping_full
            wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full

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

        sys_node_embedding = None
        csd_mixed_emb = None
        with record_function("charge spin dataset embeddings"):
            # Add charge, spin, and dataset embeddings
            chg_emb = self.charge_embedding(data_dict["charge"])
            spin_emb = self.spin_embedding(data_dict["spin"])
            if self.use_dataset_embedding:
                dataset_emb = self.dataset_embedding(data_dict["dataset"])
                csd_mixed_emb = torch.nn.SiLU()(
                    self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
                )
            else:
                csd_mixed_emb = torch.nn.SiLU()(
                    self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1))
                )
            x_message[:, 0, :] = x_message[:, 0, :] + csd_mixed_emb[data_dict["batch"]]
            sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        # full_sys = csd_mixed_emb[data_dict["batch_full"]]

        ###
        # Hook to allow MOE
        ###
        self.prepare_MOE(data_dict, graph_dict, csd_mixed_emb)

        # edge degree embedding
        with record_function("edge embedding"):
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
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv,
                graph_dict["node_offset"],
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner_and_M_mapping,
                        wigner_and_M_mapping_inv,
                        sys_node_embedding,
                        graph_dict["node_offset"],
                        use_reentrant=(
                            not self.training if self.direct_forces else False
                        ),
                    )
                else:
                    x_message = self.blocks[i](
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner_and_M_mapping,
                        wigner_and_M_mapping_inv,
                        sys_node_embedding=sys_node_embedding,
                        node_offset=graph_dict["node_offset"],
                    )

        # Final layer norm
        x_message = self.norm(x_message)

        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
            "batch": data_dict["batch"],
        }
        out.update(graph_dict)
        return out

    def _init_gp_partitions(self, graph_dict, data_dict):
        """Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        atomic_numbers_full = graph_dict["atomic_numbers_full"]
        data_batch_full = graph_dict["batch_full"]
        edge_index = graph_dict["edge_index"]
        edge_distance = graph_dict["edge_distance"]
        edge_distance_vec_full = graph_dict["edge_distance_vec"]

        node_partition = torch.tensor_split(
            torch.arange(len(atomic_numbers_full)).to(atomic_numbers_full.device),
            gp_utils.get_gp_world_size(),
        )[gp_utils.get_gp_rank()]

        assert (
            node_partition.numel() > 0
        ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]

        # full versions of data
        graph_dict["edge_distance_vec_full"] = edge_distance_vec_full
        graph_dict["edge_distance_full"] = edge_distance
        graph_dict["edge_index_full"] = edge_index
        graph_dict["edge_partition"] = edge_partition

        # gp versions of data
        graph_dict["atomic_numbers"] = atomic_numbers_full[node_partition]
        graph_dict["batch"] = data_batch_full[node_partition]
        graph_dict["edge_index"] = edge_index[:, edge_partition]
        graph_dict["edge_distance"] = edge_distance[edge_partition]
        graph_dict["edge_distance_vec"] = edge_distance_vec_full[edge_partition]
        graph_dict["node_offset"] = node_partition.min().item()

        data_dict["atomic_numbers_full"] = atomic_numbers_full
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"][node_partition]
        data_dict["batch"] = data_dict["batch"][node_partition]
        data_dict["batch_full"] = data_batch_full
        return graph_dict, data_dict

    @property
    def num_params(self):
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


class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, prefix=None, wrap_property=True):
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
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
    def __init__(self, backbone, reduce: str = "sum"):
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

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
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
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
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
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
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
    def __init__(self, backbone, reduce: str = "mean"):
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

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
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
