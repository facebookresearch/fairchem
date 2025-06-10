from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from e3nn import o3

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface, HeadInterface
from fairchem.core.models.escaip.configs import EScAIPConfigs, init_configs
from fairchem.core.models.escaip.modules.graph_attention_block import (
    EfficientGraphAttentionBlock,
)
from fairchem.core.models.escaip.modules.input_block import InputBlock
from fairchem.core.models.escaip.modules.output_block import (
    OutputLayer,
    OutputProjection,
)
from fairchem.core.models.escaip.modules.readout_block import ReadoutBlock
from fairchem.core.models.escaip.utils.data_preprocess import (
    data_preprocess_radius_graph,
)
from fairchem.core.models.escaip.utils.graph_utils import (
    compilable_scatter,
    get_displacement_and_cell,
    unpad_results,
)
from fairchem.core.models.escaip.utils.nn_utils import (
    init_linear_weights,
    no_weight_decay,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.escaip.custom_types import GraphAttentionData


@registry.register_model("EScAIP_backbone")
class EScAIPBackbone(nn.Module, BackboneInterface):
    """
    Efficiently Scaled Attention Interactomic Potential (EScAIP) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.direct_forces = cfg.global_cfg.direct_forces
        self.regress_stress = cfg.global_cfg.regress_stress
        self.use_pbc = cfg.molecular_graph_cfg.use_pbc

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug
        # generate_graph_fn = partial(
        #     self.generate_graph,
        #     cutoff=self.molecular_graph_cfg.max_radius,
        #     max_neighbors=self.molecular_graph_cfg.max_neighbors,
        #     use_pbc=self.molecular_graph_cfg.use_pbc,
        #     otf_graph=self.molecular_graph_cfg.otf_graph,
        #     enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
        #     use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        # )

        # data preprocess
        self.data_preprocess = partial(
            # data_preprocess,
            data_preprocess_radius_graph,
            # generate_graph_fn=generate_graph_fn,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                    is_last=(idx == self.gnn_cfg.num_layers - 1),
                )
                for idx in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        # self.apply(init_linear_weights)
        self.init_weights()

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def compiled_forward(self, data: GraphAttentionData):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](data, node_features, edge_features)
        global_readouts = [readouts[0]]
        node_readouts = [readouts[1]]
        edge_readouts = [readouts[2]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                data, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](data, node_features, edge_features)
            readouts = self.readout_layers[idx + 1](data, node_features, edge_features)
            global_readouts.append(readouts[0])
            node_readouts.append(readouts[1])
            edge_readouts.append(readouts[2])

        global_features, node_features, edge_features = self.output_projection(
            data=data,
            global_readouts=torch.cat(global_readouts, dim=-1),
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
        )

        return {
            "data": data,
            "global_features": global_features.to(torch.float32)
            if global_features is not None
            else None,
            "node_features": node_features.to(torch.float32),
            "edge_features": edge_features.to(torch.float32)
            if edge_features is not None
            else None,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: AtomicData):
        data["atomic_numbers"] = data["atomic_numbers"].long()
        data["atomic_numbers_full"] = data["atomic_numbers"]
        data["batch_full"] = data["batch"]

        # gradient force and stress
        displacement, orig_cell = get_displacement_and_cell(
            data, self.regress_stress, self.regress_forces, self.direct_forces
        )

        # preprocess data
        x = self.data_preprocess(data)

        results = self.forward_fn(x)
        results["displacement"] = displacement
        results["orig_cell"] = orig_cell
        return results

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)

    def init_weights(self):
        names = [
            "ffn",
            "feedforward",
            "edge_hidden",
            "node_hidden",
            "edge_attr",
            "embedding",
            "message",
        ]
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if isinstance(module, torch.nn.Linear):
                    # check if the module is in the list of names
                    if any(n in name for n in names):
                        nn.init.xavier_uniform_(
                            module.weight, gain=nn.init.calculate_gain("relu")
                        )
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    module.bias.data.zero_()


class EScAIPHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces

    def post_init(self, gain=1.41421):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_direct_force_head")
class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )
        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        return force_direction * force_magnitude  # (num_nodes, 3)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_energy_head")
class EScAIPEnergyHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.energy_reduce = self.gnn_cfg.energy_reduce
        self.use_global_readout = self.gnn_cfg.use_global_readout

        self.post_init(gain=0.1)

    def compiled_forward(self, emb):
        if self.use_global_readout:
            return self.energy_layer(emb["global_features"])

        energy_output = self.energy_layer(emb["node_features"])

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].graph_padding_mask.shape[0],
            dim=0,
            reduce=self.energy_reduce,
        )
        return energy_output.squeeze()

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(emb)
        return unpad_results(
            results={"energy": energy_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_grad_energy_force_stress_head")
class EScAIPGradientEnergyForceStressHead(EScAIPEnergyHead):
    """
    Do not support torch.compile
    """

    def __init__(
        self,
        backbone: EScAIPBackbone,
        prefix: str | None = None,
        wrap_property: bool = True,
    ):
        super().__init__(backbone)
        self.regress_stress = self.global_cfg.regress_stress
        self.regress_forces = self.global_cfg.regress_forces
        self.prefix = prefix
        self.wrap_property = wrap_property

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
        if self.use_global_readout:
            energy_output = self.energy_layer(emb["global_features"])
        else:
            energy_output = self.energy_layer(emb["node_features"])

            # the following not compatible with torch.compile (grpah break)
            # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

            energy_output = compilable_scatter(
                src=energy_output,
                index=emb["data"].node_batch,
                dim_size=emb["data"].graph_padding_mask.shape[0],
                dim=0,
                reduce=self.energy_reduce,
            ).squeeze()
        outputs[energy_key] = (
            {"energy": energy_output} if self.wrap_property else energy_output
        )

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy_output.sum()],
                [data["pos_original"], emb["displacement"]],
                create_graph=self.training,
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
                    energy_output.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces

        return unpad_results(
            results=outputs,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_rank2_head")
class EScAIPRank2Head(EScAIPHeadBase):
    """
    Rank-2 head for EScAIP model. Modified from the Rank2Block for Equiformer V2.
    """

    def __init__(
        self,
        backbone: EScAIPBackbone,
        output_name: str = "stress",
    ):
        super().__init__(backbone)
        self.output_name = output_name
        self.scalar_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.irreps2_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, node_features, edge_features, data: GraphAttentionData):
        sphere_irrep2 = o3.spherical_harmonics(
            2, data.edge_direction, True
        ).detach()  # (num_nodes, max_neighbor, 5)

        # map from invariant to irrep2
        edge_irrep2 = (
            sphere_irrep2[:, :, :, None] * edge_features[:, :, None, :]
        )  # (num_nodes, max_neighbor, 5, h)

        # sum over neighbors
        neighbor_count = data.neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        neighbor_count = neighbor_count.to(edge_irrep2.dtype)
        node_irrep2 = (
            edge_irrep2 * data.neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1) / neighbor_count.unsqueeze(-1)  # (num_nodes, 5, h)

        irrep2_output = self.irreps2_layer(node_irrep2)  # (num_nodes, 5, 1)
        scalar_output = self.scalar_layer(node_features)  # (num_nodes, 1)

        # get graph level output
        irrep2_output = compilable_scatter(
            src=irrep2_output.view(-1, 5),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        scalar_output = compilable_scatter(
            src=scalar_output.view(-1, 1),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        return irrep2_output, scalar_output.view(-1)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        irrep2_output, scalar_output = self.forward_fn(
            node_features=emb["node_features"],
            edge_features=emb["edge_features"],
            data=emb["data"],
        )
        output = {
            f"{self.output_name}_isotropic": scalar_output.unsqueeze(1),
            f"{self.output_name}_anisotropic": irrep2_output,
        }

        return unpad_results(
            results=output,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )
