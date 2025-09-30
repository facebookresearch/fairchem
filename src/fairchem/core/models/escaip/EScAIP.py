from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

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
    get_normalization_layer,
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
        self.dataset_list = cfg.global_cfg.dataset_list
        self.max_num_elements = cfg.molecular_graph_cfg.max_num_elements
        self.max_neighbors = cfg.molecular_graph_cfg.knn_k
        self.cutoff = cfg.molecular_graph_cfg.max_radius

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess_radius_graph,
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
                    is_last=(idx == self.global_cfg.num_layers - 1),
                )
                for idx in range(self.global_cfg.num_layers)
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
                for _ in range(self.global_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        self.init_weights()

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)  # type: ignore

    def compiled_forward(self, data: GraphAttentionData):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](data, node_features, edge_features)
        global_readouts = [readouts[0]]
        node_readouts = [readouts[1]]
        edge_readouts = [readouts[2]]

        # transformer blocks
        for idx in range(self.global_cfg.num_layers):
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
        # TODO: remove this when FairChem fixes the bug
        data["atomic_numbers"] = data["atomic_numbers"].long()  # type: ignore
        data["atomic_numbers_full"] = data["atomic_numbers"]  # type: ignore
        data["batch_full"] = data["batch"]  # type: ignore

        # gradient force and stress
        displacement, orig_cell = get_displacement_and_cell(
            data, self.regress_stress, self.regress_forces, self.direct_forces
        )

        # preprocess data
        x = self.data_preprocess(data)

        # compile forward function
        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        results = self.forward_fn(x)
        results["displacement"] = displacement
        results["orig_cell"] = orig_cell
        return results

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return no_weight_decay(self)

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


class EScAIPHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_direct_force_head")
class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
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
        self.node_norm = get_normalization_layer(self.reg_cfg.normalization)(
            self.global_cfg.hidden_size
        )
        self.edge_norm = get_normalization_layer(self.reg_cfg.normalization)(
            self.global_cfg.hidden_size
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        edge_features = self.edge_norm(edge_features)
        node_features = self.node_norm(node_features)

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
        self.forward_fn = (
            torch.compile(self.compiled_forward)  # type: ignore
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        force_output = self.forward_fn(  # type: ignore
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],  # type: ignore
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,  # type: ignore
            graph_padding_mask=emb["data"].graph_padding_mask,  # type: ignore
        )


@registry.register_model("EScAIP_energy_head")
class EScAIPEnergyHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
        super().__init__(backbone)
        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.energy_reduce = self.gnn_cfg.energy_reduce
        self.use_global_readout = self.gnn_cfg.use_global_readout
        self.node_norm = get_normalization_layer(self.reg_cfg.normalization)(
            self.global_cfg.hidden_size
        )

        self.post_init()

    def compiled_forward(self, emb):
        if self.use_global_readout:
            return self.energy_layer(self.node_norm(emb["global_features"]))

        energy_output = self.energy_layer(self.node_norm(emb["node_features"]))

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
        self.forward_fn = (
            torch.compile(self.compiled_forward)  # type: ignore
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        energy_output = self.forward_fn(emb)  # type: ignore
        if len(energy_output.shape) == 0:
            energy_output = energy_output.unsqueeze(0)
        return unpad_results(
            results={"energy": energy_output},
            node_padding_mask=emb["data"].node_padding_mask,  # type: ignore
            graph_padding_mask=emb["data"].graph_padding_mask,  # type: ignore
        )


@registry.register_model("EScAIP_grad_energy_force_stress_head")
class EScAIPGradientEnergyForceStressHead(EScAIPEnergyHead):  # type: ignore
    """
    Do not support torch.compile
    """

    def __init__(
        self,
        backbone: EScAIPBackbone,  # type: ignore
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
                index=emb["data"].node_batch,  # type: ignore
                dim_size=emb["data"].graph_padding_mask.shape[0],  # type: ignore
                dim=0,
                reduce=self.energy_reduce,
            ).squeeze()
            if len(energy_output.shape) == 0:
                energy_output = energy_output.unsqueeze(0)
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
            node_padding_mask=emb["data"].node_padding_mask,  # type: ignore
            graph_padding_mask=emb["data"].graph_padding_mask,  # type: ignore
        )
