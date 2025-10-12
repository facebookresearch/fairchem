from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function
import torch.nn.functional as F
from torch_scatter import scatter_add

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface, HeadInterface

from .configs import EScAIPConfigs, init_configs
from .modules.graph_attention_block import (
    GraphAttentionBlock,
)
from .modules.input_block import InputBlock
from .utils.data_preprocess import (
    data_preprocess_radius_graph,
)
from .utils.graph_utils import (
    compilable_scatter,
    get_displacement_and_cell,
    unpad_result,
    charge_renormalization,
    coulomb_energy_from_src_index,
    heisenberg_energy_from_src_index
)
from .utils.nn_utils import (
    NormalizationType,
    get_feedforward,
    #get_linear,
    get_normalization_layer,
    init_linear_weights,
    no_weight_decay,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData

    from .custom_types import GraphAttentionData


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
                GraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for idx in range(self.global_cfg.num_layers)
            ]
        )

        # init weights
        self.init_weights()

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)  # type: ignore

    def compiled_forward(self, data: GraphAttentionData):
        # input block
        with record_function("input_block"):
            neighbor_reps, global_reps = self.input_block(data)

        # transformer blocks
        for idx in range(self.global_cfg.num_layers):
            with record_function(f"transformer_block_{idx}"):
                neighbor_reps, global_reps = self.transformer_blocks[idx](
                    data, neighbor_reps, global_reps
                )

        return {
            "data": data,
            "node_reps": neighbor_reps[:, 0].to(torch.float32),
            "global_reps": global_reps.to(torch.float32),
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: AtomicData):
        # TODO: remove this when FairChem fixes this
        data["atomic_numbers"] = data["atomic_numbers"].long()  # type: ignore
        data["atomic_numbers_full"] = data["atomic_numbers"]  # type: ignore
        data["batch_full"] = data["batch"]  # type: ignore
        #print("batch size: ", data["batch_full"].shape)
        # gradient force and stress
        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = get_displacement_and_cell(
                data, self.regress_stress, self.regress_forces, self.direct_forces
            )

        # preprocess data
        with record_function("data_preprocess"), torch.autocast(
            device_type=str(data.pos.device), enabled=False
        ):
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
        results["src_index"] = x.src_index
        results["dst_index"] = x.dst_index
        # results have: 
        # 'data', 'node_reps', 'global_reps', 'displacement', 'orig_cell'
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
        self.regress_stress = backbone.regress_stress
        self.direct_forces = backbone.direct_forces

        self.use_global_path = backbone.global_cfg.use_global_path

        normalization = NormalizationType(self.reg_cfg.normalization)
        self.node_norm = get_normalization_layer(normalization)(
            self.global_cfg.hidden_size
        )

        if self.use_global_path:
            # self.global_scale = get_linear(
            #     in_features=self.global_cfg.hidden_size,
            #     out_features=self.global_cfg.hidden_size,
            #     bias=True,
            #     activation=None,
            #     dropout=self.reg_cfg.mlp_dropout,
            # )
            # self.global_bias = get_linear(
            #     in_features=self.global_cfg.hidden_size,
            #     out_features=self.global_cfg.hidden_size,
            #     bias=True,
            #     activation=None,
            #     dropout=self.reg_cfg.mlp_dropout,
            # )
            # self.global_norm = get_normalization_layer(normalization)(
            #     self.global_cfg.hidden_size
            # )
            self.global_scale = torch.nn.Parameter(
                torch.randn(1) * 0.001, requires_grad=True
            )

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

    def get_FiLM_node_reps(self, emb):
        node_reps = emb["node_reps"]
        if self.use_global_path:
            # global_reps = self.global_norm(emb["global_reps"][:, 0])
            # global_scale = 1 + self.global_scale(global_reps)[emb["data"].node_batch]
            # global_bias = self.global_bias(global_reps)[emb["data"].node_batch]
            # node_reps = self.node_norm(node_reps * global_scale + global_bias)
            global_reps = self.global_scale * emb["global_reps"][:, 0]
            node_reps = self.node_norm(node_reps + global_reps[emb["data"].node_batch])
        else:
            node_reps = self.node_norm(node_reps)
        return node_reps

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_direct_force_head")
class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
        super().__init__(backbone)
        self.force_ffn = get_feedforward(
            hidden_dim=self.global_cfg.hidden_size,
            hidden_layer_multiplier=self.gnn_cfg.output_hidden_layer_multiplier,
            output_dim=3,
            bias=True,
            activation=self.global_cfg.activation,
        )
        self.post_init()

    def compiled_forward(self, emb):
        node_reps = self.get_FiLM_node_reps(emb)
        force_direction = self.force_ffn(node_reps)
        return force_direction

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_fn = (
            torch.compile(self.compiled_forward)  # type: ignore
            if self.global_cfg.use_compile
            else self.compiled_forward
        )
        force_output = self.forward_fn(emb)  # type: ignore
        return unpad_result(
            results={"forces": force_output},
            data=emb["data"],
        )


@registry.register_model("EScAIP_energy_head")
class EScAIPEnergyHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
        super().__init__(backbone)
        self.energy_ffn = get_feedforward(
            hidden_dim=self.global_cfg.hidden_size,
            hidden_layer_multiplier=self.gnn_cfg.output_hidden_layer_multiplier,
            output_dim=1,
            bias=True,
            activation=self.global_cfg.activation,
        )
        self.energy_reduce = self.gnn_cfg.energy_reduce

        self.post_init()

    def compiled_forward(self, emb):
        node_reps = self.get_FiLM_node_reps(emb)
        energy_output = self.energy_ffn(node_reps)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].max_batch_size,
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
        
        print("shape: ", emb["data"].charge.shape)
        energy_output = self.forward_fn(emb)  # type: ignore
        if len(energy_output.shape) == 0:
            energy_output = energy_output.unsqueeze(0)
        
        return unpad_result(
            results={"energy": energy_output},
            data=emb["data"],
        )


@registry.register_model("EScAIP_energy_head_lr")
class EScAIPEnergyHeadLR(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):  # type: ignore
        super().__init__(backbone)
        self.energy_ffn = get_feedforward(
            hidden_dim=self.global_cfg.hidden_size,
            hidden_layer_multiplier=self.gnn_cfg.output_hidden_layer_multiplier,
            output_dim=1,
            bias=True,
            activation=self.global_cfg.activation,
        )
        
        self.latent_dim_out = 1 
        
        if self.gnn_cfg.heisenberg_tf: 
            self.latent_dim_out = 2

        self.charge_ffn = get_feedforward(
            hidden_dim=self.global_cfg.hidden_size_lr,
            input_dim=self.global_cfg.hidden_size,
            hidden_layer_multiplier=1,
            output_dim=self.latent_dim_out,
            bias=True,
            activation=None
        )

        if self.gnn_cfg.equil_charges_tf: 
            self.hardness_ffn = get_feedforward(
                hidden_dim=self.global_cfg.hidden_size_lr,
                input_dim=self.global_cfg.hidden_size,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=None
            )

            self.electronegativity_ffn = get_feedforward(
                hidden_dim=self.global_cfg.hidden_size_lr,
                input_dim=self.global_cfg.hidden_size,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=None
            )

        if self.gnn_cfg.heisenberg_tf:
            self.coupling_ffn = get_feedforward(
                input_dim=1,
                hidden_dim=self.global_cfg.hidden_size_lr,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=None
            )

        self.energy_reduce = self.gnn_cfg.energy_reduce

        self.post_init()

    def compiled_forward_reps(self, emb: dict[str, torch.Tensor]):
        node_reps = self.get_FiLM_node_reps(emb)
        return node_reps

    def compiled_forward(self, emb: dict[str, torch.Tensor], node_reps):
        energy_output = self.energy_ffn(node_reps)
        # the following not compatible with torch.compile (graph break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        
        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].max_batch_size,
            dim=0,
            reduce=self.energy_reduce,
        )
        return energy_output.squeeze()  # Return the entire dictionary instead of just energy_output.squeeze()
    
    def compiled_forward_charges(self, emb: dict[str, torch.Tensor], node_reps):
        
        charges_raw = self.charge_ffn(node_reps)  # (N, latent_dim_out)
        
        if self.latent_dim_out == 2:
            #print("charges shape (2 comp): ", charges_raw.shape)
            charges_two_comp = charges_raw.abs() * self.gnn_cfg.charge_scale
            charges_sum = charges_two_comp.sum(dim=1, keepdim=True)
            charges_raw = charges_sum
            alpha = charges_two_comp[:, 0].view(-1, 1, 1)
            beta = charges_two_comp[:, 1].view(-1, 1, 1)
            spin = (alpha - beta).view(-1, 1, 1)
        
        # Extract commonly used values once
        num_nodes = emb["data"].num_nodes
        num_graphs = emb["data"].num_graphs
        node_batch = emb["data"].node_batch

            
        if self.gnn_cfg.charge_scale != 1.0:
            charges_raw = charges_raw * self.gnn_cfg.charge_scale
        
        if self.gnn_cfg.constrain_charge:
            flattened_charges_raw = charges_raw.squeeze(-1)
            
            flattened_charges_raw = charge_renormalization(
                flattened_charges_raw, emb, eps=1e-8
            )

            # testing 
            """
            valid_node_batch = node_batch[:num_nodes]
            valid_charges = flattened_charges_raw[:num_nodes]
            valid_node_batch = node_batch[:num_nodes]
            target_charges = emb["data"].charge[:num_graphs]
            global_charges = compilable_scatter(
                valid_charges, 
                index=valid_node_batch,
                dim_size=emb["data"].num_graphs,
                dim=0,
                reduce="sum"
            )
            print("Global charges after renorm: ", global_charges)
            float_target_charges = target_charges.float()

            assert torch.allclose(global_charges, float_target_charges, atol=1e-4), f"Global charges {global_charges} do not match target charges {target_charges}"
            """
            # testing 
            
            e_charge_single = coulomb_energy_from_src_index(
                flattened_charges_raw, 
                emb['data'].src_index, 
                emb['data'].pairwise_distances
            )        

            charges_raw = flattened_charges_raw.unsqueeze(-1)

        else:
            
            e_charge_single = coulomb_energy_from_src_index(
                charges_raw, 
                emb['data'].src_index, 
                emb['data'].pairwise_distances
            )
        
        if self.gnn_cfg.equil_charges_tf: 
            hardness = self.hardness_ffn(node_reps)  # (N,)
            electronegativity = self.electronegativity_ffn(node_reps)  # (N,)
            # Additional logic for equilibration can be added here
            en_electrostatic = (electronegativity * charges_raw).view(-1)
            en_hardness = 0.5 * (hardness * charges_raw**2).view(-1)
            e_charge_single += en_electrostatic 
            e_charge_single += en_hardness

        if self.latent_dim_out == 2:
            
            energy_spin = heisenberg_energy_from_src_index(
                q =  charges_two_comp, 
                src_index=emb['data'].src_index,
                dist_pairwise=emb['data'].pairwise_distances,
                j_coupling_nn=self.coupling_ffn,
            )
            #print("Heisenberg energy shape: ", energy_spin.shape)
            e_charge_single += energy_spin
            #else: 
            


        e_charge = compilable_scatter(
            e_charge_single,
            index=emb["data"].node_batch,
            dim_size=emb["data"].max_batch_size,
            dim=0,
            reduce=self.energy_reduce,
        )
        #print("Coulomb energy (before scatter): ", e_charge_single.shape)
        #print("Coulomb energy (after scatter): ", e_charge.shape)
        

        
        
        return e_charge


    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        self.forward_fn_reps = (
            torch.compile(self.compiled_forward_reps) 
            if self.global_cfg.use_compile
            else self.compiled_forward_reps 
        )
        
        self.forward_fn = (
            torch.compile(self.compiled_forward) 
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        self.forward_fn_coulomb = (
            torch.compile(self.compiled_forward_charges) 
            if self.global_cfg.use_compile
            else self.compiled_forward_charges
        )

        node_reps = self.forward_fn_reps(emb)   
        energy_output_sr = self.forward_fn(emb, node_reps) 
        e_charge = self.forward_fn_coulomb(emb, node_reps)  

        if len(energy_output_sr.shape) == 0:
            energy_output_sr = energy_output_sr.unsqueeze(0)
        if len(e_charge.shape) == 0:
            e_charge = e_charge.unsqueeze(0)
        
        # unpad once
        results_to_unpad = {
            "energy": energy_output_sr,
            "energy_coul": e_charge
        }


        """
        if self.gnn_cfg.heisenberg_tf: 
            results_to_unpad["energy_spin"] = energy_lr_dict["energy_spin"] 

        
        if self.gnn_cfg.equil_charges_tf: 
            pass # likely needs to unpad contribution from equil only 
            #results_to_unpad["charges"] = dummy

        if self.ret_charges:
            # check if this is correct for spin charges
            ret_dict["charges"] = energy_lr_dict["charges"]
            if self.gnn_cfg.two_component_latent_charge:
                ret_dict["spin_charges"] = energy_lr_dict["spin_charges"]
        """

        res_unpad = unpad_result(
            results=results_to_unpad,
            data=emb["data"],
        )
        
        print(res_unpad["energy"].mean().item(), res_unpad["energy_coul"].mean().item())
        
        ret_dict = {
            "energy": res_unpad["energy"] + res_unpad["energy_coul"]
        }
        #print("ret dict: ", ret_dict)
        return ret_dict



@registry.register_model("EScAIP_grad_efs_head")
class EScAIPGradientEFSHead(EScAIPEnergyHead):  # type: ignore
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
        energy_output = self.compiled_forward(emb)
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
            data=emb["data"],
        )
