from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.allscaip.configs import AllScAIPConfigs, init_configs
from fairchem.core.models.allscaip.modules.graph_attention_block import (
    GraphAttentionBlock,
)
from fairchem.core.models.allscaip.modules.input_block import InputBlock
from fairchem.core.models.allscaip.utils.data_preprocess import (
    data_preprocess_radius_graph,
    unpad_results,
)
from fairchem.core.models.allscaip.utils.nn_utils import get_feedforward
from fairchem.core.models.base import BackboneInterface, HeadInterface
from fairchem.core.models.escaip.utils.graph_utils import (
    compilable_scatter,
    get_displacement_and_cell,
)
from fairchem.core.models.escaip.utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
    init_linear_weights,
    no_weight_decay,
)
from fairchem.core.models.uma.escn_md import GradRegressConfig
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    UMATask,
)

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.allscaip.custom_types import GraphAttentionData
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.mlip_unit import Task


@registry.register_model("AllScAIP_backbone")
class AllScAIPBackbone(nn.Module, BackboneInterface):
    """
    All-to-all Scaled Attention Interactomic Potential (AllScAIP) backbone.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(AllScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.direct_forces = cfg.global_cfg.direct_forces
        self.regress_stress = cfg.global_cfg.regress_stress
        self.regress_config = GradRegressConfig(
            direct_forces=self.direct_forces,
            forces=self.regress_forces,
            stress=self.regress_stress,
        )
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

    @classmethod
    def build_inference_settings(cls, settings: InferenceSettings) -> dict:
        """
        Build backbone config overrides from inference settings.
        """
        overrides = {}

        if settings.compile:
            if settings.max_atoms is None:
                raise ValueError(
                    "max_atoms must be set in InferenceSettings when compile=True. "
                    "AllScAIP requires padding to a fixed size for torch.compile."
                )
            overrides["use_compile"] = True
            overrides["use_padding"] = True
            overrides["max_atoms"] = settings.max_atoms
        else:
            overrides["use_compile"] = False
            overrides["use_padding"] = False

        return overrides

    def validate_tasks(self, dataset_to_tasks: dict[str, list]) -> None:
        """
        Validate that task datasets are compatible with this backbone.
        """
        if self.dataset_list:
            assert set(dataset_to_tasks.keys()).issubset(
                set(self.dataset_list)
            ), "Datasets in tasks is not a strict subset of datasets in backbone."

    def prepare_for_inference(self, data: AtomicData, settings: InferenceSettings):
        return self

    def get_default_untrained_tasks(
        self,
        checkpoint_tasks: dict[str, Task],
        inference_settings: InferenceSettings,
    ) -> list[Task]:
        return []

    def on_predict_check(self, data: AtomicData) -> None:
        pass

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

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

    def compiled_forward(self, data: GraphAttentionData):
        # input block
        with record_function("input_block"):
            neighbor_reps = self.input_block(data)

        # transformer blocks
        for idx in range(self.global_cfg.num_layers):
            neighbor_reps = self.transformer_blocks[idx](
                data, neighbor_reps, layer_idx=idx
            )

        return {
            "data": data,
            "node_reps": neighbor_reps[:, 0].to(torch.float32),
        }

    @torch.compiler.disable()
    @conditional_grad(torch.enable_grad())
    def forward(self, data: AtomicData):
        # TODO: remove this when FairChem fixes this
        data["atomic_numbers"] = data["atomic_numbers"].long()  # type: ignore
        data["atomic_numbers_full"] = data["atomic_numbers"]  # type: ignore
        data["batch_full"] = data["batch"]  # type: ignore

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
        with record_function("backbone_compile_forward"):
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


class AllScAIPHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: AllScAIPBackbone):  # type: ignore
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

        self.regress_forces = backbone.regress_forces
        self.regress_stress = backbone.regress_stress
        self.direct_forces = backbone.direct_forces

        normalization = NormalizationType(self.reg_cfg.normalization)
        self.node_norm = get_normalization_layer(normalization)(
            self.global_cfg.hidden_size
        )

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

    def get_node_reps(self, emb):
        return self.node_norm(emb["node_reps"])

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("AllScAIP_direct_force_head")
class AllScAIPDirectForceHead(AllScAIPHeadBase):
    def __init__(self, backbone: AllScAIPBackbone):  # type: ignore
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
        node_reps = self.get_node_reps(emb)
        force_direction = self.force_ffn(node_reps)
        return force_direction

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb) -> dict[str, torch.Tensor]:
        self.forward_fn = (
            torch.compile(self.compiled_forward)  # type: ignore
            if self.global_cfg.use_compile
            else self.compiled_forward
        )
        with record_function("force_head_compile_forward"):
            force_output = self.forward_fn(emb)  # type: ignore
        return unpad_results(
            results={"forces": force_output},
            data=emb["data"],
        )


@registry.register_model("AllScAIP_energy_head")
class AllScAIPEnergyHead(AllScAIPHeadBase):
    def __init__(self, backbone: AllScAIPBackbone):  # type: ignore
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
        node_reps = self.get_node_reps(emb)
        energy_output = self.energy_ffn(node_reps)

        # Mask out padded nodes to prevent them from contributing to energy
        # (padded nodes have node_batch=0 which would incorrectly add to batch 0's energy)
        energy_output = energy_output * emb["data"].node_padding_mask.unsqueeze(-1)

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
    def forward(self, data, emb) -> dict[str, torch.Tensor]:
        self.forward_fn = (
            torch.compile(self.compiled_forward)  # type: ignore
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        with record_function("energy_head_compile_forward"):
            energy_output = self.forward_fn(emb)  # type: ignore
        if len(energy_output.shape) == 0:
            energy_output = energy_output.unsqueeze(0)
        return unpad_results(
            results={"energy": energy_output},
            data=emb["data"],
        )


@registry.register_model("AllScAIP_grad_energy_force_stress_head")
class AllScAIPGradientEnergyForceStressHead(AllScAIPEnergyHead):  # type: ignore
    """
    Do not support torch.compile
    """

    def __init__(
        self,
        backbone: AllScAIPBackbone,  # type: ignore
        prefix: str | None = None,
        wrap_property: bool = True,
    ):
        super().__init__(backbone)
        self.prefix = prefix
        self.wrap_property = wrap_property

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}
        with record_function("grad_head_energy"):
            energy_output = self.compiled_forward(emb)
        if len(energy_output.shape) == 0:
            energy_output = energy_output.unsqueeze(0)

        outputs[energy_key] = (
            {"energy": energy_output} if self.wrap_property else energy_output
        )

        if self.regress_stress:
            with record_function("grad_head_stress_forces"):
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
                outputs[forces_key] = (
                    {"forces": forces} if self.wrap_property else forces
                )
                outputs[stress_key] = (
                    {"stress": stress} if self.wrap_property else stress
                )
                data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            with record_function("grad_head_forces"):
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_output.sum(), data["pos"], create_graph=self.training
                    )[0]
                )
                outputs[forces_key] = (
                    {"forces": forces} if self.wrap_property else forces
                )

        return unpad_results(
            results=outputs,
            data=emb["data"],
        )
