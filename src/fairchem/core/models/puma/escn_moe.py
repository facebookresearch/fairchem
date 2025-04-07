from __future__ import annotations

import logging
import math
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
from torch.nn import functional as F

from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.puma.escn_md import eSCNMDBackbone

with suppress(ModuleNotFoundError):
    import dgl  # try to use DGL if available

import functools

from fairchem.core.common.registry import registry
from fairchem.core.models.escn.escn import EdgeBlock
from fairchem.core.models.puma.nn.so2_layers import SO2_Convolution


def _softmax(x):
    return torch.softmax(x, dim=1) + 0.005


def _pnorm(x):
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[0], p=1.0, dim=1)


def norm_str_to_fn(act):
    if act == "softmax":
        return _softmax  # (self.task_embedding(dataset_vec), dim=1)
    elif act == "pnorm":
        return _pnorm
        # return partial(torch.nn.functional.normalize, p=1.0, dim=1)
        # torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
        # return torch.softmax(self.task_embedding(dataset_vec), dim=1)
    else:
        raise ValueError


@dataclass
class MOEGlobals:
    expert_mixing_coefficients: torch.Tensor
    routing_idxs: torch.Tensor
    natoms: torch.Tensor


class GlobalBlock(torch.nn.Module):
    """
    Global block: Estimate global embedding

    Args:
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        hidden_channels,
        num_experts,
        distance_expansion,
        max_num_elements,
        act,
    ):
        super().__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts

        # Message block
        self.edge_block = EdgeBlock(
            self.hidden_channels,
            distance_expansion,
            max_num_elements,
            self.act,
            # fc1_dist_bias=fc1_dist_bias,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_pre = nn.Linear(
            self.hidden_channels,
            self.hidden_channels,
        )
        init.constant_(self.fc1_pre.bias, 0.0)

        self.fc1_post = nn.Linear(
            self.hidden_channels,
            self.hidden_channels,
        )
        init.constant_(self.fc1_post.bias, 0.0)

        self.fc2_post = nn.Linear(
            self.hidden_channels,
            self.num_experts,
        )
        init.constant_(self.fc2_post.bias, 0.0)

    def forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index,
        batch,
        batch_size,
    ):
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.edge_block(
            edge_distance,
            atomic_numbers[edge_index[0]],  # Source atom atomic number
            atomic_numbers[edge_index[1]],  # Target atom atomic number
        )
        x_edge = self.act(self.fc1_pre(x_edge))

        x_global = torch.zeros(
            batch_size, self.hidden_channels, device=x_edge.device, dtype=x_edge.dtype
        )
        batch_idx = batch[edge_index[1]]
        x_global.index_add_(0, batch_idx, x_edge)
        num_edges = torch.zeros(batch_size, device=x_edge.device, dtype=x_edge.dtype)
        num_edges.index_add_(
            0,
            batch_idx,
            torch.ones(len(batch_idx), device=x_edge.device, dtype=x_edge.dtype),
        )
        x_global = x_global / (num_edges.view(-1, 1) + 0.001)

        # num_atoms = len(atomic_numbers)
        # x_global = torch.zeros(num_atoms, self.hidden_channels, device=x_edge.device, dtype=x_edge.dtype)
        # x_global.index_add_(0, edge_index[1], x_edge)
        # num_edges = torch.zeros(num_atoms, device=x_edge.device, dtype=x_edge.dtype)
        # num_edges.index_add_(0, edge_index[1], torch.ones(len(edge_index[1]), device=x_edge.device, dtype=x_edge.dtype))
        # x_global = x_global / (num_edges.view(-1, 1) + 0.001)

        # x_global = x_global + self.act(self.fc1_post(x_global))
        # x_global = self.act(self.fc2_post(x_global))
        # x_global = torch.softmax(x_global, dim=1) + 0.05

        x_global = self.act(self.fc1_post(x_global))
        x_global = self.fc2_post(x_global)
        return x_global


class MOELinearDGLFractional(torch.nn.Module):
    def __init__(
        self,
        fraction_moe: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.in_features = kwargs["in_features"]
        self.out_features = kwargs["out_features"]
        self.bias = kwargs["bias"]

        self.linear = None
        if fraction_moe != 1.0:
            moe_out_features = int(fraction_moe * kwargs["out_features"])
            regular_out_features = kwargs["out_features"] - moe_out_features
            logging.info(
                f"Creating channels: {moe_out_features} moe and {regular_out_features} regular"
            )
            kwargs["out_features"] = moe_out_features
            self.linear = torch.nn.Linear(
                kwargs["in_features"], regular_out_features, bias=kwargs["bias"]
            )

        self.moe_linear = MOELinearDGL(**kwargs)

    def forward(self, x):
        moe_out = self.moe_linear(x)
        if self.linear is not None:
            regular_out = self.linear(x)
            return torch.concatenate([moe_out, regular_out], dim=moe_out.ndim - 1)
        return moe_out


def init_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


class MOELinearDGL(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_moe_tensors,
        bias: bool,
    ):
        super().__init__()

        assert global_moe_tensors is not None
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_moe_tensors = global_moe_tensors

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_moe_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = dgl.ops.segment_mm(x, weights, self.global_moe_tensors.routing_idxs)
        elif x.ndim == 3:
            r = dgl.ops.segment_mm(
                x.reshape(-1, x_shape[-1]),
                weights,
                self.global_moe_tensors.routing_idxs * x_shape[1],
            ).reshape(*x_shape[:-1], -1)
        elif x.ndim == 4:
            # assume this is one per system
            assert 1 == 0  # revist natoms in graph parallel
            r = dgl.ops.segment_mm(
                x.reshape(-1, x_shape[-1]),
                weights,
                self.global_moe_tensors.natoms * x_shape[1] * x_shape[2],
            ).reshape(*x_shape[:-1], -1)
        else:
            raise ValueError("x.ndim not in (2,3) not allows")
        if self.bias is not None:
            r += self.bias
        return r


class MOELinear(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_moe_tensors,
        bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_moe_tensors = global_moe_tensors

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_moe_tensors.expert_mixing_coefficients,
            )

        start = 0
        end = 0
        out = []
        for idx in range(len(self.global_moe_tensors.routing_idxs)):
            end = start + self.global_moe_tensors.routing_idxs[idx]
            if start != end:
                assert x.shape[0] > start
                out.append(F.linear(x[start:end], weights[idx], bias=self.bias))
                start = end
        assert x.shape[0] == end

        return torch.concatenate(out, dim=0)


def recursive_replace_so2m0_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2m0_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)


def recursive_replace_so2_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)
            for so2_module in child.so2_m_conv:
                so2_module.fc = replacement_factory(so2_module.fc).to(target_device)


def recursive_replace_all_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_all_linear(child, replacement_factory)


def recursive_replace_notso2_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, SO2_Convolution):
            continue
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_notso2_linear(child, replacement_factory)


def model_search_and_replace(
    model, module_search_function, replacement_factory, layers=None
):
    if layers is None:
        layers = list(range(len(model.blocks)))
    for layer_idx in layers:
        module_search_function(model.blocks[layer_idx], replacement_factory)


def replace_linear_with_shared_linear(
    existing_linear_module,
    cache,
):
    layer_identifier = (
        existing_linear_module.in_features,
        existing_linear_module.out_features,
        existing_linear_module.bias is not None,
    )
    if layer_identifier in cache:
        return cache[layer_identifier]

    cache[layer_identifier] = existing_linear_module
    return existing_linear_module


def replace_linear_with_MOElinear(
    existing_linear_module,
    global_moe_tensors,
    num_experts,
    fraction_moe,
    moe_layer_type,
    cache=None,
):
    layer_identifier = (
        existing_linear_module.in_features,
        existing_linear_module.out_features,
        existing_linear_module.bias,
    )
    if cache is not None and layer_identifier in cache:
        return cache[layer_identifier]

    if moe_layer_type == "dgl":
        layer = MOELinearDGLFractional(
            num_experts=num_experts,
            global_moe_tensors=global_moe_tensors,
            in_features=existing_linear_module.in_features,
            out_features=existing_linear_module.out_features,
            bias=existing_linear_module.bias is not None,
            fraction_moe=fraction_moe,
        )
    elif moe_layer_type == "pytorch":
        assert fraction_moe == 1.0, "Cannot use fraction_moe with pytorch MoE layer"
        layer = MOELinear(
            num_experts=num_experts,
            global_moe_tensors=global_moe_tensors,
            in_features=existing_linear_module.in_features,
            out_features=existing_linear_module.out_features,
            bias=existing_linear_module.bias is not None,
        )
    else:
        raise ValueError("moe_layer_type must be dgl or pytorch")
    if cache is not None:
        cache[layer_identifier] = layer
    return layer


def standalone_prepare_MOE(model, data, graph, csd_mixed_emb):
    # allow this to bypass all MoE parts
    if model.num_experts == 0:
        return

    with torch.autocast(device_type=data.pos.device.type, enabled=False):
        atomic_numbers_full = data.atomic_numbers_full.long()

        if isinstance(graph, dict):
            # hand escnMD
            edge_distance_full = graph["edge_distance_full"]
            edge_index_full = graph["edge_index_full"]
            edge_index = graph["edge_index"]
        else:
            # assume its a graph object
            edge_distance_full = graph.edge_distance_full
            edge_index_full = graph.edge_index_full
            edge_index = graph.edge_index

        model.global_moe_tensors.expert_mixing_coefficients = torch.zeros(
            (len(data), model.num_experts), device=data.batch.device
        )

        # Generate MoE routing embeddings for the full systems

        embeddings = []

        if model.use_global_embedding:
            # generate MoE global routing information
            # This has an embedding for each system in the batch
            x_global_system = model.global_block(
                atomic_numbers_full,
                edge_distance_full,
                edge_index_full,
                data.batch_full,
                len(data.natoms),
            )
            embeddings.append(x_global_system.unsqueeze(0))

        if model.use_composition_embedding:
            composition = torch.zeros(
                data.natoms.shape[0],
                model.sphere_channels,
                device=data.atomic_numbers_full.device,
            ).index_reduce_(
                0,
                data.batch_full,
                model.composition_embedding(data.atomic_numbers_full),
                reduce="mean",
            )
            embeddings.append(composition.unsqueeze(0))

        embeddings.append(csd_mixed_emb[None].float())
        # if taking mean reduction over embeddings
        model.global_moe_tensors.expert_mixing_coefficients = model.routing_mlp(
            torch.vstack(embeddings).transpose(0, 1).reshape(len(data), -1)
        )

        model.global_moe_tensors.expert_mixing_coefficients = model.global_norm(
            model.moe_dropout(model.global_moe_tensors.expert_mixing_coefficients)
        )

        # Generate edge mix_size routing each edge in this instance (GP or not)
        # using its local edge and batch routing

        # Local edge_index is 2xN where [1,:] is the target node, the target node does not
        # have the gp offset applied, which means we need to lookup in the full batch_full
        # _, mix_size = torch.unique(data.batch_full[edge_index[1]], return_counts=True)
        routing_idxs = torch.zeros(
            data.natoms.shape[0],
            dtype=torch.int,
            device=data.batch_full[edge_index[1]].device,
        ).scatter_(0, data.batch_full[edge_index[1]], 1, reduce="add")

        model.global_moe_tensors.routing_idxs = routing_idxs.cpu()
        model.global_moe_tensors.natoms = data.natoms.cpu()

        with torch.no_grad():
            if model.counter % 500 == 0:
                logging.info(
                    f"{model.counter }: Expert variance: "
                    + ",".join(
                        [
                            f"{x:.2e}"
                            for x in model.global_moe_tensors.expert_mixing_coefficients.var(
                                axis=0
                            ).tolist()
                        ]
                    )
                )
                logging.info(
                    f"{model.counter }: Expert mean: "
                    + ",".join(
                        [
                            f"{x:.2e}"
                            for x in model.global_moe_tensors.expert_mixing_coefficients.mean(
                                axis=0
                            ).tolist()
                        ]
                    )
                )
                if model.use_global_embedding:
                    logging.info("Global gating by system:")
                    for idx in range(x_global_system.shape[0]):
                        logging.info(
                            f"{idx}\t"
                            + ",".join(
                                [f"{x:0.3f}" for x in x_global_system[idx].tolist()]
                            )
                        )
                    model.axs[1].imshow(x_global_system.detach().cpu().float().numpy())
                    model.axs[1].set_title("System embeddings")
                model.fig.tight_layout()
                model.plot_ready = True

        model.counter += 1


def initialize_moe(
    model,  # eSCN MOE specific
    num_experts: int = 8,
    moe_dropout: float = 0.0,
    use_global_embedding: bool = True,
    global_norm: str = "softmax",
    act=torch.nn.SiLU,
    fraction_moe: float = 1.0,
    layers_moe=None,
    use_composition_embedding: bool = False,
    moe_layer_type: str = "dgl",
    moe_single: bool = False,
    moe_type: str = "so2",
):
    model.num_experts = num_experts
    if model.num_experts == 0:
        return

    model.fraction_moe = fraction_moe
    model.act = act()

    model.counter = 0

    model.use_global_embedding = use_global_embedding
    model.use_composition_embedding = use_composition_embedding

    if model.use_global_embedding:
        model.global_block = GlobalBlock(
            model.hidden_channels,
            model.sphere_channels,
            model.distance_expansion,
            model.max_num_elements,
            act=model.act,
        )

    # keep these and resize them / populate as needed
    model.global_moe_tensors = MOEGlobals(None, None, None)

    model.moe_dropout = torch.nn.Dropout(moe_dropout)

    model.global_norm = norm_str_to_fn(global_norm)

    model.tokenize = {}

    if model.use_composition_embedding:
        model.composition_embedding = nn.Embedding(
            model.max_num_elements, model.sphere_channels
        )

    model.fig, model.axs = plt.subplots(2, 1)

    mlp_routing_dim = (
        use_global_embedding
        + use_composition_embedding
        + 1  # always use dataset/csd_mixed_emb
    ) * model.sphere_channels
    model.routing_mlp = nn.Sequential(
        nn.Linear(
            mlp_routing_dim,
            num_experts * 2,
            bias=True,
        ),
        nn.SiLU(),
        nn.Linear(
            num_experts * 2,
            num_experts * 2,
            bias=True,
        ),
        nn.SiLU(),
        nn.Linear(
            num_experts * 2,
            num_experts,
            bias=True,
        ),
        nn.SiLU(),
    )

    replacement_factory = functools.partial(
        replace_linear_with_MOElinear,
        num_experts=model.num_experts,
        fraction_moe=model.fraction_moe,
        global_moe_tensors=model.global_moe_tensors,
        moe_layer_type=moe_layer_type,
        cache={} if moe_single else None,
    )

    if moe_type == "so2":
        model_search_and_replace(
            model, recursive_replace_so2_linear, replacement_factory, layers=layers_moe
        )
    elif moe_type == "so2m0":
        model_search_and_replace(
            model,
            recursive_replace_so2m0_linear,
            replacement_factory,
            layers=layers_moe,
        )
    elif moe_type == "all":
        model_search_and_replace(
            model, recursive_replace_all_linear, replacement_factory, layers=layers_moe
        )
    elif moe_type == "notso2":
        model_search_and_replace(
            model,
            recursive_replace_notso2_linear,
            replacement_factory,
            layers=layers_moe,
        )
    else:
        raise ValueError(f"Not a valid moe_type {moe_type}")


@registry.register_model("escnmd_moe_backbone")
class eSCNMDMoeBackbone(eSCNMDBackbone):
    def __init__(
        self,
        # eSCN MOE specific
        num_experts: int = 8,
        moe_dropout: float = 0.0,
        use_global_embedding: bool = True,
        use_composition_embedding: bool = False,
        global_norm: str = "softmax",
        act=torch.nn.SiLU,
        fraction_moe: float = 1.0,
        layers_moe=None,
        moe_layer_type: str = "dgl",
        moe_single: bool = False,
        moe_type: str = "so2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        initialize_moe(
            model=self,
            num_experts=num_experts,
            moe_dropout=moe_dropout,
            use_global_embedding=use_global_embedding,
            global_norm=global_norm,
            act=act,
            fraction_moe=fraction_moe,
            layers_moe=layers_moe,
            use_composition_embedding=use_composition_embedding,
            moe_layer_type=moe_layer_type,
            moe_single=moe_single,
            moe_type=moe_type,
        )

    def prepare_MOE(self, data, graph, csd_mixed_emb):
        standalone_prepare_MOE(self, data, graph, csd_mixed_emb)


@registry.register_model("escnmd_share_so2linear_backbone")
class eSCNMDShareSO2LinearBackbone(eSCNMDBackbone):
    def __init__(
        self,
        share_layers: None | list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        replacement_factory = functools.partial(
            replace_linear_with_shared_linear,
            cache={},
        )
        model_search_and_replace(
            self, recursive_replace_so2_linear, replacement_factory, layers=share_layers
        )


class DatasetSpecificMoEWrapper(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        dataset_names,
        head_cls,
        wrap_property=True,
        head_kwargs={},  # noqa: B006
    ):
        super().__init__()
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.wrap_property = wrap_property

        self.dataset_names = sorted(dataset_names)
        self.dataset_name_to_exp = {
            value: idx for idx, value in enumerate(self.dataset_names)
        }
        self.head = registry.get_model_class(head_cls)(backbone, **head_kwargs)
        # replace all linear layers in the head with MoE
        self.global_moe_tensors = MOEGlobals(None, None, None)
        replacement_factory = functools.partial(
            replace_linear_with_MOElinear,
            num_experts=len(self.dataset_names),
            fraction_moe=1.0,
            global_moe_tensors=self.global_moe_tensors,
            moe_layer_type="pytorch",
            cache=None,
        )
        recursive_replace_all_linear(self.head, replacement_factory)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.global_moe_tensors.routing_idxs = torch.zeros(
            data.natoms.shape[0], dtype=torch.int, device=emb["batch"].device
        ).scatter(0, emb["batch"], 1, reduce="add")  # data.natoms.cpu()
        self.global_moe_tensors.natoms = emb["batch"].shape[0]
        data_batch_full = data.batch_full.cpu()

        # generate a one hot mask based on dataset , one for each system
        self.global_moe_tensors.expert_mixing_coefficients = (
            torch.zeros(
                data.natoms.shape[0],
                len(self.dataset_name_to_exp),
            )
            .scatter(
                1,
                torch.tensor(
                    [
                        self.dataset_name_to_exp[dataset_name]
                        for dataset_name in data.dataset
                    ],
                ).unsqueeze(1),
                1.0,
            )
            .to(data.pos.device)
        )

        # print(f"datanatoms:{data.natoms}:routing_idxs:{self.global_moe_tensors.routing_idxs}")

        # run the internal head
        head_output = self.head(data, emb)

        # breakout the outputs to correct heads named by datasetname
        np_dataset_names = np.array(data.dataset)
        full_output = {}
        for dataset_name in self.dataset_names:
            dataset_mask = np_dataset_names == dataset_name
            for key, moe_output_tensor in head_output.items():
                # TODO cant we use torch.zeros here?
                output_tensor = torch.full(
                    moe_output_tensor.shape, 0.0, device=moe_output_tensor.device
                )  # float('inf'))
                if dataset_mask.any():
                    if output_tensor.shape[0] == dataset_mask.shape[0]:
                        output_tensor[dataset_mask] = moe_output_tensor[dataset_mask]
                    else:  # assume atoms are the first dimension
                        atoms_mask = torch.isin(
                            data_batch_full,
                            torch.where(torch.from_numpy(dataset_mask))[0],
                        )
                        output_tensor[atoms_mask] = moe_output_tensor[atoms_mask]
                full_output[f"{dataset_name}_{key}"] = (
                    {key: output_tensor} if self.wrap_property else output_tensor
                )

        return full_output
