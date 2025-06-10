from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal


@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_forces: bool
    hidden_size: int  # divisible by 2 and num_heads
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True
    use_fp16_backbone: bool = False
    regress_stress: bool = False


@dataclass
class MolecularGraphConfigs:
    use_pbc: bool
    use_pbc_single: bool
    otf_graph: bool
    max_neighbors: int
    max_radius: float
    max_num_elements: int
    max_atoms: int
    max_batch_size: int
    knn_k: int
    knn_soft: bool
    knn_sigmoid_scale: float
    knn_lse_scale: float
    knn_use_low_mem: bool
    knn_pad_size: int
    enforce_max_neighbors_strictly: bool
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]
    use_envelope: bool


@dataclass
class GraphNeuralNetworksConfigs:
    num_layers: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int
    use_angle_embedding: Literal["scalar", "bias", "none"]
    angle_expansion_size: int = 10
    angle_embedding_size: int = 8
    use_graph_attention: bool = False
    use_message_gate: bool = False
    use_global_readout: bool = False
    use_frequency_embedding: bool = False
    freequency_list: list = field(default_factory=list)
    energy_reduce: Literal["sum", "mean"] = "mean"


@dataclass
class RegularizationConfigs:
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]
    node_ffn_dropout: float
    edge_ffn_dropout: float
    scalar_output_dropout: float
    vector_output_dropout: float


@dataclass
class EScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs


def init_configs(cls: type[EScAIPConfigs], kwargs: dict[str, Any]) -> EScAIPConfigs:
    """
    Initialize a dataclass with the given kwargs.
    """
    init_kwargs = {}
    for _field in fields(cls):
        if is_dataclass(_field.type):
            init_kwargs[_field.name] = init_configs(_field.type, kwargs)
        elif _field.name in kwargs:
            init_kwargs[_field.name] = kwargs[_field.name]
        elif _field.default is not None:
            init_kwargs[_field.name] = _field.default
        else:
            raise ValueError(
                f"Missing required configuration parameter: '{_field.name}'"
            )

    return cls(**init_kwargs)
