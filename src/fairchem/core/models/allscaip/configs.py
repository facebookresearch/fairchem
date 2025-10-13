from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Literal


@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_forces: bool
    hidden_size: int  # divisible by 2 and num_heads
    num_layers: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ] = "gelu"
    regress_stress: bool = False
    use_compile: bool = True
    use_padding: bool = True
    use_residual_scaling: bool = True
    use_node_path: bool = True
    dataset_list: list = field(default_factory=list)


@dataclass
class MolecularGraphConfigs:
    max_num_elements: int
    max_atoms: int
    max_batch_size: int
    max_radius: float
    knn_k: int
    knn_soft: bool
    knn_sigmoid_scale: float
    knn_lse_scale: float
    knn_use_low_mem: bool
    knn_pad_size: int
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"] = (
        "gaussian"
    )
    use_envelope: bool = True


@dataclass
class GraphNeuralNetworksConfigs:
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
    atten_num_heads: int
    node_direction_expansion_size: int = 10
    edge_direction_expansion_size: int = 6
    edge_distance_expansion_size: int = 512
    output_hidden_layer_multiplier: int = 2
    ffn_hidden_layer_multiplier: int = 2
    attn_num_freq: int = 32
    freequency_list: list = field(default_factory=lambda: [20, 10, 4, 10, 20])
    energy_reduce: Literal["sum", "mean"] = "sum"
    use_freq_mask: bool = True
    use_sincx_mask: bool = True


@dataclass
class RegularizationConfigs:
    normalization: Literal["layernorm", "rmsnorm", "skip"] = "rmsnorm"
    mlp_dropout: float = 0.0
    atten_dropout: float = 0.0


@dataclass
class AllScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs


def resolve_type_hint(cls, field):
    """Resolves forward reference type hints from string to actual class objects."""
    if isinstance(field.type, str):
        resolved_type = getattr(cls, field.type, None)
        if resolved_type is None:
            resolved_type = globals().get(field.type, None)  # Try global scope
        if resolved_type is None:
            return field.type  # Fallback to string if not found
        return resolved_type
    return field.type


def init_configs(cls, kwargs):
    """
    Initialize a dataclass with the given kwargs.
    """
    init_kwargs = {}
    for _field in fields(cls):
        field_name = _field.name
        field_type = resolve_type_hint(cls, _field)  # Resolve type

        if is_dataclass(field_type):  # Handle nested dataclass
            init_kwargs[_field.name] = init_configs(field_type, kwargs)
        elif field_name in kwargs:  # Direct assignment
            init_kwargs[field_name] = kwargs[field_name]
        elif _field.default is not MISSING:  # Assign default if available
            init_kwargs[field_name] = _field.default
        elif _field.default_factory is not MISSING:  # Handle default_factory
            init_kwargs[field_name] = _field.default_factory()
        else:
            raise ValueError(
                f"Missing required configuration parameter: '{field_name}' in '{cls.__name__}'"
            )

    return cls(**init_kwargs)
