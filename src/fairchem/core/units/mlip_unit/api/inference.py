"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fairchem.core.common.utils import StrEnum


class UMATask(StrEnum):
    OMOL = "omol"
    OMAT = "omat"
    ODAC = "odac"
    OC20 = "oc20"
    OC25 = "oc25"
    OMC = "omc"


CHARGE_RANGE = [-100, 100]
SPIN_RANGE = [0, 100]
DEFAULT_CHARGE = 0
DEFAULT_SPIN_OMOL = 1
DEFAULT_SPIN = 0


@dataclass
class MLIPInferenceCheckpoint:
    # contains original config that trained the model
    model_config: dict

    # the model state dict
    model_state_dict: dict

    # the ema state dict, used for inference
    ema_state_dict: dict

    # the config containing information about "tasks", a task contains
    # things like normalizers and element references and tells the model
    # how to produce the correct outputs
    tasks_config: dict


@dataclass
class InferenceSettings:
    # Flag to enable or disable the use of tf32 data type for inference.
    # TF32 will slightly reduce accuracy compared to FP32 but will still
    # keep energy conservation in most cases.
    tf32: bool = False

    # Flag to enable or disable activation checkpointing during
    # inference. This will dramatically decrease the memory footprint
    # especially for large number of atoms (ie 10+) at a slight cost to
    # inference speed. If set to None, the setting from the model
    # checkpoint will be used.
    activation_checkpointing: bool | None = None

    # Flag to enable or disable the merging of MOLE experts during
    # inference. If this is used, the input composition, total charge
    # and spin MUST remain constant throughout the simulation this will
    # slightly increase speed and reduce memory footprint used by the
    # parameters significantly
    merge_mole: bool = False

    # Flag to enable or disable the compilation of the inference model.
    compile: bool = False

    # Flag to enable or disable the use of CUDA Graphs for compute
    # This flag is no longer used and will be removed in future versions
    wigner_cuda: bool | None = None

    # Flag to enable or disable the generation of external graphs during
    # inference. If set to None, the setting from the model checkpoint
    # will be used.
    external_graph_gen: bool | None = None

    # Deprecated
    # Not recommended using! manually selects the version of graph gen
    # code if external_graph_gen is false, if set of None, will default
    # to whatever is in the checkpoint
    internal_graph_gen_version: int | None = None

    # Number of internal torch threads to use for inference
    torch_num_threads: int | None = None

    edge_chunk_size: int | None = None

    # Execution backend mode for the backbone. If set to None, the
    # checkpoint default ("general") is used. Set to "umas_fast_pytorch"
    # to enable block-diagonal SO2 GEMM conversion for faster inference.
    execution_mode: str | None = None

    # New fields for untrained derivative properties
    # These flags request computation of properties NOT in the checkpoint's task list.
    # If a property is already in the checkpoint (e.g., omol_forces task exists),
    # it will be computed regardless of these flags.
    # Specify datasets as a set of strings (e.g., {"omol", "oc20"}).
    # Empty set means no untrained properties will be computed (default).
    compute_untrained_forces: set[str] = field(default_factory=set)
    compute_untrained_stress: set[str] = field(default_factory=set)
    compute_untrained_hessian: set[str] = field(default_factory=set)
    hessian_vmap: bool = True  # Use fast vmap vs memory-efficient loop


# this is most general setting that works for most systems and models,
# not optimized for speed
def inference_settings_default():
    return InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )


# this setting is designed for running long simulations or optimizations
# where the system composition (atoms, charge, spin) stays constant over
# the course the simulation. For smaller systems
# activation_checkpointing can be turned off for some extra speed gain
def inference_settings_turbo():
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )


# this mode corresponds to the default settings used for training and evaluation
def inference_settings_traineval():
    return InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=False,
        internal_graph_gen_version=1,
    )


NAME_TO_INFERENCE_SETTING = {
    "default": inference_settings_default(),
    "turbo": inference_settings_turbo(),
}


def guess_inference_settings(settings: str | InferenceSettings):
    if isinstance(settings, str):
        assert (
            settings in NAME_TO_INFERENCE_SETTING
        ), f"inference setting name must be one of {NAME_TO_INFERENCE_SETTING.keys()}"
        return NAME_TO_INFERENCE_SETTING[settings]
    elif isinstance(settings, InferenceSettings):
        return settings
    else:
        raise ValueError(
            f"InferenceSetting can be a str or InferenceSettings object, found {settings.__class__}"
        )
