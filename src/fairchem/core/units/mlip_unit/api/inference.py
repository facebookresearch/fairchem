"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # - needed at runtime for dataclass field type resolution

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

ALLOWED_DTYPES = ["float32", "float64"]


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
    # inference speed.
    activation_checkpointing: bool = True

    # Flag to enable or disable the merging of MOLE experts during
    # inference. If this is used, the input composition, total charge
    # and spin MUST remain constant throughout the simulation this will
    # slightly increase speed and reduce memory footprint used by the
    # parameters significantly
    merge_mole: bool = False

    # Flag to enable or disable the compilation of the inference model.
    compile: bool = False

    # Deprecated
    # Flag to enable or disable the use of CUDA Graphs for compute
    # This flag is no longer used and will be removed in future versions
    wigner_cuda: bool | None = None

    # Flag to enable or disable the generation of external graphs during
    # inference.
    external_graph_gen: bool = False

    # Internal graph generation version to use during inference.
    # version 2 is the an internal implementation that is optimized for gpu.
    # version 3 uses Nvidia alchemi library's neighbor list.
    internal_graph_gen_version: int = 2

    # Number of internal torch threads to use for inference
    torch_num_threads: int | None = None

    # Used for padding edges during inference, this is useful to reduce recompiling time during dynamic inference runs
    edge_chunk_size: int | None = None

    # Flag to enable quaternion-based Wigner D matrix computation.
    use_quaternion_wigner: bool = True

    # Base precision dtype for model parameters and input data.
    # All model parameters, buffers, and float input tensors will be
    # cast to this dtype. Set to torch.float64 for higher precision.
    # current accepted values are float32 and float64
    base_precision_dtype_str: str = "float32"

    # Execution backend mode for the backbone. The default is "general".
    # Set to "umas_fast_pytorch" to enable block-diagonal SO2 GEMM conversion for faster inference.
    # Set to "umas_fast_gpu" to enable highly optimized backend with triton kernels for maximum speed.
    execution_mode: str = "general"

    def base_precision_dtype(self) -> torch.dtype:
        assert (
            self.base_precision_dtype_str in ALLOWED_DTYPES
        ), f"base_precision_dtype must be one of {ALLOWED_DTYPES}, got {self.base_precision_dtype_str}"
        return getattr(torch, self.base_precision_dtype_str)


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
        execution_mode="general",
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
    "traineval": inference_settings_traineval(),
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
