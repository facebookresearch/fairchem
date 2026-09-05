"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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

ALLOWED_DTYPES = [torch.float32, torch.float64]


class MergeMoleConsistencyError(ValueError):
    """Raised when input data is incompatible with an already merged MOLE model."""


def validate_uma_atoms_data(atoms, task_name: str, logger=None) -> None:
    """
    UMA-specific validation: handle charge/spin for OMOL task.

    Sets default values for charge and spin in atoms.info and validates
    they are within acceptable ranges.

    Args:
        atoms: ASE Atoms object
        task_name: Task name (e.g., "omol", "omat")
        logger: Optional logger for warnings. If None, uses Python logging module.
    """
    import logging as log_module

    import numpy as np

    _logger = logger if logger is not None else log_module

    # Set charge defaults
    if "charge" not in atoms.info:
        if task_name == UMATask.OMOL.value:
            _logger.warning(
                "task_name='omol' detected, but charge is not set in atoms.info. "
                "Defaulting to charge=0. Ensure charge is an integer representing "
                "the total charge on the system and is within the range -100 to 100."
            )
        atoms.info["charge"] = DEFAULT_CHARGE

    # Set spin defaults (OMOL uses spin=1, others use spin=0)
    if "spin" not in atoms.info:
        if task_name == UMATask.OMOL.value:
            atoms.info["spin"] = DEFAULT_SPIN_OMOL
            _logger.warning(
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
    # especially for large number of atoms (i.e. 10k+) at a slight cost to
    # inference speed.
    activation_checkpointing: bool = True

    # Flag to enable or disable the merging of MOLE experts during
    # inference. This slightly increases speed and significantly reduces
    # parameter memory. If composition, task, total charge, or spin changes,
    # MLIPPredictUnit falls back to an unmerged model.
    merge_mole: bool = False

    # Restrict input gradients of the two radial first layers to distance features.
    radial_first_linear_prefix_grad: bool = False

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
    # Accepts a torch.dtype or a string in ALLOWED_DTYPES (e.g. "float32").
    base_precision_dtype: torch.dtype | str = torch.float32

    # Execution backend mode for the backbone.
    # Set to "general" for the default execution mode that works across all models and hardware.
    # Set to "umas_fast_pytorch" to enable block-diagonal SO2 GEMM conversion for faster inference.
    # Set to "umas_fast_gpu" to enable highly optimized backend with triton kernels for maximum speed.
    # If None, the predictor will decide the best execution mode based on the model and hardware capabilities (e.g., will choose "umas_fast_gpu" for uma-s if running on compatible Nvidia GPU).
    execution_mode: str | None = None

    # New fields for untrained derivative properties
    # These flags request computation of properties NOT in the checkpoint's task list.
    # If a property is already in the checkpoint (e.g., omol_forces task exists),
    # it will be computed regardless of these flags.
    # Specify datasets as a set of strings (e.g., {"omol", "oc20"}).
    # Empty set means no untrained properties will be computed (default).
    predict_untrained_forces: set[str] = field(default_factory=set)
    predict_untrained_stress: set[str] = field(default_factory=set)
    predict_untrained_hessian: set[str] = field(default_factory=set)
    # Disable for backends whose custom backward operators lack vmap rules.
    # The loop uses less memory but performs one backward pass per force component.
    hessian_vmap: bool = True

    # When True, allow backbones to add their default untrained tasks
    # (e.g., eSCNMDBackbone adds stress for all energy tasks by default)
    auto_add_default_untrained_tasks: bool = True

    # Maximum number of atoms per system for padding. Required when
    # compile=True for models that use padding (e.g., AllScAIP).
    # All inputs will be padded to this size. Larger values consume more
    # VRAM but allow bigger systems; reduce if you run into OOM errors.
    max_atoms: int | None = None

    def __post_init__(self):
        if self.radial_first_linear_prefix_grad:
            if self.execution_mode != "umas_fast_gpu":
                raise ValueError(
                    "radial_first_linear_prefix_grad requires "
                    "execution_mode='umas_fast_gpu'"
                )
            if self.predict_untrained_hessian:
                raise ValueError(
                    "radial_first_linear_prefix_grad does not support Hessians"
                )
        if isinstance(self.base_precision_dtype, str):
            self.base_precision_dtype = getattr(torch, self.base_precision_dtype)
            assert (
                self.base_precision_dtype in ALLOWED_DTYPES
            ), f"base_precision_dtype must be one of {ALLOWED_DTYPES}, got {self.base_precision_dtype}"

    def to_omegaconf(self) -> dict:
        """
        Return an OmegaConf-compatible dict for use with hydra.utils.instantiate.

        torch.dtype is not natively serializable by OmegaConf, so
        base_precision_dtype is stored as a string; __post_init__ converts it
        back to a torch.dtype when InferenceSettings is reinstantiated.
        """
        config = asdict(self)
        config["base_precision_dtype"] = str(self.base_precision_dtype).replace(
            "torch.", ""
        )
        config["_target_"] = (
            "fairchem.core.units.mlip_unit.api.inference.InferenceSettings"
        )
        return config


# Default to the fast path while retaining full FP32 precision. If the input
# changes in a way that is incompatible with the merged MOLE model, the
# predictor automatically falls back to an unmerged and uncompiled model.
def inference_settings_default():
    return InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )


# Batch mode is the stable entry point for heterogeneous inputs. It currently
# uses the general unmerged and uncompiled path; keeping it as a named mode lets
# us optimize heterogeneous batches independently in future releases (for
# example, by enabling compilation without merging MOLE).
def inference_settings_batch():
    return InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )


# Turbo uses the same fast path as the default settings, with TF32 enabled for
# additional speed on supported hardware. It remains opt-in because it trades
# a small amount of precision for speed.
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
    "batch": inference_settings_batch(),
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
