"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig

from fairchem.core.common.utils import load_state_dict, match_state_dict

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def load_inference_model(
    checkpoint_location: str,
    overrides: dict | None = None,
    use_ema: bool = False,
    return_checkpoint: bool = True,
) -> tuple[torch.nn.Module, MLIPInferenceCheckpoint] | torch.nn.Module:
    checkpoint: MLIPInferenceCheckpoint = torch.load(
        checkpoint_location, map_location="cpu", weights_only=False
    )

    if overrides is not None:
        checkpoint.model_config = update_configs(checkpoint.model_config, overrides)

    model = hydra.utils.instantiate(checkpoint.model_config)
    if use_ema:
        model = torch.optim.swa_utils.AveragedModel(model)
        model_dict = model.state_dict()
        ema_state_dict = checkpoint.ema_state_dict

        n_averaged = ema_state_dict["n_averaged"]
        del model_dict["n_averaged"]
        del ema_state_dict["n_averaged"]

        matched_dict = match_state_dict(model_dict, ema_state_dict)

        matched_dict["n_averaged"] = n_averaged

        load_state_dict(model, matched_dict, strict=True)
    else:
        load_state_dict(model, checkpoint.model_state_dict, strict=True)

    return (model, checkpoint) if return_checkpoint is True else model


def load_tasks(checkpoint_location: str) -> list[Task]:
    """
    Load tasks from a checkpoint file.

    Args:
        checkpoint_location (str): Path to the checkpoint file.

    Returns:
        list[Task]: A list of instantiated Task objects from the checkpoint's tasks_config.
    """
    checkpoint: MLIPInferenceCheckpoint = torch.load(
        checkpoint_location, map_location="cpu", weights_only=False
    )
    return [
        hydra.utils.instantiate(task_config) for task_config in checkpoint.tasks_config
    ]


@contextmanager
def tf32_context_manager():
    # Store the original settings
    original_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_float32_matmul_precision = torch.get_float32_matmul_precision()
    try:
        # Set the desired settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        # Revert to the original settings
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn
        torch.set_float32_matmul_precision(original_float32_matmul_precision)


def update_configs(original_config, new_config):
    updated_config = deepcopy(original_config)
    for k, v in new_config.items():
        is_dict_config = (isinstance(v, (dict, DictConfig))) and (
            isinstance(updated_config[k], (dict, DictConfig))
        )
        if is_dict_config and k in updated_config:
            updated_config[k] = update_configs(updated_config[k], v)
        else:
            updated_config[k] = v
    return updated_config
