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

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.models.uma.compat import apply_uma_compat_fixups

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def get_backbone_class_from_checkpoint(
    checkpoint: MLIPInferenceCheckpoint,
) -> type:
    """Extract the backbone class from a checkpoint's config."""
    backbone_config = checkpoint.model_config.get("backbone", {})
    backbone_model_name = backbone_config.get("model")

    if backbone_model_name is None:
        raise ValueError("Cannot determine backbone class from checkpoint config")

    return registry.get_model_class(backbone_model_name)


def load_inference_model(
    checkpoint_location: str,
    overrides: dict | None = None,
    use_ema: bool = False,
    return_checkpoint: bool = True,
    strict: bool = True,
    preloaded_checkpoint: MLIPInferenceCheckpoint | None = None,
) -> tuple[torch.nn.Module, MLIPInferenceCheckpoint] | torch.nn.Module:
    if preloaded_checkpoint is not None:
        checkpoint = preloaded_checkpoint
    else:
        checkpoint = torch.load(
            checkpoint_location, map_location="cpu", weights_only=False
        )

    apply_uma_compat_fixups(checkpoint, checkpoint_location=checkpoint_location)

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

        load_state_dict(model, matched_dict, strict=strict)
    else:
        load_state_dict(model, checkpoint.model_state_dict, strict=strict)

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


def get_model_float32_matmul_precision(model: torch.nn.Module | None) -> str | None:
    """
    Get a model's float32 matmul policy through common module wrappers.

    Args:
        model: A model, potentially wrapped by DDP, FSDP, or AveragedModel.

    Returns:
        The backbone's configured precision, or None if it has no policy.
    """
    visited = set()
    while model is not None and id(model) not in visited:
        visited.add(id(model))
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            return getattr(backbone, "float32_matmul_precision", None)
        model = getattr(model, "module", None)
    return None


@contextmanager
def float32_matmul_precision_context(precision: str | None):
    """
    Temporarily apply and restore the process float32 matmul precision.

    Args:
        precision: A value accepted by torch.set_float32_matmul_precision, or
            None to leave the current setting unchanged.
    """
    if precision is None:
        yield
        return

    original_precision = torch.get_float32_matmul_precision()
    try:
        if precision != original_precision:
            torch.set_float32_matmul_precision(precision)
        yield
    finally:
        if precision != original_precision:
            torch.set_float32_matmul_precision(original_precision)


@contextmanager
def tf32_context_manager():
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        with float32_matmul_precision_context("high"):
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn


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
