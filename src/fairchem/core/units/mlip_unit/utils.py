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
    from collections.abc import Mapping

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


def expand_mix_csd_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Zero-pad a pretrained ``mix_csd.weight`` to the model's solvent-enabled shape.

    When finetuning a pretrained checkpoint with ``use_solvent_embedding=True``, the
    backbone's ``mix_csd`` linear layer gains a trailing ``sphere_channels``-wide input
    block (the solvent embedding is concatenated last). The pretrained weight is copied
    into the leading columns and the new block is zero-initialized, so the solvent term
    contributes nothing at step 0 (an identity warm-start). The prefix-copy is valid
    because new embedding blocks (dataset, solvent) are only ever appended, so the
    pretrained blocks are always a prefix of the wider ordering.

    This is a no-op when shapes already match (no solvent embedding, or an
    already-finetuned checkpoint whose ``mix_csd`` is already wide).

    Args:
        model: The freshly instantiated model the weights will be loaded into.
        state_dict: The checkpoint state dict.

    Returns:
        A new state dict with any narrower ``mix_csd.weight`` zero-padded to match.
    """
    model_sd = model.state_dict()
    new_sd = dict(state_dict)
    for key, ckpt_w in state_dict.items():
        if not key.endswith("mix_csd.weight"):
            continue
        model_w = model_sd.get(key)
        if model_w is None or model_w.shape == ckpt_w.shape:
            continue
        padded = model_w.new_zeros(model_w.shape)
        padded[:, : ckpt_w.shape[1]] = ckpt_w
        new_sd[key] = padded
    return new_sd


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
        state_dict = expand_mix_csd_state_dict(model, checkpoint.model_state_dict)
        load_state_dict(model, state_dict, strict=strict)

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
def tf32_context_manager(tf32: bool):
    """
    Temporarily configure and restore TF32-related precision settings.

    Args:
        tf32: Whether to enable TF32 for float32 matmul and cuDNN operations.
    """
    original_precision = torch.get_float32_matmul_precision()
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    configured_precision = "high" if tf32 else "highest"
    try:
        torch.backends.cudnn.allow_tf32 = tf32
        if configured_precision != original_precision:
            torch.set_float32_matmul_precision(configured_precision)
        yield
    finally:
        if configured_precision != original_precision:
            torch.set_float32_matmul_precision(original_precision)
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
