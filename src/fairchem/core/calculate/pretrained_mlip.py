"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Literal

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from fairchem.core import calculate
from fairchem.core._config import CACHE_DIR
from fairchem.core.units.mlip_unit import MLIPPredictUnit, load_predict_unit

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import InferenceSettings


@dataclass
class HuggingFaceCheckpoint:
    filename: str
    repo_id: Literal["facebook/UMA"]
    subfolder: str | None = None  # specify a hf repo subfolder
    revision: str | None = None  # specify a version tag, branch, commit hash
    atom_refs: dict | None = None  # specify an isolated atomic reference


@dataclass
class PretrainedModels:
    checkpoints: dict[str, HuggingFaceCheckpoint]


with (resources.files(calculate) / "pretrained_models.json").open("rb") as f:
    _MODEL_CKPTS = PretrainedModels(
        checkpoints={
            model_name: HuggingFaceCheckpoint(**hf_kwargs)
            for model_name, hf_kwargs in json.load(f).items()
        }
    )

available_models = tuple(_MODEL_CKPTS.checkpoints.keys())


def pretrained_checkpoint_path_from_name(model_name: str):
    try:
        model_checkpoint = _MODEL_CKPTS.checkpoints[model_name]
    except KeyError as err:
        raise KeyError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        ) from err
    checkpoint_path = hf_hub_download(
        filename=model_checkpoint.filename,
        repo_id=model_checkpoint.repo_id,
        subfolder=model_checkpoint.subfolder,
        revision=model_checkpoint.revision,
        cache_dir=CACHE_DIR,
    )
    return checkpoint_path


def get_predict_unit(
    model_name: str,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    device: Literal["cuda", "cpu"] | None = None,
    cache_dir: str = CACHE_DIR,
) -> MLIPPredictUnit:
    """
    Retrieves a prediction unit for a specified model.

    Args:
        model_name: Name of the model to load from available pretrained models.
        inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
            (optimized for speed but requires fixed atomic composition). Advanced use cases can
            use a custom InferenceSettings object.
        overrides: Optional dictionary of settings to override default inference settings.
        device: Optional torch device to load the model onto. If None, uses the default device.
        cache_dir: Path to folder where model files will be stored. Default is "~/.cache/fairchem"

    Returns:
        An initialized MLIPPredictUnit ready for making predictions.

    Raises:
        KeyError: If the specified model_name is not found in available models.
    """
    if model_name == "uma-sm":
        raise NotImplementedError(
            "uma-sm has been renamed to 'uma-s-1', please update and try again."
        )
    try:
        model_checkpoint = _MODEL_CKPTS.checkpoints[model_name]
    except KeyError as err:
        raise KeyError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        ) from err
    checkpoint_path = hf_hub_download(
        filename=model_checkpoint.filename,
        repo_id=model_checkpoint.repo_id,
        subfolder=model_checkpoint.subfolder,
        revision=model_checkpoint.revision,
        cache_dir=cache_dir,
    )
    atom_refs = get_isolated_atomic_energies(model_name, cache_dir)
    return load_predict_unit(
        checkpoint_path, inference_settings, overrides, device, atom_refs
    )


def get_isolated_atomic_energies(model_name: str, cache_dir: str = CACHE_DIR) -> dict:
    """
    Retrieves the isolated atomic energies for use with single atom systems into the CACHE_DIR

    Args:
        model_name: Name of the model to load from available pretrained models.
        cache_dir: Path to folder where files will be stored. Default is "~/.cache/fairchem"
    Returns:
        Atomic element reference data

    Raises:
        KeyError: If the specified model_name is not found in available models.
    """
    model_checkpoint = _MODEL_CKPTS.checkpoints[model_name]
    atomic_refs_path = hf_hub_download(
        filename=model_checkpoint.atom_refs["filename"],
        repo_id=model_checkpoint.repo_id,
        subfolder=model_checkpoint.atom_refs["subfolder"],
        revision=model_checkpoint.revision,
        cache_dir=cache_dir,
    )
    return OmegaConf.load(atomic_refs_path)
