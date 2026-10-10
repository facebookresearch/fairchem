"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
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
    form_elem_refs: dict | None = None  # specify a form elemental reference


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


class UnknownCheckpointError(KeyError, ValueError):
    """
    Raised when a string is neither a registered model name nor a checkpoint file.

    Subclasses both KeyError and ValueError because callers historically raised
    one or the other for this condition.
    """


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """
    A local checkpoint path and the reference energies that accompany it.

    Reference energies are only populated for registered model names.
    """

    path: str
    atom_refs: dict | None = None
    form_elem_refs: dict | None = None


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


def resolve_checkpoint(
    name_or_path: str, cache_dir: str = CACHE_DIR
) -> ResolvedCheckpoint:
    """
    Resolve a registered model name or checkpoint file path to a local checkpoint.

    Registered names take precedence over files with the same name. Registered
    names download the checkpoint and its reference energies; file paths are
    returned as-is without reference energies.

    Args:
        name_or_path: A name from available_models or a path to a checkpoint file.
        cache_dir: Folder where reference energy files are stored.

    Returns:
        The resolved checkpoint path and any reference energies.

    Raises:
        UnknownCheckpointError: If name_or_path is neither a registered model name
            nor an existing file.
    """
    if name_or_path in _MODEL_CKPTS.checkpoints:
        checkpoint_path = pretrained_checkpoint_path_from_name(name_or_path)
        atom_refs = get_reference_energies(name_or_path, "atom_refs", cache_dir)
        if _MODEL_CKPTS.checkpoints[name_or_path].form_elem_refs is not None:
            form_elem_refs = get_reference_energies(
                name_or_path, "form_elem_refs", cache_dir
            )["refs"]
        else:
            form_elem_refs = None
        return ResolvedCheckpoint(checkpoint_path, atom_refs, form_elem_refs)
    if os.path.isfile(name_or_path):
        return ResolvedCheckpoint(str(name_or_path))
    raise UnknownCheckpointError(
        f"Invalid model name or checkpoint path: Model '{name_or_path}' not found. "
        f"Available models: {available_models}"
    )


def get_predict_unit(
    model_name: str,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    device: Literal["cuda", "cpu"] | None = None,
    cache_dir: str = CACHE_DIR,
    workers: int = 1,
    seed: int = 41,
    gp_config=None,
) -> MLIPPredictUnit:
    """
    Retrieves a prediction unit for a specified model.

    Args:
        model_name: Name of the model to load from available pretrained models, or a
            path to a checkpoint file. Paths are loaded without reference energies.
        inference_settings: Settings for inference. Both "default" and "turbo" use the
            merge_mole + compile fast path, with automatic fallback if its fixed-input
            contract is broken. "turbo" additionally enables TF32. "batch" keeps MOLE
            unmerged for heterogeneous inputs. More advanced use cases can use a custom
            InferenceSettings object.
        overrides: Optional dictionary of settings to override default inference settings.
        device: Optional torch device to load the model onto. If None, uses the default device.
        cache_dir: Path to folder where model files will be stored. Default is "~/.cache/fairchem"
        workers: Number of parallel workers for prediction unit. Default is 1. If greater than 1,
            we will instantiate a ParallelMLIPPredictUnit instead of the normal predict unit.
        seed: Optional random seed for reproducibility. If provided, will set the random seed for
            Python's random module, NumPy, and PyTorch to ensure reproducible predictions.

    Returns:
        An initialized MLIPPredictUnit ready for making predictions.

    Raises:
        UnknownCheckpointError: If model_name is neither a registered model name nor
            an existing file. Subclasses KeyError and ValueError.
    """
    checkpoint = resolve_checkpoint(model_name, cache_dir)
    return load_predict_unit(
        checkpoint.path,
        inference_settings,
        overrides,
        device,
        checkpoint.atom_refs,
        checkpoint.form_elem_refs,
        workers,
        seed,
        gp_config=gp_config,
    )


def get_reference_energies(
    model_name: str,
    reference_type: Literal["atom_refs", "form_elem_refs"] = "atom_refs",
    cache_dir: str = CACHE_DIR,
) -> dict:
    """
    Retrieves the isolated atomic energies for use with single atom systems into the CACHE_DIR

    Args:
        model_name: Name of the model to load from available pretrained models.
        reference_type: Type of references file to download: atom_refs or bulk_refs.
        cache_dir: Path to folder where files will be stored. Default is "~/.cache/fairchem"
    Returns:
        Atomic or bulk phase element reference data

    Raises:
        KeyError: If the specified model_name is not found in available models.
    """
    model_checkpoint = _MODEL_CKPTS.checkpoints[model_name]
    file_data = getattr(model_checkpoint, reference_type)
    refs_path = hf_hub_download(
        filename=file_data["filename"],
        repo_id=model_checkpoint.repo_id,
        subfolder=file_data["subfolder"],
        revision=model_checkpoint.revision,
        cache_dir=cache_dir,
    )
    return OmegaConf.load(refs_path)
