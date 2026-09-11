"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import torch

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    guess_inference_settings,
)

# This module is deliberately a leaf: it depends only on ``api.inference``
# (itself a leaf) so that both ``components.batch_server`` and
# ``units.mlip_unit.predict`` can import ``ModelSpec`` at module level without
# forming an import cycle.

__all__ = ["ModelSpec", "ModelSpecNotRegisteredError"]


def _canonicalize_model_spec_value(value: Any) -> Any:
    """Convert nested model configuration values to stable JSON-compatible data."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_model_spec_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize_model_spec_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_canonicalize_model_spec_value(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def _resolve_source(checkpoint: str, source: str) -> Literal["path", "registry"]:
    """
    Resolve ``"auto"`` to the concrete loader that will be used.

    Resolving eagerly keeps ``"auto"`` from minting a second identity for a
    model that an explicit ``"path"``/``"registry"`` spec already names.

    Args:
        checkpoint: Checkpoint path or registered model name.
        source: The caller-supplied source.

    Returns:
        Either ``"path"`` or ``"registry"``.
    """
    if source != "auto":
        return source
    return "path" if os.path.isfile(checkpoint) else "registry"


def _canonicalize_checkpoint(checkpoint: str, source: str) -> str:
    """
    Collapse equivalent spellings of a local checkpoint path to one form.

    Only applied to files that actually exist locally, so remote URIs
    (``s3://...``) and registered model names are preserved verbatim rather
    than being mangled into a bogus absolute path.

    Args:
        checkpoint: Checkpoint path or registered model name.
        source: The already-resolved source.

    Returns:
        The realpath for an existing local file, else ``checkpoint`` unchanged.
    """
    if source == "path" and os.path.isfile(checkpoint):
        return os.path.realpath(checkpoint)
    return checkpoint


def _canonicalize_device(device: str | None) -> str | None:
    """
    Normalize a device string and reject invalid ones at construction time.

    ``None`` is preserved: it means "the replica chooses", which is a
    genuinely different request from pinning a device, so the two keep
    distinct identities.

    Args:
        device: Device string such as ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or
            ``None``.

    Returns:
        The normalized device string, or ``None``.

    Raises:
        ValueError: If ``device`` is not a valid torch device.
    """
    if device is None:
        return None
    try:
        return str(torch.device(device))
    except (RuntimeError, TypeError, ValueError) as err:
        raise ValueError(
            f"device must be a valid torch device, got {device!r}"
        ) from err


@dataclass(frozen=True)
class ModelSpec:
    """
    Typed configuration and deterministic identity for a multiplexed model.

    ``checkpoint``, ``device``, and ``source`` are canonicalized during
    construction and written back onto the (frozen) instance, so equivalent
    spellings of the same model collapse to a single ``model_id`` instead of
    each loading its own copy into a replica's bounded model cache.

    Because the canonicalized values are stored on the instance, the client's
    resolution is what travels with the request; a replica never re-resolves
    and so can never disagree about ``model_id``.
    """

    checkpoint: str
    inference_settings: InferenceSettings | str = "default"
    device: str | None = None
    overrides: dict | None = None
    # ``"auto"`` is an input convenience only; it is resolved to ``"path"`` or
    # ``"registry"`` in ``__post_init__`` and never observed afterwards.
    source: Literal["auto", "path", "registry"] = "auto"
    _canonical_config: dict[str, Any] = field(init=False, repr=False, compare=False)
    _loader_settings: InferenceSettings = field(init=False, repr=False, compare=False)
    _loader_overrides: dict | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("checkpoint must be a non-empty string")
        if self.source not in ("auto", "path", "registry"):
            raise ValueError(
                "source must be one of 'auto', 'path', or 'registry', "
                f"got {self.source!r}"
            )

        # Canonicalize identity-bearing fields before anything hashes them, and
        # write them back so the resolution travels with the pickled spec.
        source = _resolve_source(self.checkpoint, self.source)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "checkpoint", _canonicalize_checkpoint(self.checkpoint, source)
        )
        object.__setattr__(self, "device", _canonicalize_device(self.device))

        settings = copy.deepcopy(guess_inference_settings(self.inference_settings))
        overrides = (
            copy.deepcopy(self.overrides) if self.overrides is not None else None
        )
        object.__setattr__(self, "inference_settings", copy.deepcopy(settings))
        object.__setattr__(self, "overrides", copy.deepcopy(overrides))
        object.__setattr__(self, "_loader_settings", settings)
        object.__setattr__(self, "_loader_overrides", overrides)

        settings_config = settings.to_omegaconf()
        settings_config.pop("_target_", None)
        object.__setattr__(
            self,
            "_canonical_config",
            {
                "checkpoint": self.checkpoint,
                "inference_settings": _canonicalize_model_spec_value(settings_config),
                "device": self.device,
                "overrides": _canonicalize_model_spec_value(overrides or {}),
                "source": self.source,
            },
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return a copy of the stable configuration used to derive ``model_id``."""
        return copy.deepcopy(self._canonical_config)

    @cached_property
    def model_id(self) -> str:
        """Return a readable deterministic identity token for Ray Serve routing."""
        canonical_json = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
        short_name = re.sub(
            r"[^A-Za-z0-9._-]+", "-", os.path.basename(self.checkpoint)
        ).strip("-._")
        short_name = (short_name or "model")[:48]
        return f"{short_name}-{digest}"

    def __hash__(self) -> int:
        """
        Hash by ``model_id`` so specs can key sets and dicts.

        The dataclass-generated ``__hash__`` would hash the tuple of compared
        fields, which contains an unhashable ``dict`` (``overrides``) and an
        unhashable ``InferenceSettings``, making every instance unhashable
        despite ``frozen=True`` advertising the opposite.

        ``model_id`` is derived from exactly the fields ``__eq__`` compares, so
        equal specs always hash equal.
        """
        return hash(self.model_id)

    def resolve_device(self) -> str:
        """
        Resolve the device to load on, from the replica's point of view.

        Args:
            None.

        Returns:
            The device string to hand to the model loader.

        Raises:
            RuntimeError: If this spec pins a CUDA device but the replica has
                no CUDA available. Falling back to CPU here would "work" while
                running orders of magnitude slower, which is far harder to
                diagnose than an immediate failure.
        """
        if self.device is not None:
            if (
                torch.device(self.device).type == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    f"ModelSpec pins device={self.device!r} but this replica has "
                    "no CUDA device. Ray sets CUDA_VISIBLE_DEVICES per replica, "
                    "so this usually means the deployment was created with "
                    "num_gpus=0 -- pass num_gpus=1 to "
                    "setup_multiplexed_batch_predict_server."
                )
            return self.device

        if torch.cuda.is_available():
            return "cuda"

        # Unpinned specs are allowed to land on CPU, but say so: a GPU-sized
        # workload silently running on CPU is the single most expensive way
        # for this to go wrong.
        logging.warning(
            "ModelSpec %s left device unspecified and this replica has no CUDA "
            "device, so the model will load on CPU. If the cluster has GPUs, "
            "the deployment was likely created with num_gpus=0.",
            self.model_id,
        )
        return "cpu"

    def loader_settings(self) -> InferenceSettings:
        """Return typed inference settings matching this spec's identity snapshot."""
        return copy.deepcopy(self._loader_settings)

    def loader_overrides(self) -> dict | None:
        """Return model overrides matching this spec's identity snapshot."""
        return copy.deepcopy(self._loader_overrides)


class ModelSpecNotRegisteredError(KeyError):
    """Raised when a routed identity has no configuration on the replica."""
