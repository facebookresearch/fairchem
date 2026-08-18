"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
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


@dataclass(frozen=True)
class ModelSpec:
    """Typed configuration and deterministic identity for a multiplexed model."""

    checkpoint: str
    inference_settings: InferenceSettings | str = "default"
    device: str | None = None
    overrides: dict | None = None
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

    def resolve_device(self) -> str:
        """Resolve the device on the replica when the client leaves it unspecified."""
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def loader_settings(self) -> InferenceSettings:
        """Return typed inference settings matching this spec's identity snapshot."""
        return copy.deepcopy(self._loader_settings)

    def loader_overrides(self) -> dict | None:
        """Return model overrides matching this spec's identity snapshot."""
        return copy.deepcopy(self._loader_overrides)


class ModelSpecNotRegisteredError(KeyError):
    """Raised when a routed identity has no configuration on the replica."""
