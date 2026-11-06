"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from fairchem.core.regression_suite.spec.hashing import json_value
from fairchem.core.regression_suite.spec.schema import CommandError
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings

if TYPE_CHECKING:
    from collections.abc import Mapping


PROFILE_ALIASES = {
    "umas-fast": "multigpu",
    "tf32": "tf32_eager",
    "fp32": "fp32_eager",
    "fp64": "fp64_eager",
}


CONFIGS = "fairchem.core.regression_suite.configs"


def resolve_profile(name: str) -> str:
    """
    Resolve a short profile alias or profile name to its config name.
    """
    profile = PROFILE_ALIASES.get(name, name.replace("-", "_"))
    configs = importlib.resources.files(f"{CONFIGS}.profile")
    if not (configs / f"{profile}.yaml").is_file():
        names = {path.name.removesuffix(".yaml") for path in configs.iterdir()}
        choices = sorted(
            {*PROFILE_ALIASES, *(n.replace("_", "-") for n in names if "." not in n)}
        )
        raise CommandError(f"unknown profile {name!r}; choose one of {choices}")
    return profile


def profile_config(profile: str) -> DictConfig:
    """
    Load one profile config by name.
    """
    return OmegaConf.load(
        importlib.resources.files(f"{CONFIGS}.profile") / f"{profile}.yaml"
    )


def unsupported_suites_for_profile(profile: str) -> frozenset[str]:
    """
    Return the suites that a profile config or artifact label cannot run.
    """
    config = profile_config(profile.replace("-", "_"))
    return frozenset(config.get("unsupported_suites", ()))


def json_settings(settings: InferenceSettings) -> dict[str, Any]:
    """
    Return inference settings as the JSON values a run's identity records.
    """
    return {
        key: json_value(value)
        for key, value in settings.to_omegaconf().items()
        if key != "_target_"
    }


def build_inference_settings(
    profile: DictConfig,
    overrides: Mapping[str, Any],
) -> InferenceSettings:
    """
    Build a profile's settings: a named fairchem preset (``preset:``) with the
    profile's ``inference_settings`` as overrides, or the explicit
    ``inference_settings`` alone; a suite's ``overrides`` apply last.
    """
    explicit = (
        OmegaConf.to_container(profile.inference_settings, resolve=True)
        if profile.get("inference_settings") is not None
        else {}
    )
    if "preset" in profile:
        explicit.pop("_target_", None)
        return dataclasses.replace(
            guess_inference_settings(str(profile.preset)),
            **explicit,
            **overrides,
        )
    settings_config = OmegaConf.merge(
        OmegaConf.create(explicit),
        OmegaConf.create(overrides),
    )
    settings = hydra.utils.instantiate(settings_config, _convert_="all")
    if not isinstance(settings, InferenceSettings):
        raise TypeError("profile must instantiate InferenceSettings")
    return settings
