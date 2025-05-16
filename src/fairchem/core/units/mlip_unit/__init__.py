"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    guess_inference_settings,
)
from fairchem.core.units.mlip_unit.mlip_unit import MLIPPredictUnit

if TYPE_CHECKING:
    from pathlib import Path


def load_predict_unit(
    path: str | Path,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    device: str = "cuda",
) -> MLIPPredictUnit:
    """Load a MLIPPredictUnit from a checkpoint file.

    Args:
        path: Path to the checkpoint file
        inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
            (optimized for speed but requires fixed atomic composition). Advanced use cases can
            use a custom InferenceSettings object.
        overrides: Optional dictionary of settings to override default inference settings.
        device: Optional torch device to load the model onto.

    Returns:
        A MLIPPredictUnit instance ready for inference
    """
    inference_settings = guess_inference_settings(inference_settings)
    overrides = overrides or {"backbone": {"always_use_pbc": False}}

    return MLIPPredictUnit(
        path,
        device=device,
        inference_settings=inference_settings,
        overrides={"backbone": {"always_use_pbc": False}},
    )
