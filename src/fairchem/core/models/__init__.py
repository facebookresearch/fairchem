"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.common.utils import resolve_class
from fairchem.core.models.base import BackboneInterface, HeadInterface, HydraModel
from fairchem.core.models.escaip.EScAIP import (
    EScAIPBackbone,
    EScAIPDirectForceHead,
    EScAIPEnergyHead,
    EScAIPGradientEnergyForceStressHead,
)
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

MODEL_REGISTRY: dict[str, type] = {
    "hydra": HydraModel,
    "EScAIP_backbone": EScAIPBackbone,
    "EScAIP_direct_force_head": EScAIPDirectForceHead,
    "EScAIP_energy_head": EScAIPEnergyHead,
    "EScAIP_grad_energy_force_stress_head": EScAIPGradientEnergyForceStressHead,
    "escnmd_backbone": eSCNMDBackbone,
    "escnmd_moe_backbone": eSCNMDMoeBackbone,
}


def get_model_class(name: str) -> type:
    """Resolve a model class from its name or fully-qualified path.

    Args:
        name: Model name (e.g., "hydra") or full path
            (e.g., "fairchem.core.models.base.HydraModel")

    Returns:
        The model class
    """
    return resolve_class(name, MODEL_REGISTRY, "model")


__all__ = [
    "BackboneInterface",
    "HeadInterface",
    "HydraModel",
    "EScAIPBackbone",
    "EScAIPDirectForceHead",
    "EScAIPEnergyHead",
    "EScAIPGradientEnergyForceStressHead",
    "eSCNMDBackbone",
    "eSCNMDMoeBackbone",
    "MODEL_REGISTRY",
    "get_model_class",
]
