"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairchem.core.calculate._batch import InferenceBatcher
    from fairchem.core.calculate._ray_inference_cluster import (
        get_local_inference_raycluster,
        get_slurm_inference_raycluster,
    )
    from fairchem.core.calculate.ase_calculator import (
        FAIRChemCalculator,
        FormationEnergyCalculator,
    )
    from fairchem.core.calculate.dftd3_calculator import DFTD3Calculator
    from fairchem.core.components.batch_server import (
        setup_batch_predict_server,
        setup_multiplexed_batch_predict_server,
    )
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec

# Exports are imported on first access so that importing this package (or any of
# its submodules) does not pull in the Ray and inference stacks.
_LAZY_EXPORTS = {
    "FAIRChemCalculator": "fairchem.core.calculate.ase_calculator",
    "FormationEnergyCalculator": "fairchem.core.calculate.ase_calculator",
    "DFTD3Calculator": "fairchem.core.calculate.dftd3_calculator",
    "InferenceBatcher": "fairchem.core.calculate._batch",
    "InferenceSettings": "fairchem.core.units.mlip_unit.api.inference",
    "ModelSpec": "fairchem.core.units.mlip_unit.api.model_spec",
    "get_local_inference_raycluster": "fairchem.core.calculate._ray_inference_cluster",
    "get_slurm_inference_raycluster": "fairchem.core.calculate._ray_inference_cluster",
    "setup_batch_predict_server": "fairchem.core.components.batch_server",
    "setup_multiplexed_batch_predict_server": "fairchem.core.components.batch_server",
}

__all__ = [
    "FAIRChemCalculator",
    "FormationEnergyCalculator",
    "DFTD3Calculator",
    "InferenceBatcher",
    "InferenceSettings",
    "ModelSpec",
    "get_local_inference_raycluster",
    "get_slurm_inference_raycluster",
    "setup_batch_predict_server",
    "setup_multiplexed_batch_predict_server",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
