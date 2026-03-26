"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Batch inference utilities for FAIRChem models via Ray Serve.

This package provides:
- BatchPredictServer: Simple batched inference for a single model
- FAIRChemInferenceServer: Multiplexed server with model caching
- FAIRChemInferenceClient: Client for submitting inference requests
- RayServeMLIPUnit: MLIPPredictUnit-compatible wrapper for Ray Serve
"""
from __future__ import annotations

from fairchem.core.units.mlip_unit.batch.batch_predict_server import (
    BatchPredictServer,
    setup_batch_predict_server,
)
from fairchem.core.units.mlip_unit.batch.inference_client import (
    FAIRChemInferenceClient,
    get_inference_client,
)
from fairchem.core.units.mlip_unit.batch.inference_server import (
    FAIRChemInferenceServer,
    start_serve,
    wait_for_serve_ready,
)
from fairchem.core.units.mlip_unit.batch.rayserve_mlip_unit import (
    RayServeMLIPUnit,
    get_ray_connection_info,
)

__all__ = [
    "BatchPredictServer",
    "setup_batch_predict_server",
    "FAIRChemInferenceServer",
    "start_serve",
    "wait_for_serve_ready",
    "FAIRChemInferenceClient",
    "get_inference_client",
    "RayServeMLIPUnit",
    "get_ray_connection_info",
]
