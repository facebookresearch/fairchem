"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Batch inference utilities for FAIRChem models via Ray Serve.

This package provides:
- BatchPredictServer: Batched inference server deployment
- setup_batch_predict_server: Helper to deploy the server
- wait_for_serve_ready: Wait for the deployment to be ready
- get_ray_connection_info: Read connection info from head.json
"""

from __future__ import annotations

from fairchem.core.units.mlip_unit.batch.batch_predict_server import (
    BatchPredictServer,
    get_ray_connection_info,
    setup_batch_predict_server,
    wait_for_serve_ready,
)

__all__ = [
    "BatchPredictServer",
    "setup_batch_predict_server",
    "wait_for_serve_ready",
    "get_ray_connection_info",
]
