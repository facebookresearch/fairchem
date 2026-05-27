"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.calculate._ray_inference_cluster import (
    get_local_fairchem_inference_raycluster,
    get_slurm_fairchem_inference_raycluster,
)

__all__ = [
    "get_local_fairchem_inference_raycluster",
    "get_slurm_fairchem_inference_raycluster",
]
